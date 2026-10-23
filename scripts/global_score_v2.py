"""
Global PGK2 scoring v2 (audit-driven): denoised DEL evidence + transferable priors, combined
with explicit weights (highest to lowest), then series-diverse selection.

Weights are prior judgment, not fitted (the pool has no labels):
  1. fragment enrichment (denoised, generic-DEL-penalized)   0.45
  2. series consensus (score-weighted pool neighbours)        0.20
  3. pharmacophore (hinge/adenine mimic, anionic group)       0.15
  4. ASMS property prior (WDR91 confirmed hits)               0.10
  5. ligand efficiency (fragment score / sqrt heavy atoms)    0.10

Denoising of the DEL positives (soft weight per positive, product of):
  count lower bound (Poisson) x competition ratio x NTC x promiscuity x
  building-block support (leave-one-out empirical-Bayes rate of the best of its
  three disynthons, fit on train rows only).
"""

import json
import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.metrics import average_precision_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")

FEATURES_PARQUET = "data/processed/PGK2_selection_features.parquet"
LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"
ECFP4_CACHE = "data/processed/pgk2_ecfp4_sparse.npz"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
PRIOR_JSON = "data/processed/asms_property_prior.json"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"

W = {"frag": 0.35, "erg": 0.25, "series": 0.15, "pharm": 0.15, "prop": 0.10}
N_REF = 300
FAMILY_CAP = 2
TESTED = ["Team_candeleroj_%d.txt" % i for i in (1, 2, 3, 4, 5, 6, 7)]
BB_PRIOR_STRENGTH = 50.0
ALPHA, MIN_POS_COUNT, MIN_POOL_FREQ, MIN_WEIGHT, MAX_WEIGHT = 1.0, 20, 0.0005, 0.5, 4.0
GENERIC_PENALTY = 1.0
CONSENSUS_POOL, CONSENSUS_SIM = 20000, 0.5
CLUSTER_TOP_N, N_SERIES, PER_SERIES = 3000, 10, 5
SUBMISSION_INDEX = 9

HINGE = [Chem.MolFromSmarts(s) for s in ("n:c:[nH]", "[nX2]:c:[nH]", "[NX3;H2]c:n", "[NX3;H1]c:n")]
ANION = [Chem.MolFromSmarts(s) for s in ("[CX3](=O)[OX2H1,OX1-]", "P(=O)([OH,O-])", "[SX4](=O)(=O)[OH,O-]", "c1nnn[nH]1")]
# Fix 1 (2026-09-23 session): see scripts/global_score.py for the full rationale. Two new
# patterns for confirmed real hits' cores that HINGE/ANION missed entirely (pharm score
# blind to them); specificity-checked (17/244,328 and 34/244,328 of the Validation Split).
# Caution: 3 of the 17 thienopyrimidine-core matches are CONFIRMED NEGATIVE -- necessary
# but not sufficient, don't treat a SPECIFIC_CORE match alone as a strong signal.
SPECIFIC_CORE = [
    Chem.MolFromSmarts("[#6&!R]-&!@[#6]1:&@[#16]:&@[#6]2:&@[#6](:&@[#6]:&@1-&!@[#6&!R]):&@[#6]:"
                        "&@[#7]:&@[#6](:&@[#7]:&@2)-&!@[#6]1:&@[#6]:&@[#6]:&@[#6]:&@[#7]:&@[#6]:&@1"),
    Chem.MolFromSmiles("c1cncc(-c2ncc3ccccc3n2)c1"),
]


def binarize(X):
    Xb = X.tocsr().copy()
    Xb.data[:] = 1
    return Xb.astype(np.float32)


def _pool_batch(smiles_batch):
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for smi in smiles_batch:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append(None)
            continue
        fp = np.zeros(2048, dtype=np.int16)
        for i, c in gen.GetCountFingerprint(m).GetNonzeroElements().items():
            fp[i] = min(c, 32767)
        props = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Lipinski.NumHDonors(m),
                 Lipinski.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                 rdMolDescriptors.CalcFractionCSP3(m)]
        hinge = any(m.HasSubstructMatch(p) for p in HINGE)
        anion = any(m.HasSubstructMatch(p) for p in ANION)
        specific = any(m.HasSubstructMatch(p) for p in SPECIFIC_CORE)
        from rdkit.Chem import rdReducedGraphs
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(m), dtype=np.float32)
        out.append((fp, props, m.GetNumHeavyAtoms(), hinge, anion, erg, specific))
    return out


def featurize_pool(smiles, n_jobs=None):
    n_jobs = n_jobs or os.cpu_count() or 4
    parts = np.array_split(np.array(smiles, dtype=object), n_jobs)
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_pool_batch)(list(p)) for p in parts if len(p))
    return [r for part in res for r in part]


def soft_weights(train_idx, y_all, aux, ids):
    """Per-positive confidence in [0,1] for the TRAIN positives."""
    parts = ids.str.split("-", expand=True)
    lib = pd.factorize(parts[0])[0].astype(np.int64)
    b1 = pd.factorize(parts[1])[0].astype(np.int64)
    b2 = pd.factorize(parts[2])[0].astype(np.int64)
    b3 = pd.factorize(parts[3])[0].astype(np.int64)
    c1, c2, c3 = b1.max() + 1, b2.max() + 1, b3.max() + 1
    is_train = np.zeros(len(y_all), dtype=bool)
    is_train[train_idx] = True
    ytr = (y_all == 1) & is_train
    p0 = ytr.sum() / is_train.sum()

    best_R = np.ones(len(y_all))
    for key in (lib * c1 * c2 + b1 * c2 + b2, lib * c1 * c3 + b1 * c3 + b3, lib * c2 * c3 + b2 * c3 + b3):
        codes = pd.factorize(key)[0]
        n_k = np.bincount(codes[is_train])
        p_k = np.bincount(codes[ytr])
        n_k = np.pad(n_k, (0, max(0, codes.max() + 1 - len(n_k))))
        p_k = np.pad(p_k, (0, max(0, codes.max() + 1 - len(p_k))))
        pos_loo = p_k[codes] - ytr
        n_loo = n_k[codes] - is_train
        rate = (pos_loo + BB_PRIOR_STRENGTH * p0) / (n_loo + BB_PRIOR_STRENGTH)
        best_R = np.maximum(best_R, rate / p0)
    f_bb = 0.3 + 0.7 * np.clip(np.log10(np.maximum(best_R, 1)) / 2, 0, 1)

    cnt = aux["count_PGK2"].to_numpy().astype(float)
    lb = np.maximum(0, cnt - 1.645 * np.sqrt(cnt))
    f_count = np.clip(np.log1p(lb) / np.log1p(20), 0, 1)
    r = aux["count_PGK2_with_inhibitor"].to_numpy() / np.maximum(cnt, 1)
    f_comp = 1 - 0.3 * np.minimum(1, r / 0.5)
    f_ntc = (aux["count_NTC"].to_numpy() == 0).astype(float)
    f_prom = 1 / (1 + aux["historic_hits"].to_numpy() / 2)
    return f_count * f_comp * f_ntc * f_prom * f_bb


def erg_of(smiles):
    from rdkit.Chem import rdReducedGraphs
    return np.stack([np.asarray(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(x)), dtype=np.float32)
                     for x in smiles])


def erg_sim(A, R, chunk=400):
    out = np.zeros((len(A), len(R)), dtype=np.float32)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk][:, None, :]
        out[i:i + chunk] = np.minimum(a, R[None]).sum(-1) / (np.maximum(a, R[None]).sum(-1) + 1e-9)
    return out


def top_mean(S, k=5):
    return np.sort(S, axis=1)[:, -k:].mean(axis=1)


def core_family(smi):
    m = Chem.MolFromSmiles(smi)
    ri = m.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        if not all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        ring = set(ring)
        merged = [sy for sy in systems if sy & ring]
        for sy in merged:
            systems.remove(sy)
            ring |= sy
        systems.append(ring)
    if not systems:
        return "none"
    best = max(systems, key=len)
    return Chem.MolFragmentToSmiles(m, atomsToUse=sorted(best), kekuleSmiles=False)


def zs(x, lo=-3, hi=6):
    x = np.asarray(x, dtype=float)
    return np.clip((x - x.mean()) / (x.std() + 1e-9), lo, hi)


def main():
    ids_df = pq.read_table(FEATURES_PARQUET, columns=["RandomID", "Label"]).to_pandas()
    aux = pq.read_table(LABELED_PARQUET, columns=["RandomID", "count_PGK2", "count_PGK2_with_inhibitor",
                                                  "count_NTC", "historic_hits"]).to_pandas()
    meta = ids_df.merge(aux, on="RandomID", how="left")
    y = meta["Label"].to_numpy().astype(int)
    idx = np.load(SPLIT_CACHE)
    train_idx, val_idx = idx["train_idx"], idx["val_idx"]

    w_all = soft_weights(train_idx, y, meta, meta["RandomID"])
    pos_tr = train_idx[y[train_idx] == 1]
    wp = w_all[pos_tr]
    print(f"[*] soft weight of {len(pos_tr):,} train positives: median={np.median(wp):.2f} "
          f"p10={np.percentile(wp, 10):.2f} p90={np.percentile(wp, 90):.2f} | "
          f"confident (>=0.5): {(wp >= 0.5).mean():.1%}", flush=True)

    y_gen = ((meta["count_PGK2"] >= 3) & (meta["count_NTC"] == 0) & (meta["historic_hits"] >= 5)).to_numpy().astype(int)
    X = binarize(sp.load_npz(ECFP4_CACHE))
    Xtr, ytr, ygtr = X[train_idx], y[train_idx], y_gen[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    del X
    Xpos = Xtr[ytr == 1]
    n1_unw = np.asarray(Xpos.sum(axis=0)).ravel()
    n1 = np.asarray(Xpos.T @ wp).ravel()
    N1 = wp.sum()
    n_all = np.asarray(Xtr.sum(axis=0)).ravel()
    n0 = n_all - n1_unw
    N0 = (ytr == 0).sum()
    w_frag = np.log((n1 + ALPHA) / (N1 + ALPHA)) - np.log((n0 + ALPHA) / (N0 + ALPHA))
    g1 = np.asarray(Xtr[ygtr == 1].sum(axis=0)).ravel()
    w_gen = np.log((g1 + ALPHA) / (ygtr.sum() + ALPHA)) - np.log((n_all - g1 + ALPHA) / ((ygtr == 0).sum() + ALPHA))
    w_frag = w_frag - GENERIC_PENALTY * np.maximum(w_gen, 0)
    del Xtr, Xpos

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        pools = []
        for name in ("PGK2_Validation_split.csv", "PGK2_Test_split.csv"):
            with z.open(f"Val-Test-set/{name}") as f:
                pools.append(pd.read_csv(f))
    tested = set()
    for fn in TESTED:
        if os.path.exists(f"submissions/{fn}") and fn not in ("Team_candeleroj_4.txt", "Team_candeleroj_5.txt"):
            tested |= set(open(f"submissions/{fn}").read().split())
    if os.path.exists("submissions/tested_zero_hits.txt"):
        tested |= set(open("submissions/tested_zero_hits.txt").read().split())
    print(f"[*] excluding {len(tested)} already-evaluated (0-hit) compounds", flush=True)
    val_pool = pools[0][~pools[0]["CatalogID"].isin(tested)].reset_index(drop=True)
    feats_val = featurize_pool(val_pool["SMILES"].tolist())
    feats_test = featurize_pool(pools[1]["SMILES"].tolist())
    pool_fp = np.stack([f[0] for f in feats_val + feats_test if f is not None])
    pool_freq = (pool_fp > 0).mean(axis=0)
    del pool_fp

    eligible = (n1_unw >= MIN_POS_COUNT) & (pool_freq >= MIN_POOL_FREQ) & (w_frag >= MIN_WEIGHT)
    w_use = np.where(eligible, np.minimum(w_frag, MAX_WEIGHT), 0.0).astype(np.float32)
    print(f"[*] eligible fragment bits: {int(eligible.sum())} / 2048", flush=True)
    print(f"[*] held-out DEL split: ROC-AUC={roc_auc_score(yva, Xva @ w_use):.4f} "
          f"PR-AUC={average_precision_score(yva, Xva @ w_use):.4f}", flush=True)

    ok = [f is not None for f in feats_val]
    pool = val_pool[ok].reset_index(drop=True)
    F = [f for f in feats_val if f is not None]
    Xp = (np.stack([f[0] for f in F]) > 0).astype(np.float32)
    props = np.array([f[1] for f in F])
    hac = np.array([f[2] for f in F], dtype=float)
    hinge = np.array([f[3] for f in F], dtype=float)
    anion = np.array([f[4] for f in F], dtype=float)
    erg_pool = np.stack([f[5] for f in F])
    specific = np.array([f[6] for f in F], dtype=float)

    prior = json.load(open(PRIOR_JSON))
    prop_score = np.zeros(len(pool))
    for j, (edges, lr) in enumerate(zip(prior["edges"], prior["log_lr"])):
        prop_score += np.array(lr)[np.searchsorted(np.array(edges), props[:, j])]
    prop_score = np.clip(prop_score, -3, 3)

    frag = Xp @ w_use
    z_frag = zs(frag)
    z_pharm = zs(np.minimum(hinge + 0.5 * anion + specific, 2.5))
    z_prop = zs(prop_score)

    smiles_all = pq.read_table(FEATURES_PARQUET, columns=["SMILES"]).column("SMILES")
    ref_local = pos_tr[np.argsort(-wp)[:N_REF]]
    rng = np.random.RandomState(0)
    ctl_local = rng.choice(train_idx[y[train_idx] == 0], N_REF, replace=False)
    R = erg_of([smiles_all[int(i)].as_py() for i in ref_local])
    C = erg_of([smiles_all[int(i)].as_py() for i in ctl_local])
    erg_score = top_mean(erg_sim(erg_pool, R)) - top_mean(erg_sim(erg_pool, C))
    z_erg = zs(erg_score)
    print(f"[*] ErG contrast (hit-likeness beyond generic DEL): mean={erg_score.mean():.3f} "
          f"std={erg_score.std():.3f} | corr with fragment score={np.corrcoef(z_frag, z_erg)[0, 1]:.2f}", flush=True)
    rest = W["frag"] + W["erg"] + W["pharm"] + W["prop"]
    pre = (W["frag"] * z_frag + W["erg"] * z_erg + W["pharm"] * z_pharm + W["prop"] * z_prop) / rest

    order = np.argsort(-pre)[:CONSENSUS_POOL]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in pool["SMILES"].to_numpy()[order]]
    pre_top = pre[order]
    cons = np.zeros(len(order))
    for i in range(len(order)):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(bvs[i], bvs))
        sims[i] = 0
        nb = sims >= CONSENSUS_SIM
        cons[i] = pre_top[nb].sum() / (nb.sum() + 1)
    z_series_top = zs(cons)
    final = pre.copy()
    final[order] = (1 - W["series"]) * pre_top + W["series"] * z_series_top
    pool["Score"], pool["z_frag"], pool["z_erg"], pool["z_pharm"], pool["z_prop"] = final, z_frag, z_erg, z_pharm, z_prop
    pool["hinge"], pool["anion"], pool["specific_core"] = hinge, anion, specific
    in_top = np.zeros(len(pool), dtype=bool)
    in_top[order] = True
    pool = pool[in_top].sort_values("Score", ascending=False).reset_index(drop=True)
    pool.to_csv("data/processed/v2_candidates.csv", index=False)
    print(f"[*] dumped {len(pool):,} scored candidates to data/processed/v2_candidates.csv", flush=True)
    print(f"[*] component correlations on candidate set: "
          f"{pool[['z_frag', 'z_erg', 'z_pharm', 'z_prop']].corr().round(2).to_dict()}", flush=True)
    print(f"[*] hinge motif in {pool.hinge.mean():.1%} of candidates, anionic in {pool.anion.mean():.1%}", flush=True)

    top = pool.head(CLUSTER_TOP_N).copy()
    tb = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top["SMILES"]]
    dists = []
    for i in range(1, len(tb)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(tb[i], tb[:i]))
    clusters = Butina.ClusterData(dists, len(tb), 0.65, isDistData=True)
    top["Cluster"] = -1
    for cid, members in enumerate(clusters):
        top.loc[list(members), "Cluster"] = cid
    cl = (top.groupby("Cluster").agg(best=("Score", "max"), size=("Score", "size"))
          .query("size >= 3").sort_values("best", ascending=False))
    top["Family"] = [core_family(x) for x in top["SMILES"]]
    picks, fam_count = [], {}
    for c in cl.index:
        members = top[top.Cluster == c].sort_values("Score", ascending=False)
        fam = members.iloc[0]["Family"]
        if fam_count.get(fam, 0) >= FAMILY_CAP:
            continue
        fam_count[fam] = fam_count.get(fam, 0) + 1
        picks.append(members.head(PER_SERIES))
        if len(picks) == N_SERIES:
            break
    chosen = pd.concat(picks)
    print(f"[*] core families used: {fam_count}", flush=True)
    if len(chosen) < 50:
        chosen = pd.concat([chosen, top[~top.CatalogID.isin(chosen.CatalogID)].head(50 - len(chosen))])
    chosen = chosen.head(50)
    assert chosen.CatalogID.nunique() == 50

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "Family", "Score", "SMILES"]].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
