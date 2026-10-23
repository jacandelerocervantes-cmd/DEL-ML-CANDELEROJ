"""
Red-team follow-up: two independent training recipes (full-imbalance+ASL,
and negative-subsampled+BCE matching the official baseline) both converged
on the same shortcut -- top-50 picks are dominated by azole rings (triazole/
tetrazole/imidazole/pyrazole), 2-8x enriched vs. a random pool sample. Since
retraining twice didn't remove it, this script attacks it directly:
score the full official pool with the best model so far (domain-mask +
physchem + disynthon label, no negative subsampling -- PR-AUC=0.4896
held-out, the best of the three disynthon variants), hard-exclude any
candidate containing a 5-membered aromatic ring with >=2 N atoms (catches
triazole/tetrazole/imidazole/pyrazole generically via ring perception
instead of enumerating SMARTS), then re-select top-50 with the same
diversity clustering (Butina, Tanimoto cutoff 0.65) + per-scaffold-family
cap used in disynthon_score.py/combine_docking.py -- never a raw
sort_values().head(50) again (red-team finding #3).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import zipfile
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.train_pgk2_full import featurize_parallel

RDLogger.DisableLog("rdApp.*")

MODEL = "models/pretrained/pgk2_xgboost_asl_mask_physchem_disynthon.json"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
PHYSCHEM = "data/processed/pgk2_physchem.npz"
BIT_MASK = "data/processed/pgk2_bit_mask.npz"
SUBMISSION_INDEX = 25
N_SERIES, PER_SERIES, FAMILY_CAP = 10, 5, 2


def has_azole_ring(mol) -> bool:
    """Any 5-membered aromatic ring with >=2 N atoms -- catches triazole,
    tetrazole, imidazole, pyrazole, oxadiazole-with-2N etc. generically,
    instead of enumerating one SMARTS per azole type."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 5:
            continue
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        n_n = sum(1 for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() == 7)
        if n_n >= 2:
            return True
    return False


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


def main():
    train_idx = np.load(SPLIT_CACHE)["train_idx"]
    pc_train = np.load(PHYSCHEM)["X"][train_idx]
    scaler = StandardScaler().fit(pc_train)
    keep_cols = np.load(BIT_MASK)["keep"]

    model = XGBClassifier()
    model.load_model(MODEL)

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
            pool = pd.read_csv(f)
    n_tested = 0
    if os.path.exists("submissions/tested_zero_hits.txt"):
        tested = set(open("submissions/tested_zero_hits.txt").read().split())
        n_tested = len(tested)
        pool = pool[~pool.CatalogID.isin(tested)].reset_index(drop=True)
    print(f"[*] Scoring {len(pool):,} official pool candidates (excl. {n_tested:,} already tested)...", flush=True)

    fps, pcs = featurize_parallel(pool["SMILES"].tolist(), want_physchem=True)
    ok = [fp is not None for fp in fps]
    pool = pool[ok].reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in pool.SMILES]
    fp_masked = np.stack([fp for fp, k in zip(fps, ok) if k])[:, keep_cols]
    pc_scaled = scaler.transform(np.stack([pc for pc, k in zip(pcs, ok) if k])).astype(np.float32)
    X = np.hstack([fp_masked, pc_scaled])
    pool["Score"] = model.predict(X, output_margin=True)

    is_azole = np.array([has_azole_ring(m) for m in mols])
    print(f"[*] azole-ring candidates in full pool: {is_azole.sum():,} / {len(pool):,} ({100*is_azole.mean():.1f}%)", flush=True)
    n_azole_top1000 = int(is_azole[np.argsort(-pool['Score'].to_numpy())[:1000]].sum())
    print(f"[*] azole-ring share of the raw top-1000 by score (pre-filter): {n_azole_top1000}/1000 ({n_azole_top1000/10:.1f}%)", flush=True)

    filtered = pool[~is_azole].sort_values("Score", ascending=False).reset_index(drop=True)
    print(f"[*] {len(filtered):,} candidates remain after azole exclusion", flush=True)

    top = filtered.head(4000).copy()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top["SMILES"]]
    dists = []
    for i in range(1, len(bvs)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
    clusters = Butina.ClusterData(dists, len(bvs), 0.65, isDistData=True)
    top["Cluster"] = -1
    for cid, members in enumerate(clusters):
        top.loc[list(members), "Cluster"] = cid
    top["Family"] = [core_family(s) for s in top["SMILES"]]
    cl = (top.groupby("Cluster").agg(best=("Score", "max"), size=("Score", "size"))
          .query("size >= 3").sort_values("best", ascending=False))

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
    chosen = pd.concat(picks) if picks else top.iloc[0:0]
    if len(chosen) < 50:
        chosen = pd.concat([chosen, top[~top.CatalogID.isin(chosen.CatalogID)].head(50 - len(chosen))])
    chosen = chosen.head(50)
    assert chosen.CatalogID.nunique() == 50
    assert not any(has_azole_ring(Chem.MolFromSmiles(s)) for s in chosen.SMILES), "azole leaked through filter"

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] families used: {fam_count}", flush=True)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "Family", "Score", "SMILES"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
