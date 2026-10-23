"""
Building-block (monosynthon) enrichment scoring.

Unlike ECFP4 bits (hashed, approximate, prone to collisions), each DEL
compound's RandomID (`qDOS11-34-486-863`) identifies its EXACT building
blocks via data/raw/OpenDEL_libraries.zip/OpeDELLibrary/BBids_SMILES/*.csv.
Pooling by exact building-block identity gives much higher statistical power
per unit (a BB used in ~thousands of compounds, vs one ECFP bit hashing many
unrelated substructures together).

Steps:
  1. Load BB1/BB2/BB3 SMILES tables for all 10 libraries.
  2. Parse every compound's RandomID -> (lib, bb1_id, bb2_id, bb3_id).
  3. Soft-weighted enrichment per (lib, slot, bb_id), using the same
     confidence weights as global_score.py (count lower bound x competition
     x NTC x promiscuity), on TRAIN rows only.
  4. For BBs with strong, well-supported enrichment, extract their largest
     aromatic ring system (the part of the BB likely to survive as a
     pharmacophore in the product) and substructure-search the official
     pool for it.
  5. Combine with the existing fragment+ErG+pharm+prop score (data/processed/
     v2_candidates.csv is reused as the composite prior) and re-select.
"""

import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

FEATURES_PARQUET = "data/processed/PGK2_selection_features.parquet"
LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
OPENDEL_ZIP = "data/raw/OpenDEL_libraries.zip"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
CAND_CSV = "data/processed/v2_candidates.csv"
TESTED_TXT = "submissions/tested_zero_hits.txt"

BB_PRIOR_STRENGTH = 50.0
MIN_BB_SUPPORT = 30
MIN_BB_ENRICH = 3.0
SUBMISSION_INDEX = 11
W_PRIOR, W_BB = 0.55, 0.45
N_SERIES, PER_SERIES, FAMILY_CAP = 10, 5, 2


def core_ring_system(mol):
    """Largest fused aromatic ring system as a SMARTS-able fragment, or None."""
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        ring = set(ring)
        merged = [sy for sy in systems if sy & ring]
        for sy in merged:
            systems.remove(sy)
            ring |= sy
        systems.append(ring)
    if not systems:
        return None
    best = max(systems, key=len)
    if len(best) < 5:
        return None
    n_hetero = sum(1 for a in best if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)
    n_rings_fused = sum(1 for ring in ri.AtomRings() if set(ring) <= best)
    if n_hetero == 0 and n_rings_fused <= 1:
        return None  # plain benzene / single carbocycle: too generic to be specific
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(best), kekuleSmiles=False)


def load_bb_tables():
    z = zipfile.ZipFile(OPENDEL_ZIP)
    libs = sorted({n.split("/")[-1][:-4] for n in z.namelist()
                   if n.startswith("OpeDELLibrary/BBids_SMILES/") and n.endswith(".csv")})
    tables = {}
    for lib in libs:
        with z.open(f"OpeDELLibrary/BBids_SMILES/{lib}.csv") as f:
            df = pd.read_csv(f)
        tables[lib] = df
        print(f"    {lib}: {len(df)} BB rows", flush=True)
    return tables


def main():
    print("[*] Loading building-block SMILES tables...", flush=True)
    bb_tables = load_bb_tables()

    meta = pq.read_table(FEATURES_PARQUET, columns=["RandomID", "SMILES", "Label"]).to_pandas()
    aux = pq.read_table(LABELED_PARQUET, columns=["RandomID", "count_PGK2", "count_PGK2_with_inhibitor",
                                                  "count_NTC", "historic_hits"]).to_pandas()
    meta = meta.merge(aux, on="RandomID", how="left")
    parts = meta["RandomID"].str.split("-", expand=True)
    meta["lib"], meta["bb1"], meta["bb2"], meta["bb3"] = parts[0], parts[1], parts[2], parts[3]

    train_idx = np.load(SPLIT_CACHE)["train_idx"]
    is_train = np.zeros(len(meta), dtype=bool)
    is_train[train_idx] = True
    tr = meta[is_train].copy()

    cnt = tr["count_PGK2"].to_numpy().astype(float)
    lb = np.maximum(0, cnt - 1.645 * np.sqrt(cnt))
    f_count = np.clip(np.log1p(lb) / np.log1p(20), 0, 1)
    r = tr["count_PGK2_with_inhibitor"].to_numpy() / np.maximum(cnt, 1)
    f_comp = 1 - 0.3 * np.minimum(1, r / 0.5)
    f_ntc = (tr["count_NTC"].to_numpy() == 0).astype(float)
    f_prom = 1 / (1 + tr["historic_hits"].to_numpy() / 2)
    tr["w"] = (tr["Label"].to_numpy() == 1) * f_count * f_comp * f_ntc * f_prom
    p0 = tr["w"].sum() / len(tr)
    print(f"[*] train rows: {len(tr):,}, soft-positive mass: {tr['w'].sum():.0f} (p0={p0:.5f})", flush=True)

    bb_stats = []
    for slot in ("bb1", "bb2", "bb3"):
        g = tr.groupby(["lib", slot]).agg(n=("w", "size"), w=("w", "sum")).reset_index()
        g["slot"] = slot
        g = g[g.n >= MIN_BB_SUPPORT]
        g["rate"] = (g.w + BB_PRIOR_STRENGTH * p0) / (g.n + BB_PRIOR_STRENGTH)
        g["enrich"] = g["rate"] / p0
        bb_stats.append(g.rename(columns={slot: "bb_id"}))
    bb_stats = pd.concat(bb_stats, ignore_index=True)
    strong = bb_stats[bb_stats.enrich >= MIN_BB_ENRICH].sort_values("enrich", ascending=False)
    print(f"[*] {len(bb_stats):,} (lib,slot,BB) units with support>={MIN_BB_SUPPORT}; "
          f"{len(strong)} with enrich>={MIN_BB_ENRICH}x", flush=True)

    slot_col = {"bb1": "BB1", "bb2": "BB2", "bb3": "BB3"}
    queries = []
    for _, row in strong.iterrows():
        lib, slot, bb_id = row["lib"], row["slot"], row["bb_id"]
        if lib not in bb_tables:
            continue
        col_smi, col_id = f"{slot_col[slot]}_SMILES", f"{slot_col[slot]}_ID"
        tbl = bb_tables[lib]
        try:
            bb_id_num = float(bb_id)
        except ValueError:
            continue
        hit = tbl[tbl[col_id].astype(str).str.strip() == str(int(bb_id_num))]
        if hit.empty:
            hit = tbl[pd.to_numeric(tbl[col_id], errors="coerce") == bb_id_num]
        if hit.empty:
            continue
        smi = str(hit.iloc[0][col_smi]).split("|")[0].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        core = core_ring_system(mol)
        if core is None:
            continue
        patt = Chem.MolFromSmarts(core) if not Chem.MolFromSmiles(core) else Chem.MolFromSmiles(core)
        if patt is None:
            continue
        queries.append({"lib": lib, "slot": slot, "bb_id": bb_id, "enrich": row["enrich"],
                        "n": int(row["n"]), "core": core, "patt": patt})
    # dedupe identical cores, keep max enrichment
    seen, dedup = {}, []
    for q in sorted(queries, key=lambda x: -x["enrich"]):
        if q["core"] in seen:
            continue
        seen[q["core"]] = True
        dedup.append(q)
    queries = dedup[:60]
    print(f"[*] {len(queries)} distinct enriched building-block core queries (top by enrichment):", flush=True)
    for q in queries[:15]:
        print(f"    {q['lib']}/{q['slot']} bb={q['bb_id']} enrich={q['enrich']:.1f}x n={q['n']} core={q['core']}", flush=True)

    if os.path.exists(CAND_CSV):
        cand = pd.read_csv(CAND_CSV)
    else:
        with zipfile.ZipFile(VAL_TEST_ZIP) as z:
            with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
                cand = pd.read_csv(f)
        cand["Score"] = 0.0
    if os.path.exists(TESTED_TXT):
        tested = set(open(TESTED_TXT).read().split())
        cand = cand[~cand.CatalogID.isin(tested)].reset_index(drop=True)
    print(f"[*] scoring {len(cand):,} pool candidates against BB-core queries...", flush=True)

    mols = [Chem.MolFromSmiles(s) for s in cand["SMILES"]]
    bb_score = np.zeros(len(cand))
    matched_any = np.zeros(len(cand), dtype=bool)
    for q in queries:
        w = np.log(q["enrich"])
        for i, m in enumerate(mols):
            if m is not None and m.HasSubstructMatch(q["patt"]):
                bb_score[i] += w
                matched_any[i] = True
    print(f"[*] {matched_any.sum()} / {len(cand)} candidates match at least one enriched BB core", flush=True)

    cand["bb_score"] = bb_score
    z_prior = (cand["Score"] - cand["Score"].mean()) / (cand["Score"].std() + 1e-9) if "Score" in cand else 0.0
    z_bb = (bb_score - bb_score.mean()) / (bb_score.std() + 1e-9)
    cand["FinalScore"] = W_PRIOR * z_prior + W_BB * z_bb
    cand = cand.sort_values("FinalScore", ascending=False).reset_index(drop=True)

    def core_family(smi):
        m = Chem.MolFromSmiles(smi)
        c = core_ring_system(m)
        return c or "none"

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    top = cand.head(4000).copy()
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top["SMILES"]]
    dists = []
    for i in range(1, len(bvs)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
    clusters = Butina.ClusterData(dists, len(bvs), 0.65, isDistData=True)
    top["Cluster"] = -1
    for cid, members in enumerate(clusters):
        top.loc[list(members), "Cluster"] = cid
    top["Family"] = [core_family(s) for s in top["SMILES"]]
    cl = (top.groupby("Cluster").agg(best=("FinalScore", "max"), size=("FinalScore", "size"))
          .query("size >= 3").sort_values("best", ascending=False))

    picks, fam_count = [], {}
    for c in cl.index:
        members = top[top.Cluster == c].sort_values("FinalScore", ascending=False)
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

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] families used: {fam_count}", flush=True)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "Family", "FinalScore", "bb_score", "SMILES"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
