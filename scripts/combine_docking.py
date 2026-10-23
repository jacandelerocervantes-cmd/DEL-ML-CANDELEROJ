"""
Combines the fragment+ErG+pharm+prop score (data/processed/v2_candidates.csv)
with docking scores (data/docking/screen_scores.csv, PGK2 + PGK1) for
the 300 docked candidates, then re-does the series-diverse, family-capped
selection on that combined score.

Weight: 0.5 fragment-composite + 0.5 docking, since the docking AUC=0.91 gate
result earns it equal footing with the DEL-derived score, not a minor tweak.

Fix (audit Critico #5): the docking term used to be
`dock_ensemble = -(pgk2 + pgk1)`, which REWARDS a compound for binding PGK1
just as strongly as PGK2 -- i.e. it actively selects for non-selective,
pan-PGK-quinasa binders, backwards given PGK1/PGK2 share ~87% active-site
identity and selectivity is the whole point of docking here. Replaced with
`selectivity_score = pgk1 - pgk2` (higher = binds PGK2 more strongly than
PGK1; Vina energies are negative kcal/mol, so more negative = stronger
binding, hence pgk1-pgk2 > 0 when PGK2 binding is stronger).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

SUBMISSION_INDEX = 10
N_SERIES, PER_SERIES, FAMILY_CAP = 10, 5, 2
W_FRAG, W_DOCK = 0.5, 0.5


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


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    cand = pd.read_csv("data/processed/v2_candidates.csv")
    dock = pd.read_csv("data/docking/screen_scores.csv").rename(columns={"id": "CatalogID"})
    df = cand.merge(dock, on="CatalogID", how="inner").dropna(subset=["pgk2", "pgk1"])
    print(f"[*] {len(df)} candidates with both fragment-score and docking score", flush=True)

    df["selectivity_score"] = df["pgk1"] - df["pgk2"]  # higher = more PGK2-selective (see module docstring)
    df["z_frag_composite"] = zs(df["Score"])
    df["z_dock"] = zs(df["selectivity_score"])
    df["FinalScore"] = W_FRAG * df["z_frag_composite"] + W_DOCK * df["z_dock"]
    print(f"[*] corr(fragment composite, docking) = {df.z_frag_composite.corr(df.z_dock):.2f}", flush=True)
    df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df["SMILES"]]
    dists = []
    for i in range(1, len(bvs)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
    clusters = Butina.ClusterData(dists, len(bvs), 0.65, isDistData=True)
    df["Cluster"] = -1
    for cid, members in enumerate(clusters):
        df.loc[list(members), "Cluster"] = cid
    cl = (df.groupby("Cluster").agg(best=("FinalScore", "max"), size=("FinalScore", "size"))
          .query("size >= 2").sort_values("best", ascending=False))
    df["Family"] = [core_family(s) for s in df["SMILES"]]

    picks, fam_count = [], {}
    for c in cl.index:
        members = df[df.Cluster == c].sort_values("FinalScore", ascending=False)
        fam = members.iloc[0]["Family"]
        if fam_count.get(fam, 0) >= FAMILY_CAP:
            continue
        fam_count[fam] = fam_count.get(fam, 0) + 1
        picks.append(members.head(PER_SERIES))
        if len(picks) == N_SERIES:
            break
    chosen = pd.concat(picks) if picks else df.iloc[0:0]
    if len(chosen) < 50:
        chosen = pd.concat([chosen, df[~df.CatalogID.isin(chosen.CatalogID)].head(50 - len(chosen))])
    chosen = chosen.head(50)
    assert chosen.CatalogID.nunique() == 50

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] core families used: {fam_count}", flush=True)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "Family", "FinalScore", "pgk2", "pgk1", "SMILES"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
