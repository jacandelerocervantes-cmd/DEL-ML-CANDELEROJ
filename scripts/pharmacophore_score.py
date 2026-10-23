"""
Literature-anchored, DEL-INDEPENDENT scoring -- following the strategy that
won Round 2 of the 1st DREAM Target 2035 Challenge (per the project's own
research doc): "systematically expanded a small subset of literature-derived
active compounds using manually curated bioisosteric replacements... while
intentionally omitting the noisy DEL data entirely."

Anchor: CACHE #7 confirms PGK2's known selective inhibitors (compound 21,
compound 47, co-crystallized) are QUINAZOLINE-series ATP-site competitors.
Terazosin (a quinazoline drug) is independently documented bound to PGK1's
nucleotide site (PDB 4O33, 2.0 A) with the same 4-aminoquinazoline hinge
motif: an exocyclic amine/amide directly on a bicyclic N-heteroaromatic
ring mimicking adenine's hydrogen-bond pattern.

Unlike every prior scoring pass in this project, NONE of this comes from
the PGK2 DEL enrichment statistics -- it is defined purely from the known
pharmacophore (terazosin's structure + standard quinazoline bioisosteres:
purine, imidazo[4,5-b]pyridine, benzimidazole, pyrido-pyrimidine,
naphthyridine, quinoxaline -- families textbook-documented as adenine-site
hinge-binder bioisosteres). The challenge explicitly wants NON-quinazoline
hits, so quinazoline itself is deliberately excluded from the query set.

This is a genuinely independent hypothesis from the DEL-derived submissions
(#1-11): if it also returns 0, that is real evidence against the hinge-
binder hypothesis for this pocket, not just noise in our DEL labels.
"""

import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
TESTED_TXT = "submissions/tested_zero_hits.txt"
SUBMISSION_INDEX = 12
N_SERIES, PER_SERIES, FAMILY_CAP = 10, 5, 2

# Bicyclic N-heteroaromatic "hinge core" families documented as adenine-site
# bioisosteres (quinazoline itself excluded -- already known/not novel).
HINGE_CORES = {
    "purine":            "c1nc2[nH]cnc2n1",
    "imidazopyridine_a": "c1ccc2[nH]cnc2n1",
    "imidazopyridine_b": "c1ccc2nc[nH]c2n1",
    "benzimidazole":     "c1ccc2[nH]cnc2c1",
    "pyridopyrimidine":  "c1cnc2ncncc2c1",
    "naphthyridine":     "c1ccc2ncccc2n1",
    "quinoxaline":       "c1ccc2nccnc2c1",
    "pyrazolopyridine":  "c1ccc2cc[nH]nc2n1",
    "triazolopyridine":  "c1ccc2nnc[nH]c2n1" if False else "c1ccc2[nH]nnc2n1",
}

# Terazosin-like substituent pattern: an exocyclic N directly bonded to the
# hinge ring (amine or amide N), matching the 4-NH2 / 2-piperazinylcarbonyl
# decoration pattern documented for the PGK quinazoline series.
EXO_N_PATT = Chem.MolFromSmarts("[#7;!R]!@[c]")

TERAZOSIN = "COc1cc2nc(N3CCN(C(=O)C4CCCO4)CC3)nc(N)c2cc1OC"


def build_queries():
    queries = []
    for name, smarts in HINGE_CORES.items():
        m = Chem.MolFromSmiles(smarts) or Chem.MolFromSmarts(smarts)
        if m is None:
            print(f"    [!] could not parse {name}: {smarts}", flush=True)
            continue
        queries.append((name, m))
    return queries


def main():
    t_mol = Chem.MolFromSmiles(TERAZOSIN)
    print(f"[*] Anchor: terazosin MW={Descriptors.MolWt(t_mol):.0f} "
          f"logP={Descriptors.MolLogP(t_mol):.1f} (known PGK1 nucleotide-site binder, PDB 4O33)", flush=True)

    queries = build_queries()
    print(f"[*] {len(queries)} literature-anchored hinge-core bioisostere queries "
          f"(quinazoline itself excluded -- CACHE#7 says organizers want NON-quinazoline hits):", flush=True)
    for name, _ in queries:
        print(f"    {name}: {HINGE_CORES[name]}", flush=True)

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
            pool = pd.read_csv(f)
    if os.path.exists(TESTED_TXT):
        tested = set(open(TESTED_TXT).read().split())
        pool = pool[~pool.CatalogID.isin(tested)].reset_index(drop=True)
    print(f"[*] scoring {len(pool):,} pool candidates (already-tested compounds excluded)...", flush=True)

    mols = [Chem.MolFromSmiles(s) for s in pool["SMILES"]]
    core_hit = np.full(len(pool), "", dtype=object)
    exo_n = np.zeros(len(pool), dtype=bool)
    mw = np.full(len(pool), np.nan)
    logp = np.full(len(pool), np.nan)

    for i, m in enumerate(mols):
        if m is None:
            continue
        for name, patt in queries:
            if m.HasSubstructMatch(patt):
                core_hit[i] = name
                break
        if core_hit[i]:
            exo_n[i] = m.HasSubstructMatch(EXO_N_PATT)
            mw[i] = Descriptors.MolWt(m)
            logp[i] = Descriptors.MolLogP(m)

    has_core = core_hit != ""
    print(f"[*] {has_core.sum():,} candidates match a hinge-core bioisostere "
          f"({exo_n[has_core].sum():,} of those also have the terazosin-like exocyclic-N substituent)", flush=True)

    # Terazosin-like drug-likeness gate: MW 250-500, logP -1 to 4 (terazosin: MW 387, logP ~0.6)
    mw_ok = (mw >= 250) & (mw <= 500)
    logp_ok = (logp >= -1) & (logp <= 4)
    qualifies = has_core & exo_n & mw_ok & logp_ok
    print(f"[*] {qualifies.sum():,} candidates pass the full terazosin-like pharmacophore gate "
          f"(hinge core + exocyclic N + drug-like MW/logP)", flush=True)

    pool["core"] = core_hit
    pool["exo_n"] = exo_n
    pool["MW"] = mw
    pool["LogP"] = logp
    pool["mw_dist"] = np.abs(mw - Descriptors.MolWt(t_mol))
    pool["Score"] = np.where(qualifies, -pool["mw_dist"], -1e9)  # closer to terazosin MW ranks higher, among qualifiers
    cand = pool[qualifies].copy()

    if len(cand) < 50:
        print(f"[!] Only {len(cand)} candidates qualify -- fewer than 50. "
              f"Relaxing to hinge-core + exocyclic-N only (dropping MW/logP gate) to fill out to 50.", flush=True)
        relaxed = pool[has_core & exo_n].copy()
        cand = relaxed.sort_values("mw_dist").head(max(50, len(cand)))

    def core_family(smi):
        return Chem.MolFromSmiles(smi) and next((n for n, p in queries if Chem.MolFromSmiles(smi).HasSubstructMatch(p)), "none") or "none"

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in cand["SMILES"]]
    dists = []
    for i in range(1, len(bvs)):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
    clusters = Butina.ClusterData(dists, len(bvs), 0.65, isDistData=True) if len(bvs) > 1 else [(0,)]
    cand = cand.reset_index(drop=True)
    cand["Cluster"] = -1
    for cid, members in enumerate(clusters):
        cand.loc[list(members), "Cluster"] = cid

    picks, fam_count = [], {}
    cl = (cand.groupby("Cluster").agg(best=("Score", "max"), size=("Score", "size"))
          .sort_values(["size", "best"], ascending=[False, False]))
    for c in cl.index:
        members = cand[cand.Cluster == c].sort_values("Score", ascending=False)
        fam = members.iloc[0]["core"]
        if fam_count.get(fam, 0) >= FAMILY_CAP:
            continue
        fam_count[fam] = fam_count.get(fam, 0) + 1
        picks.append(members.head(PER_SERIES))
        if len(picks) == N_SERIES:
            break
    chosen = pd.concat(picks) if picks else cand.iloc[0:0]
    if len(chosen) < 50:
        chosen = pd.concat([chosen, cand[~cand.CatalogID.isin(chosen.CatalogID)].head(50 - len(chosen))])
    chosen = chosen.head(50)
    if chosen.CatalogID.nunique() < 50:
        raise ValueError(f"Only found {chosen.CatalogID.nunique()} unique candidates matching the "
                         f"pharmacophore across the whole pool -- literature-anchored filter is too strict here.")

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] core families used: {fam_count}", flush=True)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "core", "MW", "LogP", "SMILES"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
