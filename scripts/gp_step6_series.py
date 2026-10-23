"""
Official-style chemical series (Kruger et al.) as described in the challenge rules, applied to a small set:
  1. Dendrogram of ECFP4 (2048 bits, r=2) Tanimoto distance, average linkage.
  2. Breadth-first walk from the root; for each cluster compute the MCS (rdFMCS, threshold 1.0, ring atoms only
     match ring atoms, complete rings, elements + bond order).
  3. If the MCS occurs in < 100 molecules of the library (validation + test pools, 428,960), the cluster defines a series
     (not descended further); otherwise descend to the two children.
  4. Series that share a Bemis-Murcko scaffold (among their members) are merged.
NOT implemented (unspecified in the task): the final K-medoid step mentioned in the v3 document.
Two runs: (A) 13 hits + 130 confirmed negatives (as requested); (B) the 13 hits alone (the official N_clusters is
computed among the hits of a submission only).
Run: python -m scripts.gp_step6_series -> data/processed/gp/step6_series.json
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
LIB_CUT, MIN_ATOMS = 100, 5
MAX_MCS_MEMBERS = None  # optional: descend without computing an MCS for clusters larger than this (memory guard for big pools; None = original behaviour)
_cache = {}


def lib_count(smarts, lib):
    if smarts in _cache:
        return _cache[smarts]
    q = Chem.MolFromSmarts(smarts)
    n = 0
    for m in lib:
        if m.HasSubstructMatch(q):
            n += 1
            if n >= LIB_CUT:
                break
    _cache[smarts] = n
    return n


def run(ids, smiles, y, lib, label):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = [gen.GetFingerprint(m) for m in mols]
    n = len(fp)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[i], fp))
    root = to_tree(linkage(squareform(D, checks=False), method="average"))
    series, queue = [], [root]
    n_tested = 0
    while queue:  # BFS
        node = queue.pop(0)
        members = node.pre_order()
        if len(members) < 2:
            continue
        if MAX_MCS_MEMBERS and len(members) > MAX_MCS_MEMBERS:
            queue.extend([node.get_left(), node.get_right()])
            continue
        r = rdFMCS.FindMCS([mols[i] for i in members], threshold=1.0, ringMatchesRingOnly=True, completeRingsOnly=True,
                           atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder, timeout=20)
        if r.numAtoms >= MIN_ATOMS and r.smartsString:
            n_tested += 1
            c = lib_count(r.smartsString, lib)
            if c < LIB_CUT:
                series.append({"members": members, "smarts": r.smartsString, "mcs_atoms": r.numAtoms, "lib_matches": c})
                continue
        queue.extend([node.get_left(), node.get_right()])
    # merge by shared Bemis-Murcko scaffold among members
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    parent = list(range(len(series)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    by_scaf = defaultdict(list)
    for si, s in enumerate(series):
        for i in s["members"]:
            if scaf[i]:
                by_scaf[scaf[i]].append(si)
    for sl in by_scaf.values():
        for a in sl[1:]:
            parent[find(a)] = find(sl[0])
    groups = defaultdict(list)
    for si in range(len(series)):
        groups[find(si)].append(si)
    out_series = []
    for g in groups.values():
        mem = sorted({i for si in g for i in series[si]["members"]})
        out_series.append({"n_raw_series_merged": len(g), "smarts": [series[si]["smarts"] for si in g],
                           "lib_matches": [series[si]["lib_matches"] for si in g],
                           "members": [ids[i] for i in mem], "n_hits": int(sum(y[i] for i in mem)),
                           "n_neg": int(sum(1 - y[i] for i in mem))})
    in_series = {m for s in out_series for m in s["members"]}
    hit_ids = [ids[i] for i in range(n) if y[i] == 1]
    res = {"run": label, "n_compounds": n, "n_hits": int(sum(y)), "nodes_with_mcs_tested": n_tested,
           "n_raw_series": len(series), "n_series_after_scaffold_merge": len(out_series),
           "n_series_with_hits": int(sum(1 for s in out_series if s["n_hits"] > 0)),
           "hits_not_in_any_series": [h for h in hit_ids if h not in in_series],
           "series": sorted(out_series, key=lambda s: -s["n_hits"])}
    print(f"[{label}] raw series {len(series)} -> {len(out_series)} after BM-scaffold merge; series with hits: "
          f"{res['n_series_with_hits']}; hits outside any series: {len(res['hits_not_in_any_series'])}/{res['n_hits']}", flush=True)
    return res


def main():
    hits = pd.read_csv("docs/redteam_hits_confirmados.csv")[["CatalogID", "SMILES"]]
    lig = pd.read_csv("data/docking/retro/ligands.csv")
    neg = lig[lig.y == 0][["CatalogID", "SMILES"]]
    lib = []
    for f in ("data/raw/Val-Test-set/PGK2_Validation_split.csv", "data/raw/Val-Test-set/PGK2_Test_split.csv"):
        lib += [m for m in (Chem.MolFromSmiles(s) for s in pd.read_csv(f).SMILES) if m is not None]
    print(f"[*] library molecules: {len(lib):,}")
    allc = pd.concat([hits, neg], ignore_index=True)
    y = np.array([1] * len(hits) + [0] * len(neg))
    A = run(allc.CatalogID.tolist(), allc.SMILES.tolist(), y, lib, "A_hits+130neg")
    B = run(hits.CatalogID.tolist(), hits.SMILES.tolist(), np.ones(len(hits), int), lib, "B_hits_only")
    # specificity alert: negatives matching the MCS of any hit-bearing series (run A) and of series from run B
    negall = pd.read_csv("docs/redteam_negativos_confirmados.csv")
    negm = [Chem.MolFromSmiles(s) for s in negall.SMILES]
    negm130 = [Chem.MolFromSmiles(s) for s in neg.SMILES]
    alerts = []
    for tag, R in (("A", A), ("B", B)):
        for s in R["series"]:
            if s["n_hits"] == 0:
                continue
            for sm in s["smarts"]:
                q = Chem.MolFromSmarts(sm)
                a130 = [neg.CatalogID.iloc[i] for i, m in enumerate(negm130) if m.HasSubstructMatch(q)]
                aall = [negall.CatalogID.iloc[i] for i, m in enumerate(negm) if m.HasSubstructMatch(q)]
                alerts.append({"run": tag, "smarts": sm, "hits_in_series": s["n_hits"], "neg130_matching": a130,
                               "neg2269_matching_n": len(aall), "neg2269_matching": aall})
    out = {"library_size": len(lib), "A": A, "B": B, "specificity_alerts": alerts}
    json.dump(out, open("data/processed/gp/step6_series.json", "w"), indent=1)
    for tag, R in (("A", A), ("B", B)):
        print(f"\n== {R['run']} ==")
        for s in R["series"]:
            print(f"  series: hits={s['n_hits']} neg={s['n_neg']} merged_raw={s['n_raw_series_merged']} lib={s['lib_matches']} smarts={s['smarts'][0][:90]}")
    print("\n== specificity alerts (negatives matching the MCS of a hit-bearing series) ==")
    for a in alerts:
        print(f"  run {a['run']} hits={a['hits_in_series']} neg130={len(a['neg130_matching'])} neg2269={a['neg2269_matching_n']} {a['smarts'][:80]}")


if __name__ == "__main__":
    main()
