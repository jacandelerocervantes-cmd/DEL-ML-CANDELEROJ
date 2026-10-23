"""
Series (Kruger-style, scripts/gp_step6_series.py) over: 13 confirmed hits + 2,269 confirmed negatives + 111 group-testing
suspects (the three disjoint 37-compound partitions of submissions 80/81/82, i.e. their non-hit compounds).
Reports, per merged series: confirmed negatives / suspects / hits, MCS, library matches, tested-vs-untested library space;
then for series 1 and 2 (from step 6) whether the negatives are inside the exact MCS or are close neighbours.
Run: python -m scripts.gp_step7_neg_density  -> data/processed/gp/step7_neg_density.json
"""
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdSubstructLibrary

from scripts import gp_step6_series as s6

RDLogger.DisableLog("rdApp.*")


def main():
    hits = pd.read_csv("docs/redteam_hits_confirmados.csv")[["CatalogID", "SMILES"]].assign(kind="hit")
    negs = pd.read_csv("docs/redteam_negativos_confirmados.csv")[["CatalogID", "SMILES"]].assign(kind="neg")
    H = set(hits.CatalogID)
    smap = {}
    sus_rows = []
    for n in (80, 81, 82):
        ids = [l.strip() for l in open(f"submissions/Team_candeleroj_{n}.txt") if l.strip()]
        smap[n] = [i for i in ids if i not in H]
    pool = pd.concat([pd.read_csv(f"data/raw/Val-Test-set/PGK2_{s}_split.csv") for s in ("Validation", "Test")], ignore_index=True)
    smi = pool.set_index("CatalogID").SMILES
    for n, ids in smap.items():
        sus_rows += [{"CatalogID": i, "SMILES": smi[i], "kind": f"sus{n}"} for i in ids]
    sus = pd.DataFrame(sus_rows)
    allc = pd.concat([hits, negs, sus], ignore_index=True).drop_duplicates("CatalogID").reset_index(drop=True)
    print(f"[*] set: {len(allc)} = {len(hits)} hits + {len(negs)} negatives + {len(sus)} suspects", flush=True)

    # fast library counter (pattern-fingerprint screened) replaces the linear scan in step 6
    libm = [m for m in (Chem.MolFromSmiles(s) for s in pool.SMILES) if m is not None]
    sl = rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.MolHolder(), rdSubstructLibrary.PatternHolder())
    for m in libm:
        sl.AddMol(m)
    print(f"[*] indexed {len(libm):,} library molecules", flush=True)

    def fast_count(smarts, lib):
        if smarts in s6._cache:
            return s6._cache[smarts]
        q = Chem.MolFromSmarts(smarts)
        c = len(sl.GetMatches(q, maxResults=s6.LIB_CUT))
        s6._cache[smarts] = c
        return c
    s6.lib_count = fast_count

    y = (allc.kind == "hit").astype(int).to_numpy()
    R = s6.run(allc.CatalogID.tolist(), allc.SMILES.tolist(), y, None, "C_hits+2269neg+111sus")
    kind = allc.set_index("CatalogID").kind
    mols = {r.CatalogID: Chem.MolFromSmiles(r.SMILES) for r in allc.itertuples()}
    rows = []
    for s in R["series"]:
        k = [kind[m] for m in s["members"]]
        nneg, nhit, nsus = k.count("neg"), k.count("hit"), sum(1 for x in k if x.startswith("sus"))
        q = Chem.MolFromSmarts(s["smarts"][0])
        tested_match = int(sum(1 for m in mols.values() if m.HasSubstructMatch(q)))
        lib = s["lib_matches"][0]
        rows.append({"smarts": s["smarts"], "lib_matches": s["lib_matches"], "n_members": len(s["members"]), "n_neg": nneg,
                     "n_hit": nhit, "n_sus": nsus, "tested_matching_mcs": tested_match,
                     "untested_lib_est": max(lib - tested_match, 0), "example": mols[s["members"][0]] and Chem.MolToSmiles(mols[s["members"][0]]),
                     "members": s["members"]})
    R["rows"] = rows

    # series 1 & 2 from step 6: negatives inside the exact MCS vs close neighbours
    step6 = json.load(open("data/processed/gp/step6_series.json"))["B"]["series"]
    gen = s6.gen
    negfp = {i: gen.GetFingerprint(mols[i]) for i in negs.CatalogID}
    neigh = []
    for si, s in enumerate(step6, 1):
        q = Chem.MolFromSmarts(s["smarts"][0])
        hfp = [gen.GetFingerprint(mols[h]) for h in s["members"]]
        inside, near = [], []
        for i, f in negfp.items():
            sim = max(DataStructs.BulkTanimotoSimilarity(f, hfp))
            rec = {"id": i, "smiles": Chem.MolToSmiles(mols[i]), "max_tanimoto_to_series_hits": round(float(sim), 3)}
            if mols[i].HasSubstructMatch(q):
                inside.append(rec)
            elif sim >= 0.35:
                near.append(rec)
        near.sort(key=lambda r: -r["max_tanimoto_to_series_hits"])
        neigh.append({"series": si, "n_hits": s["n_hits"], "smarts": s["smarts"][0], "negatives_inside_exact_mcs": inside,
                      "n_neg_outside_mcs_tanimoto>=0.35": len(near), "top_neighbours_outside_mcs": near[:8]})
    R["series12_negatives"] = neigh
    json.dump(R, open("data/processed/gp/step7_neg_density.json", "w"), indent=1, default=str)

    df = pd.DataFrame(rows)
    print("\nSERIES SUMMARY (sorted by confirmed negatives)")
    print(f"  total series {len(df)}; with >=1 hit: {(df.n_hit>0).sum()}; negatives in series: {int(df.n_neg.sum())}/{len(negs)}; "
          f"suspects in series: {int(df.n_sus.sum())}/{len(sus)}; hits in series: {int(df.n_hit.sum())}/{len(hits)}")
    print(df.sort_values("n_neg", ascending=False).head(15)[["n_members", "n_neg", "n_sus", "n_hit", "lib_matches", "tested_matching_mcs", "untested_lib_est", "example"]].to_string())
    print("\nSERIES with suspects/hits but zero confirmed negatives:")
    z = df[(df.n_neg == 0) & ((df.n_sus > 0) | (df.n_hit > 0))]
    print(z[["n_members", "n_sus", "n_hit", "lib_matches", "untested_lib_est", "example"]].to_string())
    for n in neigh:
        print(f"\nSERIES {n['series']} ({n['n_hits']} hits): inside MCS: {[(r['id'], r['max_tanimoto_to_series_hits']) for r in n['negatives_inside_exact_mcs']]}"
              f" | outside with Tanimoto>=0.35: {n['n_neg_outside_mcs_tanimoto>=0.35']}")


if __name__ == "__main__":
    main()
