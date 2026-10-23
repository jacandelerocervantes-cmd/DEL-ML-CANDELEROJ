"""
Flag (NOT exclude) the compounds of a final 50 that are close to the 50 compounds of the first test submission, all of which returned 0 hits (so they are confirmed non-hits).
Reports the max Tanimoto (ECFP4 bits, 2048, r=2) of each candidate to those 50, with the closest one, and lists those >= 0.4 / 0.5 / 0.6.
Caution kept from the project: near analogues of a negative can still be active (activity cliffs); a similarity threshold of 0.40 against earlier rounds would have removed
4 of 5 known hits according to the red-team brief (document, not verified). So this is a priority flag, not a filter.
Usage: python -m scripts.gp_step25_neg_similarity <final50.csv> -> data/processed/gp/step28_neg_similarity_<name>.csv
"""
import os
import sys

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def main():
    path = sys.argv[1]
    cand = pd.read_csv(path)
    first = pd.read_csv("data/processed/gp/step13_first_test_selected50.csv")[["CatalogID"]]
    smi = pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv").set_index("CatalogID").SMILES  # the saved CSV has no SMILES column
    first["SMILES"] = first.CatalogID.map(smi)
    ffp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in first.SMILES]
    rows = []
    for r in cand.itertuples():
        f = gen.GetFingerprint(Chem.MolFromSmiles(r.SMILES))
        sims = DataStructs.BulkTanimotoSimilarity(f, ffp)
        j = max(range(len(sims)), key=lambda k: sims[k])
        rows.append({"CatalogID": r.CatalogID, "series_id": r.series_id, "es_ancla": bool(getattr(r, "es_ancla", False)), "max_tanimoto_a_los_50_negativos": round(sims[j], 3),
                     "negativo_mas_cercano": first.CatalogID[j], "SMILES": r.SMILES, "SMILES_negativo": first.SMILES[j]})
    out = pd.DataFrame(rows).sort_values("max_tanimoto_a_los_50_negativos", ascending=False)
    name = os.path.splitext(os.path.basename(path))[0]
    out.to_csv(f"data/processed/gp/step28_neg_similarity_{name}.csv", index=False)
    print({t: int((out.max_tanimoto_a_los_50_negativos >= t).sum()) for t in (0.4, 0.5, 0.6, 0.7)})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(out.head(20)[["CatalogID", "series_id", "es_ancla", "max_tanimoto_a_los_50_negativos", "negativo_mas_cercano", "SMILES", "SMILES_negativo"]].to_string(index=False))
    print("anclas:", out[out.es_ancla][["CatalogID", "max_tanimoto_a_los_50_negativos", "negativo_mas_cercano"]].to_string(index=False))


if __name__ == "__main__":
    main()
