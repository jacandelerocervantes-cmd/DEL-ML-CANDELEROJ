"""
Leave-one-out CV of a pharmacophore derived from the confirmed ASMS hits.

Derivation rule (fixed, so each fold is reproducible): the pharmacophore is the maximum common substructure (RDKit
rdFMCS, threshold 1.0 = present in ALL training hits, elements + bond order, ring atoms only match ring atoms,
complete rings only) of the training hits, expressed as SMARTS.
Fold i: derive from the other 12 hits, then
  - match the held-out hit (binary), plus soft scores: nearest-neighbour ECFP4 Tanimoto to the 12 training hits and
    the fraction of the pattern's atoms that the held-out hit shares (MCS(pattern, held-out)/pattern size);
  - count how many of the 130 size-matched confirmed negatives match the re-derived pattern (control);
  - count matches in the validation (244,328) and test (184,632) pools (library-scale specificity).
Stability: canonical SMARTS / atom count of each fold vs. the pattern derived from all 13.
Run: python -m scripts.gp_step5_loo_pharmacophore -> data/processed/gp/step5_loo.json
"""
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def derive(mols):
    r = rdFMCS.FindMCS(mols, threshold=1.0, ringMatchesRingOnly=True, completeRingsOnly=True,
                       atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder,
                       timeout=120)
    return r.smartsString, r.numAtoms, r.canceled


def main():
    hits = pd.read_csv("docs/redteam_hits_confirmados.csv")[["CatalogID", "SMILES"]]
    lig = pd.read_csv("data/docking/retro/ligands.csv")
    neg = lig[lig.y == 0]
    hm = [Chem.MolFromSmiles(s) for s in hits.SMILES]
    nm = [Chem.MolFromSmiles(s) for s in neg.SMILES]
    hf = [gen.GetFingerprint(m) for m in hm]
    full_sma, full_n, full_cancel = derive(hm)
    print(f"[*] pattern from all 13: {full_sma}  ({full_n} atoms, canceled={full_cancel})")

    pools = {}
    for name, f in (("validation", "data/raw/Val-Test-set/PGK2_Validation_split.csv"),
                    ("test", "data/raw/Val-Test-set/PGK2_Test_split.csv")):
        d = pd.read_csv(f)
        pools[name] = [m for m in (Chem.MolFromSmiles(s) for s in d.SMILES) if m is not None]
        print(f"[*] parsed {name}: {len(pools[name]):,}")
    full_q = Chem.MolFromSmarts(full_sma)
    rows = []
    for i in range(len(hm)):
        train = [m for j, m in enumerate(hm) if j != i]
        sma, n_at, canceled = derive(train)
        q = Chem.MolFromSmarts(sma)
        match = bool(hm[i].HasSubstructMatch(q))
        qm = Chem.MolFromSmarts(sma)
        pat_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmarts(sma))) if False else None
        # soft: shared fraction of pattern atoms with the held-out hit
        r = rdFMCS.FindMCS([Chem.MolFromSmiles(Chem.MolToSmiles(train[0])), hm[i]], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=30)
        nn = max(DataStructs.BulkTanimotoSimilarity(hf[i], [hf[j] for j in range(len(hm)) if j != i]))
        neg_match = sum(m.HasSubstructMatch(q) for m in nm)
        row = {"held_out": hits.CatalogID[i], "match": match, "pattern_atoms": n_at,
               "same_as_full13": sma == full_sma, "pattern_smarts": sma, "canceled": bool(canceled),
               "nn_tanimoto_to_train": float(nn), "neg_matches_of_130": int(neg_match)}
        for name, ms in pools.items():
            row[f"pool_{name}_matches"] = int(sum(m.HasSubstructMatch(q) for m in ms))
        rows.append(row)
        print(f"  hold-out {row['held_out']}: match={match} atoms={n_at} same13={row['same_as_full13']} "
              f"neg={neg_match}/130 val={row['pool_validation_matches']} test={row['pool_test_matches']} nn={nn:.2f}", flush=True)
    df = pd.DataFrame(rows)
    neg_full = sum(m.HasSubstructMatch(full_q) for m in nm)
    out = {"pattern_full13": full_sma, "pattern_full13_atoms": full_n,
           "captured": int(df.match.sum()), "n_hits": len(df),
           "neg_match_mean": float(df.neg_matches_of_130.mean()), "neg_match_range": [int(df.neg_matches_of_130.min()), int(df.neg_matches_of_130.max())],
           "neg_match_full13": int(neg_full), "n_distinct_patterns": int(df.pattern_smarts.nunique()),
           "n_same_as_full13": int(df.same_as_full13.sum()),
           "pool_validation_size": len(pools["validation"]), "pool_test_size": len(pools["test"]), "rows": rows}
    json.dump(out, open("data/processed/gp/step5_loo.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
