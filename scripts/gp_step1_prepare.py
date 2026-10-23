"""
Gamma-Poisson labeling, step 1: deduplicate the PGK2 DEL selection by SMILES
(counts summed, z-scores combined with Stouffer Z = sum(z)/sqrt(k), historic_hits
max) and merge the PGK2_NTC_supplement so the NTC background is not limited to
rows with count_PGK2 >= 1. Reports how the NTC distribution changes.

Output: data/processed/gp/step1_union.parquet (+ step1_report.json)
Run   : python -m scripts.gp_step1_prepare
"""
import json
import os

import numpy as np
import pandas as pd

RAW = "data/raw/PGK2_selection.parquet"
SUP = "data/raw/PGK2_NTC_supplement.parquet"
OUT_DIR = os.environ.get("GP_DIR", "data/processed/gp")  # GP_DIR=data/processed/gp_clean for the corrected run
COUNTS = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
ZS = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]


def dedup(df):
    """Vectorised equivalent of modules.denoising.stouffer.aggregate_del_duplicates."""
    g = df.groupby("SMILES", sort=False)
    out = g[COUNTS].sum()
    for z in ZS:
        k = g[z].count()
        out[z] = (g[z].sum() / np.sqrt(k.where(k > 0))).fillna(0.0)
    out["historic_hits"] = g["historic_hits"].max()
    out["compound"] = g["compound"].first()
    out["n_dup"] = g.size()
    return out.reset_index()


def ntc_summary(ntc):
    pos = ntc[ntc > 0]
    return {
        "n_compounds": int(len(ntc)),
        "n_ntc_pos": int(len(pos)),
        "frac_ntc_pos": float(len(pos) / len(ntc)),
        "sum_ntc_reads": int(ntc.sum()),
        "mean_all": float(ntc.mean()),
        "mean_pos": float(pos.mean()),
        "var_over_mean_pos": float(pos.var() / pos.mean()),
        "share_singletons_among_pos": float((pos == 1).mean()),
        "max": int(ntc.max()),
        "q99_pos": float(pos.quantile(0.99)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_parquet(RAW)
    n_raw = len(raw)
    # 28,214 raw rows have an EMPTY SMILES; grouping by SMILES fused them into one bogus row (count_NTC 2,323). Drop them.
    empty = raw.SMILES.isna() | (raw.SMILES.astype(str).str.strip() == "")
    n_empty, empty_pgk2, empty_ntc = int(empty.sum()), int(raw.loc[empty, "count_PGK2"].sum()), int(raw.loc[empty, "count_NTC"].sum())
    print(f"[*] dropping {n_empty:,} rows with empty SMILES (sum count_PGK2 {empty_pgk2:,}, sum count_NTC {empty_ntc:,})")
    raw = raw[~empty].reset_index(drop=True)
    sel = dedup(raw)
    print(f"[*] selection: {n_raw:,} rows -> {len(sel):,} unique SMILES ({100*(n_raw-len(sel))/n_raw:.2f}% merged)")

    sup = pd.read_parquet(SUP)
    sup_empty = sup.SMILES.isna() | (sup.SMILES.astype(str).str.strip() == "")
    print(f"[*] supplement rows with empty SMILES: {int(sup_empty.sum())}")
    sup = sup[~sup_empty].reset_index(drop=True)
    sup_g = sup.groupby("SMILES", sort=False)
    supd = sup_g[["count_NTC"]].sum()
    k = sup_g["zscore_NTC"].count()
    supd["zscore_NTC"] = sup_g["zscore_NTC"].sum() / np.sqrt(k)
    supd["compound"] = sup_g["compound"].first()
    supd = supd.reset_index()
    print(f"[*] supplement: {len(sup):,} rows -> {len(supd):,} unique SMILES")

    both = sel.merge(supd, on="SMILES", suffixes=("", "_sup"))
    same = int((both.count_NTC == both.count_NTC_sup).sum())
    print(f"[*] SMILES in both: {len(both):,}; count_NTC identical in {same:,} ({100*same/len(both):.1f}%)")
    print(f"    selection NTC>0 among the overlap: {(both.count_NTC>0).sum():,} | supplement NTC (all >=1)")

    only_sup = supd[~supd.SMILES.isin(sel.SMILES)].copy()
    # INFERENCE: supplement-only compounds are absent from a file filtered to count_PGK2>=1,
    # so their PGK2 / PGK2+inhibitor counts are 0 (not measured as missing).
    only_sup["count_PGK2"] = 0
    only_sup["count_PGK2_with_inhibitor"] = 0
    only_sup["zscore_PGK2"] = 0.0
    only_sup["zscore_PGK2_with_inhibitor"] = 0.0
    only_sup["historic_hits"] = np.nan
    only_sup["n_dup"] = 1
    only_sup["in_supplement_only"] = True
    sel["in_supplement_only"] = False
    # for overlapping SMILES keep the selection value, but never lower than the supplement
    upd = sel.SMILES.map(supd.set_index("SMILES").count_NTC)
    sel["count_NTC"] = np.maximum(sel.count_NTC, upd.fillna(0).astype(int))
    union = pd.concat([sel, only_sup[sel.columns]], ignore_index=True)
    union.to_parquet(f"{OUT_DIR}/step1_union.parquet", index=False)

    rep = {
        "n_raw_rows": n_raw, "n_empty_smiles_dropped": n_empty, "empty_sum_count_PGK2": empty_pgk2, "empty_sum_count_NTC": empty_ntc, "n_unique_selection": len(sel), "n_supplement_unique": len(supd),
        "overlap_smiles": len(both), "overlap_identical_ntc": same, "n_supplement_only": len(only_sup),
        "ntc_without_supplement": ntc_summary(sel.count_NTC),
        "ntc_with_supplement": ntc_summary(union.count_NTC),
        "ntc_supplement_only_positive_rows": ntc_summary(only_sup.count_NTC),
        "sum_count_PGK2": int(sel.count_PGK2.sum()), "sum_count_inh": int(sel.count_PGK2_with_inhibitor.sum()),
        "sum_count_NTC_with_supp": int(union.count_NTC.sum()),
    }
    json.dump(rep, open(f"{OUT_DIR}/step1_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
