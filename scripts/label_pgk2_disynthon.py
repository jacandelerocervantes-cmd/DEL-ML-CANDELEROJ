"""
Disynthon-level Gamma-Poisson MAP relabeling for the PGK2 DEL selection data.

Problem this fixes (the "labels defectuosas" behind 22 zero-hit submissions):
label_pgk2_real_data.py's criteria (count_PGK2>=3, count_NTC==0,
historic_hits<5) trust each trisynthon's own raw read count. At DEL
sequencing depth most trisynthons get only a handful of reads from
Poisson/PCR sampling alone -- crossing count>=3 by chance, not because the
molecule binds PGK2. Root-cause evidence already on file: models trained on
this label learned to detect "is this a DEL-library synthesis-handle
artifact" (bits mapping to BB linker fragments) rather than a PGK2
pharmacophore, because the label itself carried little chemically-grounded
signal above sequencing noise (see docs/... domain-mask fix, 22 submissions
all scoring 0 hits on the official leaderboard).

Fix -- borrow statistical power across building-block PAIRS (disynthons):
a real PGK2-binding motif should show enrichment consistently when pooled
across the ~hundreds-to-thousands of trisynthons that share a given
building-block pair (much higher effective read depth per unit), not just
spike once at the single-trisynthon level. Standard DEL-ML denoising
practice; the machinery already existed unused in
modules/denoising/stouffer.py (aggregate_disynthon_counts,
calculate_poisson_map_enrichment) since before label_pgk2_real_data.py was
written.

Two-level hierarchical Gamma-Poisson MAP:
  Level 1 (disynthon): for every (library, pair-type, bb_i, bb_j), pool
    count_PGK2 / count_NTC / count_PGK2_with_inhibitor over every
    trisynthon sharing that pair. Gamma-Poisson MAP enrichment vs NTC
    (calculate_poisson_map_enrichment) gives a shrunk rate ratio + z-score
    per disynthon, backed by pooled reads instead of 1-2.
  Level 2 (trisynthon/compound): a compound's Label requires corroboration
    from >=N of its 3 disynthons (BB1-BB2, BB1-BB3, BB2-BB3) showing
    enrichment z-score >= Z_THRESH with disynthon support >= MIN_SUPPORT
    trisynthons, AND its own count_PGK2 >= COUNT_FLOOR, AND count_NTC==0,
    AND historic_hits < HISTORIC_HITS_MAX (unchanged -- promiscuity is
    orthogonal to the count-noise problem being fixed here).

This does not replace label_pgk2_real_data.py; it adds a statistical
corroboration gate on top of the same base criteria.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scipy.stats import norm

from modules.denoising.stouffer import (
    aggregate_del_duplicates,
    enrichment_pvalue,
    estimate_nb_dispersion,
)

INPUT_PARQUET = "data/raw/PGK2_selection.parquet"
OLD_LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"
OUTPUT_PARQUET = "data/processed/PGK2_selection_labeled_disynthon.parquet"
DISYNTHON_CSV = "data/processed/pgk2_disynthon_enrichment.csv"

COUNT_COLS = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
ZSCORE_COLS = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
PAIR_DEFS = [("bb1", "bb2"), ("bb1", "bb3"), ("bb2", "bb3")]

Z_THRESH = 3.0
MIN_SUPPORT = 30
MIN_CORROB = 2
COUNT_FLOOR = 2
HISTORIC_HITS_MAX = 5
PRIOR_STRENGTH = 50.0  # pseudo-trisynthons of pool-mean-rate weight in the MAP shrinkage


def build_disynthon_tables(df: pd.DataFrame, p0: float, alpha: float = 0.0) -> dict:
    """
    Pool-mean-rate empirical Bayes enrichment per disynthon (NOT vs. NTC --
    count_NTC is degenerate in this file: the raw selection is pre-filtered
    to count_PGK2>=1, so almost every row has count_NTC==0, already flagged
    as "Mistake #1" in label_pgk2_real_data.py's own docstring for the
    single-compound case; it is equally degenerate pooled at the disynthon
    level). Instead, each disynthon's rate is compared against p0, the
    pool-wide mean count_PGK2 per trisynthon -- the same reference used
    successfully by disynthon_score.py's building-block enrichment.

    This table pools ALL trisynthons sharing a disynthon (including whichever
    one a caller might later look up) -- it is the descriptive/export view
    (written to pgk2_disynthon_enrichment.csv) and MUST NOT be used to gate
    an individual trisynthon's own Label, because a single trisynthon's own
    reads are then part of the sum used to judge it (self-inflation: a
    jackpotted/PCR-artifact trisynthon with an extreme individual count can
    push its OWN disynthon over Z_THRESH by itself). For that, see
    `attach_loo_disynthon_scores`, which excludes each row's own
    contribution before scoring it -- the leave-one-out fix applied here
    after an audit found this file's Label gate was not doing that (unlike
    `soft_weights()` in scripts/global_score.py, which already did LOO
    correctly for its building-block rates).

    MAP rate = (observed + PRIOR_STRENGTH*p0) / (n_trisynthons + PRIOR_STRENGTH)
    enrichment = MAP rate / p0
    z-score = one-sided count test of observed count_PGK2 against the
      null "this disynthon's per-trisynthon rate equals the pool mean p0"
      (expected = n_trisynthons * p0), mapped to a z-score via the probit
      transform -- same construction as stouffer.py's
      poisson_enrichment_zscore, just against a pool-mean null instead of
      an NTC-derived one. The null is Negative-Binomial (dispersion=alpha)
      rather than plain Poisson whenever `alpha` > 0: per-trisynthon
      binding/PCR efficiency varies around p0 for reasons that have
      nothing to do with true PGK2 binding, so the real null is
      overdispersed relative to Poisson; testing against plain Poisson
      understates that variance and inflates z-scores (audit finding --
      see `estimate_nb_dispersion` in stouffer.py). alpha=0.0 (default)
      reproduces the original plain-Poisson behavior exactly.
    """
    tables = {}
    for a, b in PAIR_DEFS:
        key = f"{a}_{b}"
        g = df.groupby(["lib", a, b], as_index=False).agg(
            count_PGK2=("count_PGK2", "sum"),
            count_PGK2_with_inhibitor=("count_PGK2_with_inhibitor", "sum"),
            count_NTC=("count_NTC", "sum"),
            n_trisynthons=("count_PGK2", "size"),
        )
        observed = g["count_PGK2"].to_numpy(dtype=float)
        n = g["n_trisynthons"].to_numpy(dtype=float)
        g["rate_map"] = (observed + PRIOR_STRENGTH * p0) / (n + PRIOR_STRENGTH)
        g["enrichment"] = g["rate_map"] / p0
        expected = np.maximum(n * p0, 1e-9)
        # `alpha` is the PER-TRISYNTHON dispersion; this is a POOLED SUM over
        # n trisynthons, so the sum's own dispersion is alpha/n (see
        # estimate_nb_dispersion's docstring) -- using unscaled alpha here
        # would overstate the pooled null's variance and make the disynthon
        # test spuriously more conservative the larger a disynthon's support.
        dispersion = alpha / np.maximum(n, 1.0)
        p_value = enrichment_pvalue(observed, expected, dispersion)
        g["enrichment_zscore"] = -norm.ppf(p_value)
        g["pair_type"] = key
        tables[key] = g
        n_enriched = int((g["enrichment_zscore"] >= Z_THRESH).sum())
        n_supported = int(((g["enrichment_zscore"] >= Z_THRESH) & (g["n_trisynthons"] >= MIN_SUPPORT)).sum())
        print(f"    {key}: {len(g):,} disynthons, {n_enriched:,} with z>={Z_THRESH} "
              f"({n_supported:,} also have support>={MIN_SUPPORT}); "
              f"z percentiles [50/90/99/99.9]={np.percentile(g['enrichment_zscore'], [50, 90, 99, 99.9]).round(2)}",
              flush=True)
    return tables


def attach_loo_disynthon_scores(df: pd.DataFrame, p0: float, alpha: float = 0.0) -> pd.DataFrame:
    """
    Per-compound, LEAVE-ONE-OUT version of the disynthon enrichment z-score
    used in `build_disynthon_tables`: for a trisynthon's own (z_<pair>,
    n_<pair>) it excludes that trisynthon's own count_PGK2 and its own unit
    from the disynthon's total before computing the MAP rate / z-score, so a
    trisynthon can never contribute to "corroborating" its own Label.

    Fix for the self-inflation bug: `build_disynthon_tables` pools every
    trisynthon sharing a disynthon INCLUDING the one being scored. At
    MIN_SUPPORT=30, a single trisynthon is normally only ~1/30 of the group,
    but a PCR/sequencing jackpot (a trisynthon with a wildly disproportionate
    individual read count -- a known DEL artifact, not evidence of binding)
    can dominate its own disynthon's sum on its own, pushing z above
    Z_THRESH and satisfying the corroboration gate using nothing but its own
    suspect count. Leave-one-out removes that path structurally: a
    trisynthon's Label can now only be corroborated by ITS NEIGHBORS'
    evidence, never by itself.

    Also computes `flag_possible_jackpot_<pair>`: True when a trisynthon's
    own count_PGK2 is >10x the (whole-group, non-LOO) per-trisynthon median
    of its disynthon, yet its own LOO enrichment z-score for that disynthon
    is below Z_THRESH -- i.e. an extreme individual read count with no
    corroboration from its neighbors, the profile expected of a PCR/sequencing
    jackpot rather than a real binder. (The group median includes the row
    itself; with MIN_SUPPORT-sized groups of >=30 members this has a
    negligible effect on the median compared to a true leave-one-out median,
    and is far cheaper to compute.)
    """
    out = df.copy()
    for a, b in PAIR_DEFS:
        key = f"{a}_{b}"
        zc, nc, jc = f"z_{key}", f"n_{key}", f"flag_possible_jackpot_{key}"

        grp = out.groupby(["lib", a, b])["count_PGK2"]
        group_sum = grp.transform("sum")
        group_size = grp.transform("size")
        group_median = grp.transform("median")

        observed_loo = (group_sum - out["count_PGK2"]).to_numpy(dtype=float)
        n_loo = (group_size - 1).to_numpy(dtype=float)
        n_loo_safe = np.maximum(n_loo, 0.0)

        expected = np.maximum(n_loo_safe * p0, 1e-9)
        # Same pooled-sum dispersion scaling as build_disynthon_tables (alpha/n,
        # not alpha) -- here n is n_loo_safe (n_loo==0 rows are floored to 1 to
        # avoid a division by zero; their z-score is overwritten to -inf below
        # regardless, so the exact value doesn't matter for them).
        dispersion = alpha / np.maximum(n_loo_safe, 1.0)
        p_value = enrichment_pvalue(observed_loo, expected, dispersion)
        z_loo = -norm.ppf(p_value)
        # A trisynthon alone in its disynthon (n_loo==0) has no neighbors at
        # all to corroborate it -- treat as unsupported, not "infinitely
        # enriched" (enrichment_pvalue with expected~0 would otherwise blow up).
        z_loo = np.where(n_loo_safe > 0, z_loo, -np.inf)

        out[zc] = z_loo
        out[nc] = n_loo_safe

        own_count = out["count_PGK2"].to_numpy(dtype=float)
        median = group_median.to_numpy(dtype=float)
        is_outlier = own_count > 10.0 * np.maximum(median, 1e-9)
        out[jc] = is_outlier & (z_loo < Z_THRESH)

    return out


def main(count_floor=COUNT_FLOOR, historic_hits_max=HISTORIC_HITS_MAX,
         z_thresh=Z_THRESH, min_support=MIN_SUPPORT, min_corrob=MIN_CORROB):
    print("=" * 70)
    print("[*] DISYNTHON-LEVEL GAMMA-POISSON MAP RELABELING")
    print("=" * 70)

    cols = ["compound", "SMILES", *COUNT_COLS, *ZSCORE_COLS, "historic_hits"]
    df = pq.read_table(INPUT_PARQUET, columns=cols).to_pandas()
    n_raw = len(df)
    print(f"[*] Loaded {n_raw:,} raw rows.")

    print("[*] Deduplicating by SMILES...")
    df = aggregate_del_duplicates(df, count_cols=COUNT_COLS, zscore_cols=ZSCORE_COLS)
    print(f"[*] {n_raw:,} -> {len(df):,} unique compounds "
          f"({100 * (n_raw - len(df)) / n_raw:.2f}% duplicates merged)")

    parts = df["compound"].str.split("-", expand=True)
    df["lib"], df["bb1"], df["bb2"], df["bb3"] = parts[0], parts[1], parts[2], parts[3]

    n_ntc_nonzero = int((df["count_NTC"] > 0).sum())
    p0 = float(df["count_PGK2"].sum()) / len(df)
    print(f"[*] count_NTC>0 for {n_ntc_nonzero:,}/{len(df):,} compounds "
          f"({100 * n_ntc_nonzero / len(df):.3f}%) -- confirms NTC background is degenerate here; "
          f"using pool-mean rate p0={p0:.4f} reads/trisynthon as the enrichment null instead.")

    alpha = estimate_nb_dispersion(df["count_PGK2"].to_numpy())
    var_over_mean = float(df["count_PGK2"].var(ddof=1) / df["count_PGK2"].mean())
    print(f"[*] [FIX: Poisson->Negative-Binomial] count_PGK2 Var/mean={var_over_mean:.3f} "
          f"(1.0 == Poisson); trimmed-moments NB dispersion alpha={alpha:.6g}. "
          + ("Plain Poisson understated the null variance -- using NB null for the "
             "enrichment z-score below (audit finding)." if alpha > 1e-8 else
             "Data is consistent with Poisson (alpha~0); NB null reduces to Poisson."))

    print("[*] Level 1: building per-disynthon pool-mean-rate MAP enrichment tables "
          "(POOLED, self-inclusive -- descriptive/export view only, see below)...")
    disynthon_tables = build_disynthon_tables(df, p0, alpha)

    print("[*] Level 2a [OLD, self-inflating -- kept only for the before/after comparison]: "
          "attaching each compound's 3 POOLED (self-inclusive) disynthon z-scores...")
    z_cols_pooled, n_cols_pooled = [], []
    for a, b in PAIR_DEFS:
        key = f"{a}_{b}"
        zc, nc = f"z_pooled_{key}", f"n_pooled_{key}"
        t = disynthon_tables[key][["lib", a, b, "enrichment_zscore", "n_trisynthons"]].rename(
            columns={"enrichment_zscore": zc, "n_trisynthons": nc}
        )
        df = df.merge(t, on=["lib", a, b], how="left")
        z_cols_pooled.append(zc)
        n_cols_pooled.append(nc)
    corrob_pooled = np.zeros(len(df), dtype=int)
    for zc, nc in zip(z_cols_pooled, n_cols_pooled):
        corrob_pooled += ((df[zc] >= z_thresh) & (df[nc] >= min_support)).fillna(False).astype(int)
    old_is_label = (
        (df["count_PGK2"] >= count_floor)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
        & (corrob_pooled >= min_corrob)
    )

    print("[*] Level 2b [FIX: leave-one-out]: attaching each compound's 3 disynthon z-scores "
          "computed EXCLUDING the compound's own reads (see attach_loo_disynthon_scores docstring "
          "for why the pooled version above lets a single jackpotted trisynthon corroborate itself)...")
    df = attach_loo_disynthon_scores(df, p0, alpha)
    z_cols = [f"z_{a}_{b}" for a, b in PAIR_DEFS]
    n_cols = [f"n_{a}_{b}" for a, b in PAIR_DEFS]
    jackpot_cols = [f"flag_possible_jackpot_{a}_{b}" for a, b in PAIR_DEFS]

    corrob = np.zeros(len(df), dtype=int)
    for zc, nc in zip(z_cols, n_cols):
        corrob += ((df[zc] >= z_thresh) & (df[nc] >= min_support)).fillna(False).astype(int)
    df["n_corroborating_disynthons"] = corrob
    df["min_disynthon_zscore"] = df[z_cols].min(axis=1)
    df["flag_possible_jackpot"] = df[jackpot_cols].any(axis=1)

    is_orthosteric_count = (
        (df["count_PGK2"] >= count_floor)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * df["count_PGK2"])
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
    )
    is_label = (
        (df["count_PGK2"] >= count_floor)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
        & (df["n_corroborating_disynthons"] >= min_corrob)
    )
    df["is_orthosteric"] = is_orthosteric_count.astype(np.int8)
    df["Label"] = is_label.astype(np.int8)
    df["RandomID"] = df["compound"]

    n_pos = int(df["Label"].sum())
    n_pos_old = int(old_is_label.sum())
    n_count_only = int(((df["count_PGK2"] >= count_floor) & (df["count_NTC"] == 0)
                         & (df["historic_hits"] < historic_hits_max)).sum())
    lost_to_loo = int((old_is_label & ~is_label).sum())
    gained_by_loo = int((~old_is_label & is_label).sum())
    n_jackpot_among_old_positives = int((old_is_label & df["flag_possible_jackpot"]).sum())
    print(f"[*] count-only floor (count_PGK2>={count_floor}, NTC==0, historic_hits<{historic_hits_max}): "
          f"{n_count_only:,} pass")
    print(f"[*] + disynthon corroboration, OLD self-inclusive gate (>={min_corrob}/3 pairs, z>={z_thresh}, "
          f"support>={min_support}): {n_pos_old:,} compounds")
    print(f"[*] + disynthon corroboration, FIXED leave-one-out gate (same thresholds): "
          f"{n_pos:,} compounds ({100 * n_pos / len(df):.4f}%)")
    print(f"[*] LEAVE-ONE-OUT FIX IMPACT: {lost_to_loo:,} compounds that were Label==1 under the "
          f"self-inclusive gate LOSE their label once their own reads are excluded from the "
          f"corroboration evidence (i.e. they were corroborating themselves) -- these are the "
          f"best estimate of how many 'positives' were actually single-trisynthon artifacts. "
          f"{gained_by_loo:,} compounds gain Label==1 that didn't have it before (their neighbors "
          f"corroborate them even though their own pooled contribution wasn't enough on its own).")
    print(f"[*] Of the {n_pos_old:,} old-gate positives, {n_jackpot_among_old_positives:,} "
          f"({100 * n_jackpot_among_old_positives / max(n_pos_old, 1):.2f}%) are flagged "
          f"flag_possible_jackpot (own count >10x their disynthon's median, but LOO z-score below "
          f"Z_THRESH once their own reads are excluded) -- suspected PCR/sequencing amplification "
          f"artifacts, not corroborated binding evidence.")

    print("[*] [Poisson-vs-NB ablation] recomputing the LOO corroboration gate with the OLD "
          "plain-Poisson null (alpha=0) instead of this fix's NB null, holding the LOO fix fixed, "
          "to isolate this fix's own impact from the LOO fix above...")
    loo_poisson = attach_loo_disynthon_scores(df, p0, 0.0)
    corrob_poisson = np.zeros(len(df), dtype=int)
    for zc, nc in zip(z_cols, n_cols):
        corrob_poisson += ((loo_poisson[zc] >= z_thresh) & (loo_poisson[nc] >= min_support)).fillna(False).astype(int)
    is_label_poisson_loo = (
        (df["count_PGK2"] >= count_floor)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
        & (corrob_poisson >= min_corrob)
    )
    n_pos_poisson_loo = int(is_label_poisson_loo.sum())
    lost_to_nb = int((is_label_poisson_loo & ~is_label).sum())
    gained_by_nb = int((~is_label_poisson_loo & is_label).sum())
    print(f"[*] LOO gate with OLD Poisson null: {n_pos_poisson_loo:,} compounds")
    print(f"[*] LOO gate with THIS FIX's NB null: {n_pos:,} compounds "
          f"({100 * n_pos / len(df):.4f}%)")
    print(f"[*] POISSON->NB FIX IMPACT: {lost_to_nb:,} compounds LOSE Label under the "
          f"overdispersion-aware NB null (they only cleared Z_THRESH because plain Poisson "
          f"understated the null variance -- the audit's over-liberal-gate finding); "
          f"{gained_by_nb:,} gain Label (NB shifted which disynthon is each compound's "
          f"binding min_disynthon_zscore across the >={min_corrob}/3 corroboration requirement).")

    if os.path.exists(OLD_LABELED_PARQUET):
        old = pq.read_table(OLD_LABELED_PARQUET, columns=["RandomID", "Label"]).to_pandas()
        old = old.rename(columns={"Label": "OldLabel"})
        cmp = df[["RandomID", "Label"]].merge(old, on="RandomID", how="inner")
        both = int(((cmp.Label == 1) & (cmp.OldLabel == 1)).sum())
        old_only = int(((cmp.Label == 0) & (cmp.OldLabel == 1)).sum())
        new_only = int(((cmp.Label == 1) & (cmp.OldLabel == 0)).sum())
        print(f"[*] Overlap with old (count-threshold-only) labels: "
              f"{both:,} in both, {old_only:,} old-only (dropped by disynthon gate), "
              f"{new_only:,} new-only (rescued by disynthon corroboration)")

    os.makedirs("data/processed", exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, OUTPUT_PARQUET)
    print(f"[*] Wrote labeled+deduplicated dataset to {OUTPUT_PARQUET}")

    combined = pd.concat(disynthon_tables.values(), ignore_index=True)
    combined.to_csv(DISYNTHON_CSV, index=False)
    print(f"[*] Wrote {len(combined):,} disynthon enrichment rows to {DISYNTHON_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disynthon-level Gamma-Poisson MAP relabeling for PGK2")
    parser.add_argument("--count-floor", type=int, default=COUNT_FLOOR)
    parser.add_argument("--historic-hits-max", type=int, default=HISTORIC_HITS_MAX)
    parser.add_argument("--z-thresh", type=float, default=Z_THRESH)
    parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--min-corrob", type=int, default=MIN_CORROB)
    args = parser.parse_args()
    main(count_floor=args.count_floor, historic_hits_max=args.historic_hits_max,
         z_thresh=args.z_thresh, min_support=args.min_support, min_corrob=args.min_corrob)
