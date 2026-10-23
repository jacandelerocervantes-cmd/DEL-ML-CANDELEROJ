"""
Derives real hit labels for the official PGK2 DEL selection
(`data/raw/PGK2_selection.parquet`, 7,703,070 rows / 7,487,567 unique
SMILES after dedup), following the AIRCHECK data-page methodology exactly.

Corrects two earlier mistakes in this file (kept as history, not hidden):
1. First attempt normalized count_PGK2 against per-row NTC via Poisson MAP
   (modules/denoising/stouffer.py) — degenerate here because this file is
   pre-filtered to count_PGK2>=1, so NTC is almost always exactly 0.
2. Second attempt used `historic_hits >= 1` as a POSITIVE signal ("this
   compound has a track record"). That was backwards: AIRCHECK's own docs
   define historic_hits as "Times this structure appeared as a hit in
   prior DEL campaigns (**promiscuity indicator**)" — a HIGH value means
   the compound is a serial non-specific "frequent hitter" across
   unrelated targets, i.e. evidence AGAINST it being a real PGK2 binder.
   The positive correlation we measured (0.123) was actually us rediscovering
   that sticky/promiscuous compounds tend to have both high counts AND high
   historic_hits — not validation of our (wrong) label.

Step 1 — Deduplicate by SMILES (AIRCHECK's documented aggregation scheme):
    count_*    -> sum
    zscore_*   -> Stouffer's Z combination
    historic_hits -> max (worst-case promiscuity across duplicate encodings)
    compound, SMILES -> first
(implemented in modules/denoising/stouffer.py::aggregate_del_duplicates,
already matching this exact scheme.)

Step 2 — Label using AIRCHECK's own recommended active-compound criteria,
broadened per the official challenge rules text ("ASMS hits confirmed in
the kinase assay are expected to occupy either the ATP (cofactor) site or
the phosphoglycerate (substrate) binding site" — i.e. a hit need NOT show
inhibitor competition, only orthosteric/ATP-site binders do):

    BCM's precise definition (orthosteric binders only):
        count_PGK2 >= 3 AND count_PGK2_with_inhibitor < 0.1 * count_PGK2
        AND count_NTC == 0 AND historic_hits < 5

    Our broadened definition (orthosteric OR allosteric binders — drops the
    inhibitor-competition requirement, which the rules explicitly say would
    exclude real allosteric/substrate-pocket hits):
        count_PGK2 >= 3 AND count_NTC == 0 AND historic_hits < 5

We label with the broadened definition by default (`--orthosteric-only` to
use BCM's narrower one) and always keep an `is_orthosteric` column so the
distinction survives into training as a feature, not just a label choice.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modules.denoising.stouffer import aggregate_del_duplicates

INPUT_PARQUET = "data/raw/PGK2_selection.parquet"
OUTPUT_PARQUET = "data/processed/PGK2_selection_labeled.parquet"
ZSCORE_COLS = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
COUNT_COLS = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]


def main(count_threshold: int = 3, historic_hits_max: int = 5, orthosteric_only: bool = False):
    print("=" * 70)
    print("[*] LABELING REAL PGK2 DEL SELECTION DATA (AIRCHECK-documented criteria)")
    print("=" * 70)

    cols = ["compound", "SMILES", *COUNT_COLS, *ZSCORE_COLS, "historic_hits"]
    df = pq.read_table(INPUT_PARQUET, columns=cols).to_pandas()
    n_raw = len(df)
    print(f"[*] Loaded {n_raw:,} raw rows.")

    print("[*] Deduplicating by SMILES (sum counts, Stouffer Z, max historic_hits)...")
    df = aggregate_del_duplicates(df, count_cols=COUNT_COLS, zscore_cols=ZSCORE_COLS)
    print(f"[*] {n_raw:,} -> {len(df):,} unique compounds ({100 * (n_raw - len(df)) / n_raw:.2f}% duplicates merged; "
          f"AIRCHECK reports 2.8% on the full file, consistent).")

    is_orthosteric = (
        (df["count_PGK2"] >= count_threshold)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * df["count_PGK2"])
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
    )
    is_allosteric_or_orthosteric = (
        (df["count_PGK2"] >= count_threshold)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < historic_hits_max)
    )
    df["is_orthosteric"] = is_orthosteric.astype(np.int8)
    df["Label"] = (is_orthosteric if orthosteric_only else is_allosteric_or_orthosteric).astype(np.int8)

    n_pos = int(df["Label"].sum())
    n_ortho = int(is_orthosteric.sum())
    print(f"[*] BCM orthosteric-only criteria: {n_ortho:,} compounds")
    print(f"[*] Broadened (orthosteric OR allosteric) criteria: {int(is_allosteric_or_orthosteric.sum()):,} compounds")
    print(f"[*] Using {'orthosteric-only' if orthosteric_only else 'broadened'} as Label==1: "
          f"{n_pos:,} compounds ({100 * n_pos / len(df):.3f}%)")

    df["RandomID"] = df["compound"]
    os.makedirs("data/processed", exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, OUTPUT_PARQUET)
    print(f"[*] Wrote labeled+deduplicated dataset to {OUTPUT_PARQUET}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label the real PGK2 DEL selection data")
    parser.add_argument("--orthosteric-only", action="store_true",
                         help="Use BCM's narrower orthosteric-binder-only criteria instead of the broadened one")
    args = parser.parse_args()
    main(orthosteric_only=args.orthosteric_only)
