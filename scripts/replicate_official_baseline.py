"""
Exact replication of the official Target 2035 Challenge #1 (WDR91) baseline,
to validate our pipeline apples-to-apples against a published reference
number before trusting it on PGK2.

Official baseline (from the challenge rules text):
    "A baseline LightGBM model trained on ECFP4 using all DEL-hits and 2
    times the number of negatives identified 5 hits, corresponding to 3
    chemical series, in the top 50 ranked molecules in the validation
    split, and 1 hit in the top 50 ranked molecules in the test split."

This script reproduces that setup as closely as the released data allows:
  - Model: LightGBM (not XGBoost) on ECFP4 (official precomputed column).
  - Training data: ALL 138 confirmed hits + exactly 2x that many negatives
    (276), sampled from the train partition only.
  - Evaluation: three-way Bemis-Murcko scaffold split (train / validation /
    test, zero scaffold overlap between any two — matching "no scaffold
    overlap between the two" from the official test-set description),
    scored against the FULL validation and test partitions (not a
    subsample), reporting hits-in-top-50 + distinct scaffolds ("series")
    for each, directly comparable to the 5/3 and 1 numbers above.

Caveat: the official challenge's actual validation/test split is a fixed,
externally-defined partition of the ASMS library; we don't have that exact
partition, only the reunified 339,258-row labeled set. Our train/val/test
here is our own scaffold-disjoint partition of the same molecules, so the
comparison is methodologically equivalent but not bit-for-bit the same
split the leaderboard used.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from rdkit import RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from modules.validation.scaffold_split import get_murcko_scaffold, three_way_scaffold_split

RDLogger.DisableLog("rdApp.*")

TRAIN_PARQUET = "data/raw/DREAM_Challenge_1_TrainSet.parquet"


def hits_in_top_n(scored_df: pd.DataFrame, n: int = 50, label_col: str = "Label", score_col: str = "Score", smiles_col: str = "SMILES"):
    top = scored_df.sort_values(score_col, ascending=False).head(n)
    hit_rows = top[top[label_col] == 1]
    n_hits = len(hit_rows)
    n_series = hit_rows[smiles_col].apply(get_murcko_scaffold).nunique() if n_hits else 0
    return n_hits, n_series


def main(neg_ratio: int = 2, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    print("=" * 70)
    print("[*] OFFICIAL BASELINE REPLICATION (WDR91, Challenge #1)")
    print("=" * 70)

    df = pq.read_table(TRAIN_PARQUET, columns=["SMILES", "RandomID", "LABEL", "ECFP4"]).to_pandas()
    df = df.rename(columns={"LABEL": "Label"})
    print(f"[*] Loaded {len(df)} compounds ({int(df['Label'].sum())} confirmed hits).")

    print("[*] Computing 3-way scaffold-disjoint split (train/val/test, stratified by Label)...")
    train_df, val_df, test_df = three_way_scaffold_split(df, val_ratio=val_ratio, test_ratio=test_ratio)
    print(f"[*] Train: {len(train_df)} ({int(train_df['Label'].sum())} hits) | "
          f"Val: {len(val_df)} ({int(val_df['Label'].sum())} hits) | "
          f"Test: {len(test_df)} ({int(test_df['Label'].sum())} hits)")

    pos_train = train_df[train_df["Label"] == 1]
    neg_pool = train_df[train_df["Label"] == 0]
    n_neg = min(len(pos_train) * neg_ratio, len(neg_pool))
    neg_train = neg_pool.sample(n=n_neg, random_state=seed)
    fit_df = pd.concat([pos_train, neg_train], ignore_index=True)
    print(f"[*] Fitting on official-baseline ratio: {len(pos_train)} hits + {len(neg_train)} negatives "
          f"({neg_ratio}x) = {len(fit_df)} training rows.")

    X_fit = np.stack(fit_df["ECFP4"].apply(lambda a: np.asarray(a, dtype=np.int16)).to_numpy())
    y_fit = fit_df["Label"].to_numpy()

    model = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, verbosity=-1)
    model.fit(X_fit, y_fit)

    for name, split_df in [("VALIDATION", val_df), ("TEST", test_df)]:
        X = np.stack(split_df["ECFP4"].apply(lambda a: np.asarray(a, dtype=np.int16)).to_numpy())
        scores = model.predict_proba(X)[:, 1]
        scored = split_df.assign(Score=scores)
        roc = roc_auc_score(scored["Label"], scores) if scored["Label"].nunique() == 2 else float("nan")
        prauc = average_precision_score(scored["Label"], scores) if scored["Label"].nunique() == 2 else float("nan")
        n_hits, n_series = hits_in_top_n(scored, n=50)
        print(f"\n[{name}] n={len(scored)} | hits available={int(scored['Label'].sum())} | "
              f"ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}")
        print(f"[{name}] hits-in-top-50 = {n_hits} ({n_series} distinct scaffolds/'series')")

    print("\n" + "=" * 70)
    print("[*] OFFICIAL BASELINE (reference): 5 hits / 3 series in top-50 validation, "
          "1 hit in top-50 test.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate the official WDR91 LightGBM baseline")
    parser.add_argument("--neg-ratio", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    main(neg_ratio=args.neg_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
