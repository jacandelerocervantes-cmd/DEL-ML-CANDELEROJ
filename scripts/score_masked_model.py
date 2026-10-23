"""
Scores the official PGK2_Validation_split.csv pool with the domain-mask +
physchem XGBoost model (models/pretrained/pgk2_xgboost_asl_mask_physchem_reg.json),
i.e. ECFP4 restricted to bits that are NOT DEL-library artifacts (see
scripts/make_domain_mask.py) plus the 8 physchem descriptors.

Writes:
  - data/processed/pgk2_val_ranking_mask_physchem_reg.csv (full ranked pool,
    for future rank-depth probes)
  - submissions/Team_candeleroj_<submission_index>.txt (top-50)

Run on the GCP VM where the cached artifacts already exist:
  data/processed/pgk2_scaffold_split_idx.npz
  data/processed/pgk2_physchem.npz
  data/processed/pgk2_bit_mask.npz
  data/raw/Val-Test-set.zip
  models/pretrained/pgk2_xgboost_asl_mask_physchem_reg.json
"""

import argparse
import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.train_pgk2_full import featurize_parallel

MODEL = "models/pretrained/pgk2_xgboost_asl_mask_physchem_reg.json"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
PHYSCHEM = "data/processed/pgk2_physchem.npz"
BIT_MASK = "data/processed/pgk2_bit_mask.npz"
FULL_RANKING_CSV = "data/processed/pgk2_val_ranking_mask_physchem_reg.csv"


def main(submission_index: int, team_name: str):
    train_idx = np.load(SPLIT_CACHE)["train_idx"]
    pc_train = np.load(PHYSCHEM)["X"][train_idx]
    scaler = StandardScaler().fit(pc_train)

    keep_cols = np.load(BIT_MASK)["keep"]
    print(f"[*] Domain mask: {len(keep_cols)}/2048 ECFP4 bits kept", flush=True)

    model = XGBClassifier()
    model.load_model(MODEL)

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
            pool = pd.read_csv(f)

    print(f"[*] Featurizing {len(pool):,} official validation candidates...", flush=True)
    fps, pcs = featurize_parallel(pool["SMILES"].tolist(), want_physchem=True)
    ok = [fp is not None for fp in fps]
    pool = pool[ok].reset_index(drop=True)
    fp_masked = np.stack([fp for fp, k in zip(fps, ok) if k])[:, keep_cols]
    pc_scaled = scaler.transform(np.stack([pc for pc, k in zip(pcs, ok) if k])).astype(np.float32)
    X = np.hstack([fp_masked, pc_scaled])

    pool["Score"] = model.predict(X, output_margin=True)
    pool = pool.sort_values("Score", ascending=False).reset_index(drop=True)
    pool["Rank"] = np.arange(1, len(pool) + 1)
    os.makedirs(os.path.dirname(FULL_RANKING_CSV), exist_ok=True)
    pool.to_csv(FULL_RANKING_CSV, index=False)
    print(f"[*] Full masked ranking of {len(pool):,} compounds -> {FULL_RANKING_CSV}", flush=True)

    top50 = pool.head(50)
    os.makedirs("submissions", exist_ok=True)
    out_path = f"submissions/Team_{team_name}_{submission_index}.txt"
    top50["CatalogID"].to_csv(out_path, index=False, header=False)
    print(f"[*] Top-50 (domain-masked model) -> {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-index", type=int, default=22)
    parser.add_argument("--team-name", default="candeleroj")
    args = parser.parse_args()
    main(submission_index=args.submission_index, team_name=args.team_name)
