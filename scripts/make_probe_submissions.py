"""
Builds "probe" submissions for the validation-split leaderboard: 50-compound
blocks taken at different depths of a model's full ranking of the official
validation pool, plus a random-50 control. The leaderboard's N_hits per
block estimates hit density vs. rank, i.e. whether the model carries any
signal at all beyond random selection, and where the real hits sit.

Uses the physchem+regularized XGBoost (models/pretrained/
pgk2_xgboost_asl_physchem_reg.json); the physchem StandardScaler is refit on
the same cached train rows used in training.
"""

import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.train_pgk2_full import featurize_parallel

MODEL = "models/pretrained/pgk2_xgboost_asl_physchem_reg.json"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
PHYSCHEM = "data/processed/pgk2_physchem.npz"
FULL_RANKING_CSV = "data/processed/pgk2_val_ranking_physchem_reg.csv"
PROBE_DIR = "submissions/probes"
BLOCK_STARTS = [51, 101, 251, 501, 1001, 5001, 20001, 100001]  # 1-indexed ranks


def main():
    train_idx = np.load(SPLIT_CACHE)["train_idx"]
    pc_train = np.load(PHYSCHEM)["X"][train_idx]
    scaler = StandardScaler().fit(pc_train)

    model = XGBClassifier()
    model.load_model(MODEL)

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
            pool = pd.read_csv(f)

    fps, pcs = featurize_parallel(pool["SMILES"].tolist(), want_physchem=True)
    ok = [fp is not None for fp in fps]
    pool = pool[ok].reset_index(drop=True)
    X = np.hstack([
        np.stack([fp for fp, k in zip(fps, ok) if k]),
        scaler.transform(np.stack([pc for pc, k in zip(pcs, ok) if k])).astype(np.float32),
    ])
    pool["Score"] = model.predict(X, output_margin=True)
    pool = pool.sort_values("Score", ascending=False).reset_index(drop=True)
    pool["Rank"] = np.arange(1, len(pool) + 1)
    pool.to_csv(FULL_RANKING_CSV, index=False)
    print(f"[*] Full ranking of {len(pool):,} compounds -> {FULL_RANKING_CSV}", flush=True)

    os.makedirs(PROBE_DIR, exist_ok=True)
    for start in BLOCK_STARTS:
        block = pool.iloc[start - 1:start - 1 + 50]
        assert block["CatalogID"].nunique() == 50
        path = f"{PROBE_DIR}/probe_rank{start}_{start + 49}.txt"
        block["CatalogID"].to_csv(path, index=False, header=False)
        print(f"    wrote {path}", flush=True)

    rng = np.random.RandomState(0)
    rand = pool.iloc[rng.choice(len(pool), 50, replace=False)]
    path = f"{PROBE_DIR}/probe_random50_control.txt"
    rand["CatalogID"].to_csv(path, index=False, header=False)
    print(f"    wrote {path}", flush=True)


if __name__ == "__main__":
    main()
