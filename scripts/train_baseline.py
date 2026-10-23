"""
Advanced End-to-End Pipeline for Target 2035 DEL-ML (PGK2)

Wires together every SOTA component reviewed in
`DEL-ML Drug Design Research.docx` that is feasible without the official
AIRCHECK/HitGen data (pending platform approval):

  1. Disynthon aggregation (modules/denoising/stouffer.py) - skipped
     automatically if the input lacks per-cycle building-block columns.
  2. Gamma-Poisson MAP enrichment + Poisson z-score (replaces fold-change).
  3. Datamol-parallelized Bemis-Murcko scaffold split (no data leakage).
  4. HitGen-matching ECFP4 fingerprints (tools/Target2035_Aircheck_Utils),
     so features line up exactly with the official pre-computed fingerprint
     columns once the real parquet files are downloaded.
  5. XGBoost trained with Asymmetric Loss (modules/losses/asymmetric_loss.py)
     instead of plain cross-entropy, to survive the >99.9% negative skew.
  6. Ranking + official-format submission generation (modules/submission/
     validator.py): a plain list of exactly 50 predicted hit compound IDs,
     per docs/operativa/TARGET2035-CHALLENGE2-PGK2-REGLAS-OFICIALES.md.

WHAT IS STILL A PLACEHOLDER
----------------------------
`data/reference/DEL_hit_candidates_CACHE7.csv` contains only *positive*
PGK2 binders (no NTC/inhibitor conditions, no building-block IDs, no
confirmed negatives). Real AIRCHECK data is required for a scientifically
meaningful model. Until then, this script:
  - synthesizes a small negative set from `data/reference/
    chembl_repurposing_library.json` purely so the pipeline runs
    end-to-end and every module gets exercised — these are NOT valid DEL
    negatives (wrong chemical domain, tiny sample size) and the resulting
    model must not be trusted for real hit calling.
  - once `data/raw/PGK2_selection.parquet` (train) and the ASMS
    validation/test parquet exist, set --train-parquet / --test-parquet
    and the placeholder path is bypassed entirely.

Usage
-----
    python scripts/train_baseline.py
    python scripts/train_baseline.py --train-parquet data/raw/PGK2_selection.parquet \
                                      --test-parquet data/raw/PGK2_ASMS_test.parquet
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from modules.denoising.stouffer import aggregate_del_duplicates, calculate_poisson_map_enrichment
from modules.losses.asymmetric_loss import make_asymmetric_objective
from modules.submission.validator import generate_submission_txt, top_n_discovered_compounds
from modules.validation.scaffold_split import scaffold_split

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_ecfp4(smiles: str, n_bits: int = 2048):
    """ECFP4 (radius=2, 2048 bits), matching HitGenECFP4's settings so
    features are directly comparable to the official pre-computed
    fingerprints once real AIRCHECK data is available."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = _morgan_gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def featurize(df: pd.DataFrame, smiles_col: str = "SMILES"):
    fps, valid_idx = [], []
    for idx, smi in df[smiles_col].items():
        fp = smiles_to_ecfp4(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)
    return np.array(fps), df.loc[valid_idx].reset_index(drop=True)


def load_placeholder_dataset(positives_csv: str, decoys_json: str) -> pd.DataFrame:
    """
    PLACEHOLDER ONLY. Builds a minimal binary-labeled dataset from the
    public CACHE7 PGK2 hit list (label=1) and the unrelated ChEMBL
    repurposing library (label=0, purely to exercise the pipeline).
    Real training must use the AIRCHECK PGK2 DEL selection instead.
    """
    pos = pd.read_csv(positives_csv)
    pos = pos.rename(columns={"count_PGK2": "PGK2"})
    pos["Label"] = 1

    with open(decoys_json, "r", encoding="utf-8") as f:
        decoys_raw = json.load(f)["compounds"]
    decoys = pd.DataFrame(
        [{"SMILES": c["canonical_smiles"], "PGK2": 0, "Label": 0} for c in decoys_raw if c.get("canonical_smiles")]
    )

    print(
        f"[!] PLACEHOLDER DATA: {len(pos)} real PGK2 hits (label=1) vs. "
        f"{len(decoys)} unrelated ChEMBL decoys (label=0). This mix is only "
        "for exercising the pipeline end-to-end and is NOT a scientifically "
        "valid training set — swap in --train-parquet as soon as AIRCHECK "
        "access is granted."
    )
    df = pd.concat([pos, decoys], ignore_index=True)
    df["RandomID"] = [f"CACHE7_{i}" for i in range(len(df))]
    return df


def load_official_dataset(train_parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(train_parquet)
    required = {"SMILES", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{train_parquet} is missing expected columns {missing}. "
            "Confirm the AIRCHECK schema (SMILES, Label, RandomID, and "
            "optionally PGK2/PGK2_inhibitor/NTC + BB1_ID/BB2_ID/BB3_ID)."
        )
    if "RandomID" not in df.columns:
        df["RandomID"] = [f"ROW_{i}" for i in range(len(df))]
    return df


def run_pipeline(
    train_parquet: str = None,
    test_parquet: str = None,
    positives_csv: str = "data/reference/DEL_hit_candidates_CACHE7.csv",
    decoys_json: str = "data/reference/chembl_repurposing_library.json",
    team_name: str = "HSelithBio",
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
):
    print("=" * 70)
    print("[*] TARGET 2035 PGK2 DEL-ML - ADVANCED PIPELINE")
    print("=" * 70)

    # --- 1. Load ---------------------------------------------------------
    if train_parquet and os.path.exists(train_parquet):
        print(f"[*] Loading OFFICIAL AIRCHECK training data: {train_parquet}")
        df = load_official_dataset(train_parquet)
    else:
        if train_parquet:
            print(f"[!] --train-parquet '{train_parquet}' not found, falling back to placeholder data.")
        df = load_placeholder_dataset(positives_csv, decoys_json)

    # --- 2. Denoising / enrichment (only when raw multi-condition + BB
    #        columns are present, i.e. the real AIRCHECK selection file) --
    if {"PGK2", "NTC"}.issubset(df.columns):
        print("[*] Computing Gamma-Poisson MAP enrichment + Poisson z-scores...")
        df = calculate_poisson_map_enrichment(df)
    else:
        print("[*] Skipping enrichment scoring: no NTC/competitor columns in this dataset "
              "(expected once the real PGK2_selection.parquet is used).")

    df = aggregate_del_duplicates(df, count_cols=[c for c in ["PGK2", "PGK2_inhibitor", "NTC"] if c in df.columns])
    # aggregate_del_duplicates drops Label/RandomID via groupby; recover them.
    if "Label" not in df.columns:
        raise RuntimeError("Label column lost during deduplication — check aggregate_del_duplicates call.")

    # --- 3. Featurize ------------------------------------------------------
    print("[*] Extracting 2048-bit ECFP4 fingerprints (HitGen-matching settings)...")
    X, df = featurize(df)
    y = df["Label"].to_numpy()
    print(f"[*] Featurized {len(X)} valid molecules | positives={int(y.sum())} negatives={int((1 - y).sum())}")

    # --- 4. Scaffold split (OOD-safe validation) ---------------------------
    print("[*] Performing datamol-parallelized Bemis-Murcko scaffold split (80/20)...")
    train_df, val_df = scaffold_split(df, smiles_col="SMILES", val_ratio=0.2, seed=42)
    X_train, y_train = X[train_df.index], train_df["Label"].to_numpy()
    X_val, y_val = X[val_df.index], val_df["Label"].to_numpy()
    print(f"[*] Train: {len(train_df)} compounds | Val: {len(val_df)} compounds")

    if len(set(y_train)) < 2:
        print("[!] Training split has only one class present — cannot fit a classifier yet. "
              "This will resolve once real negatives (NTC/decoy reads) are available.")
        return

    # --- 5. Train with Asymmetric Loss --------------------------------------
    print(f"[*] Training XGBoost with Asymmetric Loss (gamma_neg={asl_gamma_neg}, clip={asl_clip})...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=make_asymmetric_objective(gamma_pos=0.0, gamma_neg=asl_gamma_neg, clip=asl_clip),
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- 6. Evaluate ---------------------------------------------------------
    val_scores = model.predict(X_val, output_margin=True)
    if len(set(y_val)) == 2:
        roc = roc_auc_score(y_val, val_scores)
        prauc = average_precision_score(y_val, val_scores)
        print(f"[*] Validation ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}")
    else:
        print("[!] Validation split has only one class present — skipping ROC/PR-AUC.")

    # --- 7. Score the candidate pool and rank ---------------------------------
    if test_parquet and os.path.exists(test_parquet):
        print(f"[*] Scoring OFFICIAL ASMS candidate pool: {test_parquet}")
        test_df = pd.read_parquet(test_parquet)
        X_test, test_df = featurize(test_df)
        scores = model.predict(X_test, output_margin=True)
        ranked = test_df.assign(Score=scores)
    else:
        print("[!] No --test-parquet given: ranking the held-out validation split instead, "
              "as a stand-in for 'the list of compounds we discovered'.")
        ranked = val_df.assign(Score=val_scores)

    discovered = top_n_discovered_compounds(ranked, n=min(50, len(ranked)))
    print("[*] Top 50 discovered candidates (the official submission unit):")
    print(discovered[["RandomID", "SMILES", "Score"]].to_string(index=False))

    if len(ranked) >= 50:
        path = generate_submission_txt(ranked, team_name=team_name)
        print(f"[*] Official-format submission (50 hit IDs) written to: {path}")
    else:
        out_path = os.path.join("submissions", f"discovered_candidates_{team_name}.csv")
        os.makedirs("submissions", exist_ok=True)
        discovered.to_csv(out_path, index=False)
        print(
            f"[!] Fewer than 50 scored candidates ({len(ranked)}) — cannot populate a full "
            f"submission yet. Wrote the ranked candidate list instead: {out_path}"
        )

    print("=" * 70)
    return discovered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target 2035 PGK2 DEL-ML advanced pipeline")
    parser.add_argument("--train-parquet", default=None, help="Path to official PGK2_selection.parquet")
    parser.add_argument("--test-parquet", default=None, help="Path to official ASMS validation/test parquet")
    parser.add_argument("--team-name", default="HSelithBio")
    args = parser.parse_args()
    run_pipeline(train_parquet=args.train_parquet, test_parquet=args.test_parquet, team_name=args.team_name)
