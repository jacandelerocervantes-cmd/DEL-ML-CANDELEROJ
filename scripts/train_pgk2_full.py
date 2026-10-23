"""
Full-scale PGK2 training: XGBoost + Asymmetric Loss on the real,
labeled+deduplicated+featurized DEL selection (7,487,570 compounds,
27,650 candidate hits, ECFP4 precomputed by scripts/fingerprint_pgk2.py).

Meant to run on the GCP CPU VM (data/processed/PGK2_selection_features.parquet
must exist there — see fingerprint_pgk2.py).

Steps:
  1. Stratified Bemis-Murcko scaffold split (train/val, no scaffold leakage,
     per-class so the rare positives aren't skewed into one side) — done on
     lightweight columns only (SMILES/Label), never the fingerprints.
  2. Train XGBoost with Asymmetric Loss (modules/losses/asymmetric_loss.py)
     for the ~270:1 imbalance.
  3. Evaluate ROC-AUC / PR-AUC on the held-out scaffold split (our own
     sanity check — NOT the official blind validation split, which has no
     public labels).
  4. Score the OFFICIAL unlabeled `PGK2_Validation_split.csv` candidate
     pool, rank, and write the top-50 as a challenge-format submission
     (modules/submission/validator.py: 50 IDs, one per line).

MEMORY NOTE: ECFP4 is sparse (each molecule has ~50-150 nonzero bits out of
2048), but scripts/fingerprint_pgk2.py stored it densely in the parquet
file (a mistake — fine for computing it in parallel, wrong for loading it
back). Loading 7.49M x 2048 int16 densely (~31GB) OOM-killed this script on
the 32GB VM. This version reads the parquet in row-group chunks and
converts each chunk to a `scipy.sparse.csr_matrix` immediately, keeping the
data sparse (roughly 100x smaller) for the rest of the pipeline; XGBoost's
sklearn API accepts sparse matrices natively.
"""

import argparse
import os
import sys
import time
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from joblib import Parallel, delayed
from rdkit import RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from modules.losses.asymmetric_loss import make_asymmetric_objective
from modules.submission.validator import generate_submission_txt, top_n_discovered_compounds
from modules.validation.scaffold_split import scaffold_split

RDLogger.DisableLog("rdApp.*")

FEATURES_PARQUET = "data/processed/PGK2_selection_features.parquet"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
MODEL_OUT = "models/pretrained/pgk2_xgboost_asl.json"


PHYSCHEM_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3",
]


def _ecfp4_physchem_batch(smiles_batch, want_physchem: bool):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    if want_physchem:
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, pcs = [], []
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            pcs.append(None)
            continue
        fp = gen.GetCountFingerprint(mol)
        arr = np.zeros(2048, dtype=np.int16)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = min(count, 32767)
        fps.append(arr)
        if want_physchem:
            pcs.append(np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
            ], dtype=np.float32))
    return fps, pcs


def featurize_parallel(smiles_list, n_jobs=None, want_physchem=False):
    n_jobs = n_jobs or os.cpu_count() or 4
    batches = np.array_split(smiles_list, n_jobs)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_ecfp4_physchem_batch)(list(b), want_physchem) for b in batches if len(b) > 0
    )
    fps = [fp for batch, _ in results for fp in batch]
    pcs = [pc for _, batch in results for pc in batch] if want_physchem else None
    return fps, pcs


LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"


def load_metadata(orthosteric_only: bool = False, label_source: str = "") -> pd.DataFrame:
    meta = pq.read_table(FEATURES_PARQUET, columns=["RandomID", "SMILES", "Label"]).to_pandas()
    if orthosteric_only:
        ortho = pq.read_table(LABELED_PARQUET, columns=["RandomID", "is_orthosteric"]).to_pandas()
        meta = meta.drop(columns=["Label"]).merge(ortho, on="RandomID", how="left")
        meta["Label"] = meta["is_orthosteric"].fillna(0).astype(np.int8)
        meta = meta.drop(columns=["is_orthosteric"])
    if label_source:
        # Swap in a different parquet's Label column (e.g. the disynthon-corroborated
        # relabeling) without recomputing the cached ECFP4/physchem features -- joined
        # by RandomID (== the raw "compound" DEL synthesis-handle ID in both files).
        override = pq.read_table(label_source, columns=["RandomID", "Label"]).to_pandas()
        override = override.rename(columns={"Label": "LabelOverride"})
        meta = meta.drop(columns=["Label"]).merge(override, on="RandomID", how="left")
        meta["Label"] = meta["LabelOverride"].fillna(0).astype(np.int8)
        meta = meta.drop(columns=["LabelOverride"])
    return meta


def load_sparse_ecfp4() -> sp.csr_matrix:
    """
    Reads the ECFP4 column row-group by row-group and converts each block
    to CSR immediately, so we never hold a dense 7.49M x 2048 array (~31GB)
    in memory — only ever one row-group's worth densely, plus the running
    sparse result (~100x smaller than dense for this data).
    """
    pf = pq.ParquetFile(FEATURES_PARQUET)
    blocks = []
    for i in range(pf.num_row_groups):
        col = pf.read_row_group(i, columns=["ECFP4"]).column("ECFP4")
        dense = np.stack([np.asarray(x) for x in col.combine_chunks().to_pylist()])
        blocks.append(sp.csr_matrix(dense))
        del dense
    return sp.vstack(blocks, format="csr")


def main(asl_gamma_neg: float = 4.0, asl_clip: float = 0.05, team_name: str = "candeleroj", val_ratio: float = 0.15,
         orthosteric_only: bool = False, run_tag: str = "", add_physchem: bool = False,
         max_depth: int = 7, n_estimators: int = 600, reg_lambda: float = 1.0, reg_alpha: float = 0.0,
         submission_index: int = 1, mask_file: str = "", label_source: str = "", neg_per_pos: float = 0.0):
    print("=" * 70)
    print("[*] PGK2 FULL-SCALE XGBoost + ASL TRAINING"
          + (" (ORTHOSTERIC-ONLY LABELS)" if orthosteric_only else "")
          + (f" (LABELS FROM {label_source})" if label_source else "")
          + (f" (NEG SUBSAMPLE {neg_per_pos}x POS -- official-baseline recipe)" if neg_per_pos > 0 else "")
          + (" (+PHYSCHEM)" if add_physchem else ""), flush=True)
    print("=" * 70)

    t0 = time.time()
    meta = load_metadata(orthosteric_only=orthosteric_only, label_source=label_source)
    print(f"[*] Loaded metadata for {len(meta):,} compounds ({int(meta['Label'].sum()):,} positive) "
          f"in {time.time() - t0:.0f}s", flush=True)

    split_cache = "data/processed/pgk2_scaffold_split_idx.npz"
    if os.path.exists(split_cache):
        t0 = time.time()
        cached = np.load(split_cache)
        train_idx, val_idx = cached["train_idx"], cached["val_idx"]
        train_meta = meta.loc[train_idx]
        val_meta = meta.loc[val_idx]
        print(f"[*] Loaded cached scaffold split from {split_cache} in {time.time() - t0:.0f}s | "
              f"Train: {len(train_meta):,} ({int(train_meta['Label'].sum()):,} hits) | "
              f"Val: {len(val_meta):,} ({int(val_meta['Label'].sum()):,} hits)", flush=True)
    else:
        print("[*] Computing scaffold-stratified train/val split (this scans all 7.49M SMILES)...", flush=True)
        t0 = time.time()
        train_meta, val_meta = scaffold_split(meta, smiles_col="SMILES", val_ratio=val_ratio, seed=42)
        train_idx = train_meta.index.to_numpy()
        val_idx = val_meta.index.to_numpy()
        os.makedirs(os.path.dirname(split_cache), exist_ok=True)
        np.savez(split_cache, train_idx=train_idx, val_idx=val_idx)
        print(f"[*] Split done in {(time.time() - t0) / 60:.1f} min (cached to {split_cache}) | "
              f"Train: {len(train_meta):,} ({int(train_meta['Label'].sum()):,} hits) | "
              f"Val: {len(val_meta):,} ({int(val_meta['Label'].sum()):,} hits)", flush=True)

    if neg_per_pos > 0:
        # Replicates the official kickoff-deck baseline recipe ("all DEL-hits
        # and 2x the number of negatives") instead of training on the full
        # ~1800:1 imbalance with ASL. Positives keep every row; negatives are
        # subsampled from TRAIN only (val_meta stays at natural imbalance for
        # an honest, if still OOD-blind, held-out read).
        pos_mask = train_meta["Label"].to_numpy() == 1
        rng = np.random.RandomState(42)
        neg_pool = np.where(~pos_mask)[0]
        n_neg = min(len(neg_pool), int(round(neg_per_pos * pos_mask.sum())))
        neg_sample = rng.choice(neg_pool, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([np.where(pos_mask)[0], neg_sample]))
        train_idx = train_idx[keep]
        train_meta = train_meta.iloc[keep]
        print(f"[*] Negative-subsampled train set: {len(train_meta):,} rows "
              f"({int(pos_mask.sum()):,} pos + {n_neg:,} neg, ratio 1:{neg_per_pos:g})", flush=True)

    ecfp4_cache = "data/processed/pgk2_ecfp4_sparse.npz"
    print("[*] Loading ECFP4 as a sparse matrix (row-group streamed)...", flush=True)
    t0 = time.time()
    if os.path.exists(ecfp4_cache):
        X_all = sp.load_npz(ecfp4_cache)
        print(f"[*] Loaded cached sparse ECFP4 matrix from {ecfp4_cache} in {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        X_all = load_sparse_ecfp4()
        sp.save_npz(ecfp4_cache, X_all)
        print(f"[*] Sparse ECFP4 matrix: shape={X_all.shape}, "
              f"{X_all.nnz:,} nonzeros ({X_all.nnz / (X_all.shape[0] * X_all.shape[1]):.4%} density), "
              f"~{X_all.data.nbytes / 1e9:.2f}GB in memory, loaded+cached in {(time.time() - t0) / 60:.1f} min", flush=True)

    keep_cols = None
    if mask_file:
        keep_cols = np.load(mask_file)["keep"]
        X_all = X_all[:, keep_cols]
        print(f"[*] Domain mask applied: {len(keep_cols)} ECFP4 bits kept", flush=True)

    scaler = None
    if add_physchem:
        physchem_cache = "data/processed/pgk2_physchem.npz"
        print(f"[*] Loading cached physchem descriptors from {physchem_cache}...", flush=True)
        pc_all = np.load(physchem_cache)["X"]
        pc_train_raw, pc_val_raw = pc_all[train_idx], pc_all[val_idx]
        del pc_all
        scaler = StandardScaler().fit(pc_train_raw)
        pc_train = sp.csr_matrix(scaler.transform(pc_train_raw).astype(np.float32))
        pc_val = sp.csr_matrix(scaler.transform(pc_val_raw).astype(np.float32))
        X_train = sp.hstack([X_all[train_idx], pc_train], format="csr")
        X_val = sp.hstack([X_all[val_idx], pc_val], format="csr")
        print(f"[*] Concatenated {pc_train.shape[1]} physchem columns -> X_train shape={X_train.shape}", flush=True)
    else:
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
    y_train = train_meta["Label"].to_numpy()
    y_val = val_meta["Label"].to_numpy()
    del X_all

    print(f"[*] Training XGBoost (ASL gamma_neg={asl_gamma_neg}, clip={asl_clip}, "
          f"max_depth={max_depth}, n_estimators={n_estimators}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha}) "
          f"on {X_train.shape[0]:,} compounds...", flush=True)
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective=make_asymmetric_objective(gamma_pos=0.0, gamma_neg=asl_gamma_neg, clip=asl_clip),
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    print(f"[*] Training done in {(time.time() - t0) / 60:.1f} min", flush=True)

    val_scores = model.predict(X_val, output_margin=True)
    roc = roc_auc_score(y_val, val_scores)
    prauc = average_precision_score(y_val, val_scores)
    print(f"[*] Held-out scaffold-split validation: ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}", flush=True)

    model_out = MODEL_OUT.replace(".json", f"{run_tag}.json") if run_tag else MODEL_OUT
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save_model(model_out)
    print(f"[*] Model saved to {model_out}", flush=True)

    print(f"[*] Scoring OFFICIAL PGK2_Validation_split.csv from {VAL_TEST_ZIP}...", flush=True)
    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        with z.open("Val-Test-set/PGK2_Validation_split.csv") as f:
            official_val = pd.read_csv(f)
    print(f"[*] {len(official_val):,} official validation candidates loaded.", flush=True)

    t0 = time.time()
    fps, pcs = featurize_parallel(official_val["SMILES"].tolist(), want_physchem=add_physchem)
    valid_mask = [fp is not None for fp in fps]
    scored_df = official_val[valid_mask].copy()
    X_official = np.stack([fp for fp, ok in zip(fps, valid_mask) if ok])
    if keep_cols is not None:
        X_official = X_official[:, keep_cols]
    if add_physchem:
        pc_official = np.stack([pc for pc, ok in zip(pcs, valid_mask) if ok])
        pc_official_scaled = scaler.transform(pc_official).astype(np.float32)
        X_official = np.hstack([X_official, pc_official_scaled])
    scored_df["Score"] = model.predict(X_official, output_margin=True)
    scored_df["RandomID"] = scored_df["CatalogID"]
    print(f"[*] Featurized + scored {len(scored_df):,} official candidates in {time.time() - t0:.0f}s", flush=True)
    off_scores = scored_df["Score"].to_numpy()
    off_probs = 1 / (1 + np.exp(-np.clip(off_scores, -30, 30)))
    print(f"[*] Official-pool score stats: min={off_scores.min():.3f} max={off_scores.max():.3f} "
          f"std={off_scores.std():.3f} | prob range=[{off_probs.min():.5f}, {off_probs.max():.5f}] "
          f"unique_scores={len(np.unique(off_scores)):,}", flush=True)

    top50 = top_n_discovered_compounds(scored_df, n=50)
    print("[*] Top-50 predicted PGK2 hits (official validation split):", flush=True)
    print(top50[["CatalogID", "SMILES", "Score"]].to_string(index=False), flush=True)

    os.makedirs("submissions", exist_ok=True)
    path = generate_submission_txt(scored_df, team_name=team_name, submission_index=submission_index)
    print(f"[*] Challenge-format submission (50 hit IDs) written to: {path}", flush=True)

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-scale PGK2 XGBoost+ASL training")
    parser.add_argument("--team-name", default="candeleroj")
    parser.add_argument("--orthosteric-only", action="store_true",
                         help="Use BCM's narrower orthosteric-binder-only labels (relabeled from the cached "
                              "features/split, no fingerprint recompute needed)")
    parser.add_argument("--add-physchem", action="store_true",
                         help="Concatenate 8 standardized physchem descriptors (MolWt/LogP/TPSA/...) to ECFP4 "
                              "-- run scripts/compute_physchem_pgk2.py first")
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05,
                         help="ASL negative-shift clip; pass --gamma-neg 0 --asl-clip 0 for plain BCE "
                              "(matches the official kickoff-deck baseline, which uses no ASL)")
    parser.add_argument("--submission-index", type=int, default=1)
    parser.add_argument("--mask-file", default="", help="npz with 'keep' ECFP4 column indices (make_domain_mask.py)")
    parser.add_argument("--label-source", default="",
                         help="Parquet with RandomID+Label to override the baked-in Label "
                              "(e.g. data/processed/PGK2_selection_labeled_disynthon.parquet)")
    parser.add_argument("--neg-per-pos", type=float, default=0.0,
                         help="Subsample TRAIN negatives to this multiple of the positive count "
                              "(0 = disabled, use full imbalance). The official kickoff-deck baseline "
                              "(5 hits/3 series) used 'all DEL-hits and 2x the number of negatives' -- try 2.0.")
    parser.add_argument("--run-tag", default="", help="Suffix for the saved model filename")
    args = parser.parse_args()
    main(team_name=args.team_name, orthosteric_only=args.orthosteric_only, add_physchem=args.add_physchem,
         max_depth=args.max_depth, n_estimators=args.n_estimators, reg_lambda=args.reg_lambda,
         reg_alpha=args.reg_alpha, asl_gamma_neg=args.gamma_neg, asl_clip=args.asl_clip,
         submission_index=args.submission_index, mask_file=args.mask_file, label_source=args.label_source,
         neg_per_pos=args.neg_per_pos, run_tag=args.run_tag or ("_orthosteric" if args.orthosteric_only else ""))
