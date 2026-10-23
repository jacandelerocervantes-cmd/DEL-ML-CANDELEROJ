"""
Shared data loading + evaluation harness for the "five angles vs real ASMS" experiment
(2026-09-27 session, branch experiment/five-angles-vs-asms).

Ground truth: data/processed/confirmed_submission_outcomes.csv, crossed against Synapse
results (14 confirmed hits, 2,321 confirmed negatives, 289 unresolved "presumed hits" with
confirmed_label==NaN, excluded from train/eval). SMILES looked up from the official
PGK2_Validation_split.csv (244,328 rows) since the outcomes table only has CatalogID.

Every method in this experiment is evaluated on the SAME labeled set with the SAME
protocol (repeated stratified k-fold, since n_pos=14 is too small for a single holdout
to be stable) so results are comparable head-to-head, following the project's own
"prueba de fuego" convention (LOG-EXPERIMENTAL-2026-09-24.md): validate against confirmed
ASMS labels before trusting any method.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
N_SPLITS, N_REPEATS = 5, 20  # 100 folds total to stabilize AUC estimate with only 14 positives

ROOT = "."


def butina_clusters(smiles_list, cutoff=0.6, n_bits=2048, radius=2):
    """Groups near-neighbors (Tanimoto distance <= cutoff) into the same cluster id.
    Used as the CV group so structurally-similar hits (same series) can't leak across
    train/test the way a random stratified fold does -- see the 0.998 AUC sanity check
    in exp5_baseline_ecfp4.py that this function exists to fix."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    group = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            group[m] = cid
    return group


def cv_evaluate_grouped(scores_fn, groups, y, n_splits=5, seed=SEED):
    """Group-aware CV: no cluster is ever split across train and test. With only 14
    positives spread over ~2-3 dominant series, this is the honest version of
    cv_evaluate -- random StratifiedKFold lets near-duplicate hits leak across folds."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, praucs = [], []
    oof_scores = np.full(len(y), np.nan)
    n_pos_in_test_folds = []
    for train_idx, test_idx in sgkf.split(np.zeros(len(y)), y, groups):
        test_scores = scores_fn(train_idx, test_idx)
        oof_scores[test_idx] = test_scores
        y_test = y[test_idx]
        n_pos_in_test_folds.append(int(y_test.sum()))
        if len(set(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, test_scores))
        praucs.append(average_precision_score(y_test, test_scores))
    valid = ~np.isnan(oof_scores)
    pooled_auc = roc_auc_score(y[valid], oof_scores[valid]) if len(set(y[valid])) > 1 else float("nan")
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "prauc_mean": float(np.mean(praucs)) if praucs else float("nan"),
        "prauc_std": float(np.std(praucs)) if praucs else float("nan"),
        "pooled_oof_auc": float(pooled_auc),
        "n_folds_with_both_classes": len(aucs),
        "n_pos_per_fold": n_pos_in_test_folds,
        "oof_scores": oof_scores.reshape(1, -1),
    }


def cached_groups(df, cutoff=0.6, cache_path=f"{ROOT}/data/processed/gp/exp5_groups.npy"):
    """Same Butina cluster ids for every method in this experiment, so head-to-head
    comparisons use identical train/test splits, not just the same cutoff parameter."""
    import os
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True).item()
        if cached["ids"] == df["CatalogID"].tolist():
            return cached["groups"]
    groups = butina_clusters(df["SMILES"].tolist(), cutoff=cutoff)
    np.save(cache_path, {"ids": df["CatalogID"].tolist(), "groups": groups})
    return groups


def load_labeled_set():
    out = pd.read_csv(f"{ROOT}/data/processed/confirmed_submission_outcomes.csv")
    labeled = out.dropna(subset=["confirmed_label"]).copy()
    labeled["confirmed_label"] = labeled["confirmed_label"].astype(int)

    val = pd.read_csv(f"{ROOT}/data/raw/Val-Test-set/PGK2_Validation_split.csv")
    merged = labeled.merge(val, on="CatalogID", how="left")

    missing = merged["SMILES"].isna().sum()
    if missing:
        # A few confirmed compounds may come from redteam CSVs (test-split adjacent);
        # fall back to those before dropping.
        extra = pd.concat([
            pd.read_csv(f"{ROOT}/docs/redteam_hits_confirmados.csv")[["CatalogID", "SMILES"]],
            pd.read_csv(f"{ROOT}/docs/redteam_negativos_confirmados.csv")[["CatalogID", "SMILES"]],
        ]).drop_duplicates("CatalogID")
        merged = merged.set_index("CatalogID")
        merged["SMILES"] = merged["SMILES"].fillna(extra.set_index("CatalogID")["SMILES"])
        merged = merged.reset_index()

    still_missing = merged["SMILES"].isna().sum()
    merged = merged.dropna(subset=["SMILES"]).reset_index(drop=True)
    n_pos = int((merged["confirmed_label"] == 1).sum())
    n_neg = int((merged["confirmed_label"] == 0).sum())
    print(f"[exp5_common] labeled set: {len(merged)} rows ({n_pos} hits, {n_neg} negatives), "
          f"{still_missing} dropped for missing SMILES")
    return merged[["CatalogID", "SMILES", "confirmed_label"]].rename(columns={"confirmed_label": "y"})


def cv_evaluate(scores_fn, X_ids, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """
    scores_fn(train_idx, test_idx) -> test_scores (fit on train_idx, score test_idx).
    Returns per-fold AUC/PR-AUC plus pooled out-of-fold predictions for a single global AUC.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, praucs = [], []
    oof_scores = np.full((n_repeats, len(y)), np.nan)
    rep = -1
    for i, (train_idx, test_idx) in enumerate(rskf.split(X_ids, y)):
        if i % n_splits == 0:
            rep += 1
        test_scores = scores_fn(train_idx, test_idx)
        y_test = y[test_idx]
        if len(set(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, test_scores))
        praucs.append(average_precision_score(y_test, test_scores))
        oof_scores[rep, test_idx] = test_scores
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "prauc_mean": float(np.mean(praucs)), "prauc_std": float(np.std(praucs)),
        "n_folds": len(aucs),
        "oof_scores": oof_scores,
    }


def hit_rank_summary(oof_scores, y, top_k=(50, 100, 250)):
    """Average rank (1=best) of true hits across repeats, and how many repeats put
    each hit in the top-k, mirroring the challenge's actual top-50 selection."""
    n_reps = oof_scores.shape[0]
    pos_idx = np.where(y == 1)[0]
    ranks_per_rep = []
    topk_hits = {k: [] for k in top_k}
    for r in range(n_reps):
        s = oof_scores[r]
        valid = ~np.isnan(s)
        order = np.argsort(-np.where(valid, s, -np.inf))
        rank_of = {idx: rank for rank, idx in enumerate(order, start=1)}
        n_valid = valid.sum()
        ranks_per_rep.append({int(p): rank_of.get(p) for p in pos_idx if valid[p]})
        for k in top_k:
            topk_hits[k].append(sum(1 for p in pos_idx if valid[p] and rank_of[p] <= k))
    return ranks_per_rep, {k: float(np.mean(v)) for k, v in topk_hits.items()}
