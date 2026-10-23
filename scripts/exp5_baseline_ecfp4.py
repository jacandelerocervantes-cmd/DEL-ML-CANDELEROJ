"""
Baseline for the five-angle comparison: ECFP4 (2048-bit count Morgan, r=2) + LightGBM,
same fingerprint recipe as submissions/writeup_candeleroj.md, trained/evaluated directly
on the 2,335 ASMS-confirmed compounds (in-domain, not transferred from DEL) under the
repeated-CV protocol in exp5_common.py. This is the number every other angle must beat.

Run: python -m scripts.exp5_baseline_ecfp4
"""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMClassifier

from scripts.exp5_common import load_labeled_set, cv_evaluate, cv_evaluate_grouped, cached_groups, hit_rank_summary, SEED


def ecfp4_matrix(smiles_list, n_bits=2048, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, countSimulation=True)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetCountFingerprint(mol)
        for idx, cnt in fp.GetNonzeroElements().items():
            X[i, idx % n_bits] = cnt
    return X


def main():
    df = load_labeled_set()
    X = ecfp4_matrix(df["SMILES"].tolist())
    y = df["y"].to_numpy()

    def scores_fn(train_idx, test_idx):
        model = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y[train_idx] == 0).sum() / max(1, (y[train_idx] == 1).sum()),
            random_state=SEED, verbosity=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        return model.predict_proba(X[test_idx])[:, 1]

    result_random = cv_evaluate(scores_fn, df["CatalogID"].to_numpy(), y)
    ranks, topk_random = hit_rank_summary(result_random["oof_scores"], y)

    groups = cached_groups(df)
    print(f"[baseline] {len(set(groups))} clusters over {len(df)} compounds "
          f"(cutoff=0.6); hits fall in {len(set(groups[y == 1]))} distinct clusters")
    result_grouped = cv_evaluate_grouped(scores_fn, groups, y)
    ranks_g, topk_grouped = hit_rank_summary(result_grouped["oof_scores"], y)

    summary = {
        "method": "ecfp4_lightgbm_indomain",
        "n": len(df), "n_pos": int(y.sum()),
        "random_cv_LEAKY": {
            "auc_mean": result_random["auc_mean"], "auc_std": result_random["auc_std"],
            "prauc_mean": result_random["prauc_mean"],
            "mean_hits_in_topk": topk_random,
        },
        "cluster_disjoint_cv_HONEST": {
            "auc_mean": result_grouped["auc_mean"], "auc_std": result_grouped["auc_std"],
            "pooled_oof_auc": result_grouped["pooled_oof_auc"],
            "prauc_mean": result_grouped["prauc_mean"],
            "n_folds_with_both_classes": result_grouped["n_folds_with_both_classes"],
            "n_pos_per_fold": result_grouped["n_pos_per_fold"],
            "mean_hits_in_topk": topk_grouped,
        },
    }
    print(json.dumps(summary, indent=1))
    with open("data/processed/gp/exp5_baseline_ecfp4.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
