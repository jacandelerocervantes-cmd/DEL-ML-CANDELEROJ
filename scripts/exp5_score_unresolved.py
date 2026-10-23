"""
Angle 5 (active-learning prep, practical version): score the 289 CatalogIDs with
confirmed_label==NaN in confirmed_submission_outcomes.csv -- these are compounds known to
be in a submission round that returned >=1 hit, but Synapse never revealed which one. This
is the exact target of the incoming Active Learning Step (labels released Oct 1-31):
prioritizing which of these 289 to resolve first, and which of the two validated models
(ECFP4+LightGBM vs ChemBERTa+LogReg) agree, is a genuinely actionable output -- not a
retrospective metric, an actual ranked worklist.

Trains both final models on ALL 2,335 labeled compounds (no holdout -- this is for
deployment, not evaluation; evaluation already happened in exp5_baseline_ecfp4.py and
exp5_chemberta.py under cluster-disjoint CV) and scores the 289 unresolved compounds.

Run: python -m scripts.exp5_score_unresolved
"""
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from scripts.exp5_common import load_labeled_set, SEED
from scripts.exp5_baseline_ecfp4 import ecfp4_matrix
from scripts.exp5_chemberta import embed_smiles


def main():
    labeled = load_labeled_set()
    out = pd.read_csv("data/processed/confirmed_submission_outcomes.csv")
    unresolved = out[out["confirmed_label"].isna()].copy()
    val = pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv")
    unresolved = unresolved.merge(val, on="CatalogID", how="left").dropna(subset=["SMILES"])
    print(f"[score_unresolved] {len(unresolved)} of 289 unresolved compounds have SMILES "
          f"in the validation pool")

    y = labeled["y"].to_numpy()

    # ECFP4 + LightGBM, fit on everything
    X_train_ecfp = ecfp4_matrix(labeled["SMILES"].tolist())
    X_unres_ecfp = ecfp4_matrix(unresolved["SMILES"].tolist())
    lgbm = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                           random_state=SEED, verbosity=-1)
    lgbm.fit(X_train_ecfp, y)
    unresolved["score_ecfp4_lgbm"] = lgbm.predict_proba(X_unres_ecfp)[:, 1]

    # ChemBERTa + LogReg, fit on everything (reuses cached labeled embeddings)
    X_train_emb = embed_smiles(labeled["SMILES"].tolist())
    X_unres_emb = embed_smiles(unresolved["SMILES"].tolist())
    logreg = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED)
    logreg.fit(X_train_emb, y)
    unresolved["score_chemberta_logreg"] = logreg.predict_proba(X_unres_emb)[:, 1]

    unresolved["rank_ecfp4"] = unresolved["score_ecfp4_lgbm"].rank(ascending=False)
    unresolved["rank_chemberta"] = unresolved["score_chemberta_logreg"].rank(ascending=False)
    unresolved["consensus_rank"] = unresolved[["rank_ecfp4", "rank_chemberta"]].mean(axis=1)
    unresolved["models_agree_top30"] = (
        (unresolved["rank_ecfp4"] <= 30) & (unresolved["rank_chemberta"] <= 30)
    )

    unresolved_sorted = unresolved.sort_values("consensus_rank")
    cols = ["CatalogID", "SMILES", "n_rounds", "round_outcome", "score_ecfp4_lgbm",
            "score_chemberta_logreg", "rank_ecfp4", "rank_chemberta", "models_agree_top30"]
    out_path = "data/processed/gp/exp5_unresolved_ranked.csv"
    unresolved_sorted[cols].to_csv(out_path, index=False)

    n_agree = int(unresolved["models_agree_top30"].sum())
    print(f"[score_unresolved] wrote {out_path}")
    print(f"[score_unresolved] {n_agree} compounds ranked in BOTH models' top-30 "
          f"(highest-priority to resolve first)")
    print(unresolved_sorted[unresolved_sorted["models_agree_top30"]][
        ["CatalogID", "score_ecfp4_lgbm", "score_chemberta_logreg", "n_rounds"]
    ].to_string(index=False))

    corr = unresolved["score_ecfp4_lgbm"].corr(unresolved["score_chemberta_logreg"], method="spearman")
    with open("data/processed/gp/exp5_unresolved_summary.json", "w") as f:
        json.dump({
            "n_unresolved_scored": len(unresolved),
            "n_agree_top30": n_agree,
            "spearman_corr_between_models": float(corr),
        }, f, indent=1)


if __name__ == "__main__":
    main()
