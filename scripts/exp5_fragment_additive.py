"""
Angle 2: additive/compositional fragment model, trained directly in ASMS space (not
transferred from DEL). Instead of one holistic ECFP4 fingerprint per molecule or a single
MCS pharmacophore (which by construction requires ALL hits to share one motif -- see
LOG-EXPERIMENTAL Paso 7), decompose each molecule into its BRICS fragments and fit an
L1-regularized logistic regression on fragment *presence*. This tests whether hit activity
is explained by a SUM of a few independent fragment contributions (possibly from different
series) rather than one dominant scaffold -- the true "building-block additive" hypothesis
from the challenge's own disynthon framing, applied to whole ASMS molecules since DEL
building-block IDs don't exist for these Enamine compounds (0/143 overlap, LOG-EXPERIMENTAL
Paso 5).

Run: python -m scripts.exp5_fragment_additive
"""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression

from scripts.exp5_common import load_labeled_set, cv_evaluate_grouped, cached_groups, hit_rank_summary, SEED


def brics_fragment_matrix(smiles_list, min_count=3):
    frag_sets = []
    vocab = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        frags = set()
        if mol is not None:
            try:
                for f in BRICS.BRICSDecompose(mol, minFragmentSize=2):
                    frags.add(f)
            except Exception:
                pass
        frag_sets.append(frags)
        for f in frags:
            vocab[f] = vocab.get(f, 0) + 1
    kept = [f for f, c in vocab.items() if c >= min_count]
    kept_idx = {f: i for i, f in enumerate(kept)}
    X = lil_matrix((len(smiles_list), len(kept)), dtype=np.float32)
    for i, frags in enumerate(frag_sets):
        for f in frags:
            if f in kept_idx:
                X[i, kept_idx[f]] = 1
    return X.tocsr(), kept


def main():
    df = load_labeled_set()
    y = df["y"].to_numpy()
    groups = cached_groups(df)

    X, vocab = brics_fragment_matrix(df["SMILES"].tolist())
    print(f"[fragment_additive] {X.shape[1]} BRICS fragments seen >=3 times, "
          f"{X.shape[0]} compounds")

    def scores_fn(train_idx, test_idx):
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, max_iter=2000,
            class_weight="balanced", random_state=SEED,
        )
        model.fit(X[train_idx], y[train_idx])
        return model.predict_proba(X[test_idx])[:, 1]

    result = cv_evaluate_grouped(scores_fn, groups, y)
    _, topk = hit_rank_summary(result["oof_scores"], y)

    # Fit once on everything to inspect which fragments the model actually leans on
    # (interpretability check: is it really "additive many-motif" or did it just
    # re-derive the single known pyridin-3-yl-pyrimidine motif in fragment form?)
    full_model = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000,
                                     class_weight="balanced", random_state=SEED)
    full_model.fit(X, y)
    coef = full_model.coef_[0]
    top_frags = sorted(zip(vocab, coef), key=lambda t: -t[1])[:10]

    summary = {
        "method": "brics_fragment_l1_logreg_indomain",
        "n": len(df), "n_pos": int(y.sum()), "n_fragments": X.shape[1],
        "cluster_disjoint_cv": {
            "auc_mean": result["auc_mean"], "auc_std": result["auc_std"],
            "pooled_oof_auc": result["pooled_oof_auc"],
            "prauc_mean": result["prauc_mean"],
            "n_folds_with_both_classes": result["n_folds_with_both_classes"],
            "n_pos_per_fold": result["n_pos_per_fold"],
            "mean_hits_in_topk": topk,
        },
        "n_nonzero_fragment_weights": int((coef != 0).sum()),
        "top_positive_fragments": [(f, round(c, 3)) for f, c in top_frags if c > 0],
    }
    print(json.dumps(summary, indent=1))
    with open("data/processed/gp/exp5_fragment_additive.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
