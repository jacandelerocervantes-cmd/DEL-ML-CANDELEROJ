"""
Angle 3 (the LLM angle): embed molecules with ChemBERTa (seyonec/ChemBERTa-zinc-base-v1),
a RoBERTa-style transformer pretrained by masked-language-modeling on ~77M ZINC SMILES --
no DEL, ASMS, or PGK2 data anywhere in its pretraining. This is the fresh-perspective test:
does chemical knowledge learned from a huge *unrelated* corpus transfer across the DEL<->ASMS
domain gap better than ECFP4 (which has zero prior outside this exact dataset)? A downstream
logistic regression is trained on top of the frozen embeddings, in-domain, same cluster-
disjoint CV as the other angles, same labeled set.

The project's own reference point: organizers cite MAMMAL/MMELON (IBM, also a pretrained
chemical LLM) getting ~1 hit in the top-50 on this exact split -- weak but nonzero signal
from a model with no access to labels. That's the bar this angle needs to clear to be
worth pursuing further, not the leaky 0.998 AUC from exp5_baseline_ecfp4.py.

Run: python -m scripts.exp5_chemberta
"""
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

from scripts.exp5_common import load_labeled_set, cv_evaluate_grouped, cached_groups, hit_rank_summary, SEED

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
EMB_CACHE = "data/processed/gp/exp5_chemberta_embeddings.npy"


def embed_smiles(smiles_list, batch_size=32):
    import os
    if os.path.exists(EMB_CACHE):
        cached = np.load(EMB_CACHE, allow_pickle=True).item()
        if cached["smiles"] == smiles_list:
            print("[chemberta] using cached embeddings")
            return cached["emb"]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            embs.append(pooled.numpy())
            if i % (batch_size * 10) == 0:
                print(f"[chemberta] embedded {i}/{len(smiles_list)}")
    emb = np.concatenate(embs, axis=0)
    np.save(EMB_CACHE, {"smiles": smiles_list, "emb": emb})
    return emb


def main():
    df = load_labeled_set()
    y = df["y"].to_numpy()
    groups = cached_groups(df)

    X = embed_smiles(df["SMILES"].tolist())
    print(f"[chemberta] embedding matrix: {X.shape}")

    def scores_fn(train_idx, test_idx):
        model = LogisticRegression(max_iter=3000, class_weight="balanced",
                                    C=1.0, random_state=SEED)
        model.fit(X[train_idx], y[train_idx])
        return model.predict_proba(X[test_idx])[:, 1]

    result = cv_evaluate_grouped(scores_fn, groups, y)
    _, topk = hit_rank_summary(result["oof_scores"], y)

    summary = {
        "method": "chemberta_zinc_base_v1_frozen_embeddings_logreg",
        "n": len(df), "n_pos": int(y.sum()), "embedding_dim": X.shape[1],
        "cluster_disjoint_cv": {
            "auc_mean": result["auc_mean"], "auc_std": result["auc_std"],
            "pooled_oof_auc": result["pooled_oof_auc"],
            "prauc_mean": result["prauc_mean"],
            "n_folds_with_both_classes": result["n_folds_with_both_classes"],
            "n_pos_per_fold": result["n_pos_per_fold"],
            "mean_hits_in_topk": topk,
        },
    }
    print(json.dumps(summary, indent=1))
    with open("data/processed/gp/exp5_chemberta.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
