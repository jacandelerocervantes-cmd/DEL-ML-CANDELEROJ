"""
Real-data validation of both pipelines (XGBoost+ASL and DELBERT+LoRA+ASL)
against the DREAM Target 2035 Challenge #1 (WDR91) data, while the real
PGK2 datasets (OpenDEL_libraries, PGK2, PGK2_CACHE_Val_Test_Set,
PGK2_NTC_Supplement) remain "Pending approval" on AIRCHECK.

Inputs
------
data/raw/DREAM_Challenge_1_TrainSet.parquet
    339,258 WDR91 DEL compounds, LABEL (138 positives / 339,120 negatives —
    this exact 138-hit count matches the challenge write-up), plus 9
    official precomputed fingerprint families (ECFP4, ECFP6, FCFP4, FCFP6,
    MACCS, RDK, AVALON, ATOMPAIR, TOPTOR) and MW/ALOGP.
data/raw/DREAM_Target2035_Challenge_test_data.csv
    234 unlabeled candidate compounds (Ccompound_ID, SMILES) to rank.

This is REAL labeled data (unlike the CACHE7-vs-ChEMBL placeholder used by
scripts/train_baseline.py and scripts/finetune_delbert.py), so the metrics
here are an honest read on both architectures — the actual dry run we
wanted before PGK2 access lands.

Memory note: the free-RAM check on this machine showed ~1.3GB free at time
of writing, and the full ECFP4 column alone is ~2.8GB dense. To stay safe,
both tracks train on all 138 positives + a random subsample of negatives
(`--n-neg`, default 20,000) rather than the full 339,120. Pass a larger
--n-neg (or 0 for "use everything") on a machine with more headroom.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

from modules.delbert.lora import apply_lora, count_trainable_parameters
from modules.delbert.model import DELBERT, DELBERTClassifier, mask_tokens
from modules.delbert.tokenizer import FingerprintTokenizer, compute_fp_dict, fp_dict_from_columns
from modules.losses.asymmetric_loss import asymmetric_loss_torch, make_asymmetric_objective
from modules.submission.validator import top_n_discovered_compounds
from modules.validation.scaffold_split import scaffold_split

TRAIN_PARQUET = "data/raw/DREAM_Challenge_1_TrainSet.parquet"
TEST_CSV = "data/raw/DREAM_Target2035_Challenge_test_data.csv"
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_subsampled_train(n_neg: int, seed: int = 42) -> pd.DataFrame:
    cols = ["SMILES", "RandomID", "LABEL", "ECFP4", "MACCS"]
    df = pq.read_table(TRAIN_PARQUET, columns=cols).to_pandas()
    pos = df[df["LABEL"] == 1]
    neg = df[df["LABEL"] == 0]
    if n_neg and n_neg < len(neg):
        neg = neg.sample(n=n_neg, random_state=seed)
    print(f"[*] Using {len(pos)} positives + {len(neg)} negatives (of {len(df)} total rows).")
    out = pd.concat([pos, neg], ignore_index=True).rename(columns={"LABEL": "Label"})
    out["ECFP4"] = out["ECFP4"].apply(lambda a: np.asarray(a, dtype=np.int8))
    out["MACCS"] = out["MACCS"].apply(lambda a: np.asarray(a, dtype=np.int8))
    return out


def hits_in_top_n(scored_df: pd.DataFrame, n: int = 50, label_col: str = "Label", score_col: str = "Score", smiles_col: str = "SMILES"):
    """
    Mirrors the official challenge's own reporting metric ("X hits,
    corresponding to Y chemical series, in the top-50 ranked molecules"):
    counts true positives in the top-`n` ranked candidates, and how many
    distinct Bemis-Murcko scaffolds ("series") those hits span.
    """
    from modules.validation.scaffold_split import get_murcko_scaffold

    top = scored_df.sort_values(score_col, ascending=False).head(n)
    hit_rows = top[top[label_col] == 1]
    n_hits = len(hit_rows)
    n_series = hit_rows[smiles_col].apply(get_murcko_scaffold).nunique() if n_hits else 0
    return n_hits, n_series


def smiles_to_ecfp4_counts(smiles: str) -> np.ndarray:
    """Matches the official HitGenECFP4 hashed-count definition."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem import AllChem

    fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048, useFeatures=False)
    arr = np.zeros(2048, dtype=np.int8)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx] = min(count, 127)
    return arr


# --------------------------------------------------------------------------
# Track A: XGBoost + Asymmetric Loss on precomputed ECFP4
# --------------------------------------------------------------------------
def run_xgboost_track(df: pd.DataFrame, test_df: pd.DataFrame, asl_gamma_neg: float, asl_clip: float):
    print("\n" + "=" * 70)
    print("[TRACK A] XGBoost + Asymmetric Loss (precomputed official ECFP4)")
    print("=" * 70)

    train_df, val_df = scaffold_split(df, smiles_col="SMILES", val_ratio=0.2, seed=42)
    X_train = np.stack(train_df["ECFP4"].to_numpy())
    y_train = train_df["Label"].to_numpy()
    X_val = np.stack(val_df["ECFP4"].to_numpy())
    y_val = val_df["Label"].to_numpy()
    print(f"[*] Train: {len(train_df)} ({y_train.sum()} pos) | Val: {len(val_df)} ({y_val.sum()} pos)")

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

    val_scores = model.predict(X_val, output_margin=True)
    roc = roc_auc_score(y_val, val_scores)
    prauc = average_precision_score(y_val, val_scores)
    print(f"[*] Validation (real WDR91 scaffold-held-out): ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}")

    hits_at_50, n_series = hits_in_top_n(val_df.assign(Score=val_scores), n=50)
    print(
        f"[!] NOTE: {hits_at_50}/{n_series} above is hits-in-top-50 over our small {len(val_df)}-candidate "
        f"held-out slice, NOT the official ~339K-candidate pool — trivially easier, not comparable to the "
        f"baseline below. See the full-pool number right after this."
    )

    print("[*] Re-scoring the FULL 339,258-compound pool (excluding train rows) for a fair "
          "apples-to-apples comparison against the official baseline metric...")
    full_df = pq.read_table(TRAIN_PARQUET, columns=["SMILES", "RandomID", "LABEL", "ECFP4"]).to_pandas()
    full_df = full_df[~full_df["RandomID"].isin(train_df["RandomID"])].reset_index(drop=True)
    X_full = np.stack(full_df["ECFP4"].apply(lambda a: np.asarray(a, dtype=np.int8)).to_numpy())
    full_df = full_df.assign(Score=model.predict(X_full, output_margin=True), Label=full_df["LABEL"])
    full_hits_at_50, full_n_series = hits_in_top_n(full_df, n=50)
    print(
        f"[*] OFFICIAL-METRIC COMPARISON (full {len(full_df)}-compound pool, {int(full_df['Label'].sum())} true "
        f"hits total): {full_hits_at_50} true hits ({full_n_series} distinct Bemis-Murcko scaffolds/'series') "
        f"in our top-50 ranked candidates — vs. the official baseline (LightGBM+ECFP4+2x negatives): "
        f"5 hits / 3 series in top-50 validation, 1 hit in top-50 test."
    )
    del full_df, X_full

    print(f"[*] Scoring {len(test_df)} unlabeled candidates from {TEST_CSV}...")
    fps = [smiles_to_ecfp4_counts(s) for s in test_df["SMILES"]]
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    X_test = np.stack([fps[i] for i in valid])
    scored = test_df.iloc[valid].copy()
    scored["Score"] = model.predict(X_test, output_margin=True)

    ranked = top_n_discovered_compounds(scored, n=len(scored), score_col="Score")
    os.makedirs("submissions", exist_ok=True)
    out_path = "submissions/wdr91_xgboost_discovered_candidates.csv"
    ranked.to_csv(out_path, index=False)
    print(f"[*] Ranked candidate list written to: {out_path}")
    print(ranked[["Ccompound_ID", "SMILES", "Score"]].head(10).to_string(index=False))
    return {"roc_auc": roc, "pr_auc": prauc, "ranked": ranked}


# --------------------------------------------------------------------------
# Track B: DELBERT (MLM pretrain, from precomputed fingerprints) + LoRA + ASL
# --------------------------------------------------------------------------
class _FPDataset(Dataset):
    def __init__(self, fp_dicts, tokenizer, max_len, labels=None):
        self.fp_dicts = fp_dicts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.fp_dicts)

    def __getitem__(self, idx):
        ids, mask = self.tokenizer.encode(self.fp_dicts[idx], max_len=self.max_len)
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.long)
        if self.labels is None:
            return ids_t, mask_t
        return ids_t, mask_t, torch.tensor(float(self.labels[idx]), dtype=torch.float32)


def run_delbert_track(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    asl_gamma_neg: float,
    asl_clip: float,
    device: str,
):
    print("\n" + "=" * 70)
    print("[TRACK B] DELBERT (MLM pretrain -> LoRA fine-tune, Asymmetric Loss)")
    print("=" * 70)

    fp_dicts = [fp_dict_from_columns(row, fp_cols=["ECFP4", "MACCS"]) for _, row in df.iterrows()]
    tokenizer = FingerprintTokenizer().fit(fp_dicts)
    print(f"[*] Vocabulary size (from official precomputed ECFP4+MACCS): {tokenizer.vocab_size()} tokens.")

    # --- Pretrain (MLM) ---------------------------------------------------
    pre_ds = _FPDataset(fp_dicts, tokenizer, max_len)
    pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True)
    backbone = DELBERT(vocab_size=tokenizer.vocab_size(), d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(backbone.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    mask_id = tokenizer.token_to_id["[MASK]"]
    special_ids = {tokenizer.token_to_id[t] for t in ["[PAD]", "[MASK]", "[CLS]", "[UNK]"]}

    print(f"[*] MLM pre-training on {len(pre_ds)} real WDR91 DEL compounds, {pretrain_epochs} epochs...")
    backbone.train()
    for epoch in range(pretrain_epochs):
        total, n = 0.0, 0
        for ids, mask in pre_loader:
            ids, mask = ids.to(device), mask.to(device)
            masked_ids, labels = mask_tokens(ids, mask, mask_id, tokenizer.vocab_size(), special_token_ids=special_ids)
            logits = backbone.forward_mlm(masked_ids, mask)
            loss = loss_fn(logits.view(-1, tokenizer.vocab_size()), labels.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"    epoch {epoch + 1}/{pretrain_epochs} | MLM loss = {total / max(n, 1):.4f}")

    os.makedirs("models/pretrained/delbert_wdr91", exist_ok=True)
    torch.save(
        {"model_state": backbone.state_dict(),
         "config": {"vocab_size": tokenizer.vocab_size(), "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "d_ff": 256, "max_len": max_len}},
        "models/pretrained/delbert_wdr91/delbert_pretrained.pt",
    )
    tokenizer.save("models/pretrained/delbert_wdr91/tokenizer.json")

    # --- Fine-tune (LoRA + ASL) --------------------------------------------
    backbone = apply_lora(backbone, r=8, alpha=16)
    model = DELBERTClassifier(backbone).to(device)
    for p in model.head.parameters():
        p.requires_grad = True
    print(f"[*] Trainable params after LoRA: {count_trainable_parameters(model):,}")

    train_df, val_df = scaffold_split(df, smiles_col="SMILES", val_ratio=0.2, seed=42)
    train_fps = [fp_dict_from_columns(row, fp_cols=["ECFP4", "MACCS"]) for _, row in train_df.iterrows()]
    val_fps = [fp_dict_from_columns(row, fp_cols=["ECFP4", "MACCS"]) for _, row in val_df.iterrows()]
    train_ds = _FPDataset(train_fps, tokenizer, max_len, labels=train_df["Label"].to_numpy())
    val_ds = _FPDataset(val_fps, tokenizer, max_len, labels=val_df["Label"].to_numpy())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    print(f"[*] Fine-tuning: train={len(train_ds)} val={len(val_ds)}, {finetune_epochs} epochs...")
    model.train()
    for epoch in range(finetune_epochs):
        total, n = 0.0, 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = asymmetric_loss_torch(logits, labels, gamma_pos=0.0, gamma_neg=asl_gamma_neg, clip=asl_clip)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"    epoch {epoch + 1}/{finetune_epochs} | ASL loss = {total / max(n, 1):.4f}")

    model.eval()
    scores, labels_all = [], []
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask = ids.to(device), mask.to(device)
            scores.append(model(ids, mask).cpu().numpy())
            labels_all.append(labels.numpy())
    scores, labels_all = np.concatenate(scores), np.concatenate(labels_all)
    roc = roc_auc_score(labels_all, scores)
    prauc = average_precision_score(labels_all, scores)
    print(f"[*] Validation (real WDR91 scaffold-held-out): ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}")

    hits_at_50, n_series = hits_in_top_n(val_df.assign(Score=scores), n=50)
    print(
        f"[*] Official-metric comparison: {hits_at_50} true hits ({n_series} distinct Bemis-Murcko "
        f"scaffolds/'series') in the top-50 ranked candidates of our held-out split — vs. the official "
        f"baseline (LightGBM+ECFP4+2x negatives): 5 hits / 3 series in top-50 validation, 1 hit in top-50 test."
    )

    print(f"[*] Scoring {len(test_df)} unlabeled candidates from {TEST_CSV}...")
    test_fp_dicts = [compute_fp_dict(s) for s in test_df["SMILES"]]
    valid = [i for i, fp in enumerate(test_fp_dicts) if fp]
    test_ds = _FPDataset([test_fp_dicts[i] for i in valid], tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=16)
    test_scores = []
    with torch.no_grad():
        for ids, mask in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            test_scores.append(model(ids, mask).cpu().numpy())
    test_scores = np.concatenate(test_scores)

    scored = test_df.iloc[valid].copy()
    scored["Score"] = test_scores
    ranked = top_n_discovered_compounds(scored, n=len(scored), score_col="Score")
    out_path = "submissions/wdr91_delbert_discovered_candidates.csv"
    ranked.to_csv(out_path, index=False)
    print(f"[*] Ranked candidate list written to: {out_path}")
    print(ranked[["Ccompound_ID", "SMILES", "Score"]].head(10).to_string(index=False))
    return {"roc_auc": roc, "pr_auc": prauc, "ranked": ranked}


def main():
    parser = argparse.ArgumentParser(description="Validate both pipelines on real WDR91 Challenge #1 data")
    parser.add_argument("--n-neg", type=int, default=20000, help="Negative subsample size (0 = use all 339,120)")
    parser.add_argument("--skip-delbert", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    df = load_subsampled_train(args.n_neg)
    test_df = pd.read_csv(TEST_CSV)

    results = {}
    results["xgboost"] = run_xgboost_track(df, test_df, asl_gamma_neg=4.0, asl_clip=0.05)

    if not args.skip_delbert:
        results["delbert"] = run_delbert_track(
            df, test_df,
            max_len=256, d_model=128, n_layers=4, n_heads=4,
            pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
            asl_gamma_neg=4.0, asl_clip=0.05, device=args.device,
        )

    print("\n" + "=" * 70)
    print("[SUMMARY] Real WDR91 validation results")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:10s} | ROC-AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
