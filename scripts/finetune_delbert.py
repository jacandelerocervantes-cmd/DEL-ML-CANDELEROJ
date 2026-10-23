"""
LoRA fine-tuning of a pretrained DELBERT for PGK2 hit classification, using
Asymmetric Loss (see modules/losses/asymmetric_loss.py) to survive the
extreme hit/non-hit imbalance, then ranking + official-format submission
generation (modules/submission/validator.py): a plain list of exactly 50
predicted hit compound IDs, per
docs/operativa/TARGET2035-CHALLENGE2-PGK2-REGLAS-OFICIALES.md.

Requires a pretrained checkpoint from scripts/pretrain_delbert.py.

Same placeholder-data caveat as scripts/train_baseline.py applies here:
without real AIRCHECK negatives, this is a pipeline smoke test, not a
trustworthy hit-caller. Swap in --train-parquet / --test-parquet once the
real PGK2 data lands.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from modules.delbert.lora import apply_lora, count_trainable_parameters
from modules.delbert.model import DELBERT, DELBERTClassifier
from modules.delbert.tokenizer import FingerprintTokenizer, compute_fp_dict
from modules.losses.asymmetric_loss import asymmetric_loss_torch
from modules.submission.validator import generate_submission_txt, top_n_discovered_compounds
from modules.validation.scaffold_split import scaffold_split


class FingerprintClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: FingerprintTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = df.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        fp = compute_fp_dict(row["SMILES"])
        ids, mask = self.tokenizer.encode(fp, max_len=self.max_len)
        label = float(row.get("Label", 0))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def load_placeholder_labeled(positives_csv: str, decoys_json: str) -> pd.DataFrame:
    pos = pd.read_csv(positives_csv).rename(columns={"count_PGK2": "PGK2"})
    pos["Label"] = 1
    with open(decoys_json, "r", encoding="utf-8") as f:
        decoys_raw = json.load(f)["compounds"]
    decoys = pd.DataFrame(
        [{"SMILES": c["canonical_smiles"], "Label": 0} for c in decoys_raw if c.get("canonical_smiles")]
    )
    df = pd.concat([pos, decoys], ignore_index=True)
    df["RandomID"] = [f"CACHE7_{i}" for i in range(len(df))]
    return df


def run_finetune(
    pretrained_dir: str = "models/pretrained/delbert",
    train_parquet: str = None,
    test_parquet: str = None,
    positives_csv: str = "data/reference/DEL_hit_candidates_CACHE7.csv",
    decoys_json: str = "data/reference/chembl_repurposing_library.json",
    lora_r: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 16,
    epochs: int = 8,
    lr: float = 1e-3,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    team_name: str = "HSelithBio",
    device: str = "cpu",
):
    print("=" * 70)
    print("[*] DELBERT LoRA FINE-TUNING (Asymmetric Loss)")
    print("=" * 70)

    ckpt_path = os.path.join(pretrained_dir, "delbert_pretrained.pt")
    tok_path = os.path.join(pretrained_dir, "tokenizer.json")
    if not (os.path.exists(ckpt_path) and os.path.exists(tok_path)):
        raise FileNotFoundError(
            f"No pretrained DELBERT found under {pretrained_dir}/. Run scripts/pretrain_delbert.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    tokenizer = FingerprintTokenizer.load(tok_path)
    backbone = DELBERT(**checkpoint["config"])
    backbone.load_state_dict(checkpoint["model_state"])
    backbone = apply_lora(backbone, r=lora_r, alpha=lora_alpha).to(device)
    model = DELBERTClassifier(backbone).to(device)
    for p in model.head.parameters():
        p.requires_grad = True
    print(f"[*] Trainable parameters after LoRA: {count_trainable_parameters(model):,}")

    if train_parquet and os.path.exists(train_parquet):
        df = pd.read_parquet(train_parquet)
        if "RandomID" not in df.columns:
            df["RandomID"] = [f"ROW_{i}" for i in range(len(df))]
    else:
        if train_parquet:
            print(f"[!] --train-parquet '{train_parquet}' not found, using placeholder data.")
        df = load_placeholder_labeled(positives_csv, decoys_json)
        print(
            f"[!] PLACEHOLDER DATA: {int((df['Label'] == 1).sum())} real PGK2 hits vs. "
            f"{int((df['Label'] == 0).sum())} unrelated ChEMBL decoys — smoke test only."
        )

    train_df, val_df = scaffold_split(df, smiles_col="SMILES", val_ratio=0.2, seed=42)
    if train_df["Label"].nunique() < 2:
        print("[!] Training split has only one class — cannot fine-tune yet (need real negatives).")
        return

    max_len = checkpoint["config"]["max_len"]
    train_loader = DataLoader(FingerprintClsDataset(train_df, tokenizer, max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FingerprintClsDataset(val_df, tokenizer, max_len), batch_size=batch_size)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    print(f"[*] Fine-tuning on device={device} | train={len(train_df)} val={len(val_df)} | {epochs} epochs")
    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = asymmetric_loss_torch(logits, labels, gamma_pos=0.0, gamma_neg=asl_gamma_neg, clip=asl_clip)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"    epoch {epoch + 1}/{epochs} | ASL loss = {total_loss / max(n_batches, 1):.4f}")

    model.eval()
    val_scores, val_labels = [], []
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask = ids.to(device), mask.to(device)
            val_scores.append(model(ids, mask).cpu().numpy())
            val_labels.append(labels.numpy())
    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)

    if len(set(val_labels.tolist())) == 2:
        print(f"[*] Validation ROC-AUC={roc_auc_score(val_labels, val_scores):.4f} | "
              f"PR-AUC={average_precision_score(val_labels, val_scores):.4f}")
    else:
        print("[!] Validation split has only one class — skipping ROC/PR-AUC.")

    if test_parquet and os.path.exists(test_parquet):
        print(f"[*] Scoring OFFICIAL ASMS candidate pool: {test_parquet}")
        score_df = pd.read_parquet(test_parquet)
        if "RandomID" not in score_df.columns:
            score_df["RandomID"] = [f"ROW_{i}" for i in range(len(score_df))]
    else:
        print("[!] No --test-parquet given: ranking the held-out validation split instead.")
        score_df = val_df.copy()

    loader = DataLoader(FingerprintClsDataset(score_df.assign(Label=0), tokenizer, max_len), batch_size=batch_size)
    scores = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            ids, mask = ids.to(device), mask.to(device)
            scores.append(model(ids, mask).cpu().numpy())
    score_df = score_df.assign(Score=np.concatenate(scores))

    discovered = top_n_discovered_compounds(score_df, n=min(50, len(score_df)))
    print("[*] Top 50 DELBERT-discovered candidates (the official submission unit):")
    print(discovered[["RandomID", "SMILES", "Score"]].to_string(index=False))

    os.makedirs("submissions", exist_ok=True)
    if len(score_df) >= 50:
        path = generate_submission_txt(score_df, team_name=f"{team_name}_DELBERT")
        print(f"[*] Official-format submission (50 hit IDs) written to: {path}")
    else:
        out_path = os.path.join("submissions", f"delbert_discovered_candidates_{team_name}.csv")
        discovered.to_csv(out_path, index=False)
        print(f"[!] Fewer than 50 scored candidates ({len(score_df)}) — wrote ranked list instead: {out_path}")

    print("=" * 70)
    return discovered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELBERT LoRA fine-tuning + scoring")
    parser.add_argument("--pretrained-dir", default="models/pretrained/delbert")
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--team-name", default="HSelithBio")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_finetune(
        pretrained_dir=args.pretrained_dir,
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        team_name=args.team_name,
        epochs=args.epochs,
        device=args.device,
    )
