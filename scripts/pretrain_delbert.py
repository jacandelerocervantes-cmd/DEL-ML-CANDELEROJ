"""
DELBERT self-supervised MLM pre-training on unlabeled DEL fingerprints.

Default hyperparameters (d_model=128, 4 layers, small batch) are sized to
finish in minutes on a CPU laptop against the placeholder corpus
(`data/reference/DEL_hit_candidates_CACHE7.csv`, ~1.4k compounds) purely to
prove the architecture, tokenizer, and training loop are correct end to end.

TO SCALE TO THE REAL 898M-COMPOUND PGK2 CORPUS
------------------------------------------------
This is a GPU-and-cloud job, not a laptop job. Once
`data/raw/PGK2_selection.parquet` is available:
  1. Provision a GPU VM, e.g.:
     gcloud compute instances create delbert-pretrain \
       --zone=us-central1-a --machine-type=a2-highgpu-1g \
       --accelerator=type=nvidia-tesla-a100,count=1 \
       --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
       --boot-disk-size=500GB
     (or submit as a Vertex AI custom training job instead of a raw VM.)
  2. rsync this repo + the parquet file to the VM.
  3. Re-run this exact script with larger flags, e.g.:
     python scripts/pretrain_delbert.py --input-parquet data/raw/PGK2_selection.parquet \
       --d-model 512 --n-layers 8 --n-heads 8 --batch-size 512 --epochs 3 --device cuda
  No code changes needed — only the entry-point flags and the machine.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from modules.delbert.model import DELBERT, mask_tokens
from modules.delbert.tokenizer import FingerprintTokenizer, compute_fp_dict


class FingerprintMLMDataset(Dataset):
    def __init__(self, smiles_list, tokenizer: FingerprintTokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for smi in smiles_list:
            fp = compute_fp_dict(smi)
            if fp:
                self.examples.append(fp)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, mask = self.tokenizer.encode(self.examples[idx], max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def load_unlabeled_smiles(input_parquet: str, positives_csv: str, decoys_json: str) -> list:
    if input_parquet and os.path.exists(input_parquet):
        df = pd.read_parquet(input_parquet, columns=["SMILES"])
        print(f"[*] Loaded {len(df)} SMILES from official parquet: {input_parquet}")
        return df["SMILES"].tolist()

    import json

    pos = pd.read_csv(positives_csv)["SMILES"].tolist()
    with open(decoys_json, "r", encoding="utf-8") as f:
        decoys = [c["canonical_smiles"] for c in json.load(f)["compounds"] if c.get("canonical_smiles")]
    print(
        f"[!] No official parquet given/found — pre-training on the {len(pos)} placeholder "
        f"PGK2 hits + {len(decoys)} ChEMBL decoys only, as an architecture/pipeline smoke test."
    )
    return pos + decoys


def run_pretrain(
    input_parquet: str = None,
    positives_csv: str = "data/reference/DEL_hit_candidates_CACHE7.csv",
    decoys_json: str = "data/reference/chembl_repurposing_library.json",
    output_dir: str = "models/pretrained/delbert",
    max_len: int = 256,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
):
    print("=" * 70)
    print("[*] DELBERT MLM PRE-TRAINING")
    print("=" * 70)

    smiles_list = load_unlabeled_smiles(input_parquet, positives_csv, decoys_json)

    print("[*] Computing multi-family fingerprints (ECFP4 + MACCS)...")
    fp_dicts = [compute_fp_dict(s) for s in smiles_list]
    fp_dicts = [fp for fp in fp_dicts if fp]
    print(f"[*] {len(fp_dicts)} / {len(smiles_list)} molecules featurized successfully.")

    tokenizer = FingerprintTokenizer().fit(fp_dicts)
    print(f"[*] Vocabulary size: {tokenizer.vocab_size()} tokens.")

    dataset = FingerprintMLMDataset(smiles_list, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DELBERT(
        vocab_size=tokenizer.vocab_size(), d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    mask_id = tokenizer.token_to_id["[MASK]"]
    special_ids = {tokenizer.token_to_id[t] for t in ["[PAD]", "[MASK]", "[CLS]", "[UNK]"]}

    print(f"[*] Training on device={device} | {len(dataset)} examples | {epochs} epochs")
    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for input_ids, attention_mask in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            masked_ids, labels = mask_tokens(
                input_ids, attention_mask, mask_id, tokenizer.vocab_size(), special_token_ids=special_ids
            )
            logits = model.forward_mlm(masked_ids, attention_mask)
            loss = loss_fn(logits.view(-1, tokenizer.vocab_size()), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"    epoch {epoch + 1}/{epochs} | MLM loss = {total_loss / max(n_batches, 1):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"vocab_size": tokenizer.vocab_size(), "d_model": d_model, "n_heads": n_heads,
                       "n_layers": n_layers, "d_ff": d_ff, "max_len": max_len},
        },
        os.path.join(output_dir, "delbert_pretrained.pt"),
    )
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"[*] Saved pretrained DELBERT + tokenizer to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELBERT MLM pre-training")
    parser.add_argument("--input-parquet", default=None, help="Official unlabeled DEL parquet (SMILES column)")
    parser.add_argument("--output-dir", default="models/pretrained/delbert")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_pretrain(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
