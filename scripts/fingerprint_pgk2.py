"""
Computes ECFP4 fingerprints (HitGen-matching: radius=2, 2048 bits, hashed
counts) for the full labeled PGK2 DEL selection, parallelized via joblib
(loky backend, one process per core) so it scales past a single-core RDKit
loop. Meant to run on the
GCP VM (8+ vCPUs) against the full 7.7M-compound file — on a laptop this
would be the bottleneck the whole project has been routing around.

Input:  data/processed/PGK2_selection_labeled.parquet (or raw + on-the-fly
        labeling if the labeled file isn't present — see label_pgk2_real_data.py)
Output: data/processed/PGK2_selection_features.parquet
        (RandomID, SMILES, Label, ECFP4 columns; historic_hits kept as an
        auxiliary feature)
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from joblib import Parallel, delayed

LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"
RAW_PARQUET = "data/raw/PGK2_selection.parquet"
OUTPUT_PARQUET = "data/processed/PGK2_selection_features.parquet"


_MORGAN_GEN = None


def smiles_to_ecfp4_counts(smiles: str):
    # NOTE: this originally used AllChem.GetHashedMorganFingerprint, which
    # prints a C++-level "please use MorganGenerator" line to stderr on
    # EVERY call — a message that lives outside RDKit's Python logging
    # shim, so RDLogger.DisableLog("rdApp.*") cannot suppress it (verified:
    # it was still present even with that disabled at import time). With 8
    # parallel workers each hammering stderr for 7.7M molecules, that I/O
    # was the actual bottleneck, not the fingerprinting itself. Using the
    # modern rdFingerprintGenerator API (same radius/bits/hashed-count
    # semantics as HitGenECFP4) avoids the warning at the source.
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _MORGAN_GEN.GetCountFingerprint(mol)
    arr = np.zeros(2048, dtype=np.int16)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx] = min(count, 32767)
    return arr


def load_source() -> pd.DataFrame:
    if os.path.exists(LABELED_PARQUET):
        print(f"[*] Loading pre-labeled data from {LABELED_PARQUET}")
        return pq.read_table(LABELED_PARQUET, columns=["RandomID", "SMILES", "Label", "historic_hits"]).to_pandas()

    print(f"[!] {LABELED_PARQUET} not found — deriving labels on the fly from {RAW_PARQUET}")
    cols = ["compound", "SMILES", "count_PGK2", "historic_hits"]
    df = pq.read_table(RAW_PARQUET, columns=cols).to_pandas()
    df["RandomID"] = df["compound"]
    df["Label"] = ((df["count_PGK2"] >= 3) | (df["historic_hits"] >= 1)).astype(np.int8)
    return df[["RandomID", "SMILES", "Label", "historic_hits"]]


def _compute_batch(smiles_batch):
    return [smiles_to_ecfp4_counts(s) for s in smiles_batch]


def main(chunk_size: int = 200_000, n_jobs: int = None):
    print("=" * 70)
    print("[*] PGK2 FULL-SCALE FINGERPRINTING (ECFP4, joblib/loky-parallelized)")
    print("=" * 70)

    n_jobs = n_jobs or os.cpu_count() or 4
    print(f"[*] Using {n_jobs} worker processes (loky backend).")

    df = load_source()
    print(f"[*] {len(df):,} compounds to featurize | {int(df['Label'].sum()):,} positive labels")

    os.makedirs("data/processed", exist_ok=True)
    writer = None
    n_done, n_valid = 0, 0
    t0 = time.time()

    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            smiles_list = chunk["SMILES"].tolist()

            # Split this chunk into one sub-batch per worker so each loky
            # task amortizes RDKit/process overhead over many molecules,
            # instead of dispatching one task per molecule.
            sub_batches = np.array_split(smiles_list, n_jobs)
            results = parallel(delayed(_compute_batch)(list(b)) for b in sub_batches if len(b) > 0)
            fps = [fp for batch_result in results for fp in batch_result]

            valid_mask = [fp is not None for fp in fps]
            out_chunk = chunk[valid_mask].copy()
            out_chunk["ECFP4"] = [fp.tolist() for fp, ok in zip(fps, valid_mask) if ok]

            table = pa.Table.from_pandas(out_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)
            writer.write_table(table)

            n_done += len(chunk)
            n_valid += len(out_chunk)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(df) - n_done) / rate if rate > 0 else float("inf")
            print(f"    {n_done:,}/{len(df):,} ({n_valid:,} valid) | {rate:.0f} mol/s | ETA {eta / 60:.1f} min")

    if writer:
        writer.close()
    print(f"[*] Done in {(time.time() - t0) / 60:.1f} min. Wrote {n_valid:,} featurized compounds to {OUTPUT_PARQUET}")
    print("=" * 70)


if __name__ == "__main__":
    main()
