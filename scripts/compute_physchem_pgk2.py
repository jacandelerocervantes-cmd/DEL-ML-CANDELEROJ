"""
Computes a small set of physicochemical descriptors (MolWt, LogP, TPSA,
HBD, HBA, RotatableBonds, RingCount, FractionCSP3) for every compound in
PGK2_selection_features.parquet, in the SAME row order, so they can be
concatenated to the ECFP4 sparse matrix as extra dense columns.

Why: feature-importance inspection of the first XGBoost model showed the
top-importance ECFP4 bits map to DEL synthesis-handle/building-block
fragments (e.g. a dimethylaminomethyl tail, a specific heterocyclic core)
that are massively enriched in the DEL positive class purely because of
which building blocks were used to make the library — not because they are
real PGK2 pharmacophores. That model scored ~100% confidence on the entire
external ASMS validation pool (no discrimination at all), because none of
those Enamine/REAL compounds carry the DEL-specific synthetic signature.
Physchem descriptors don't encode "which building block library made this"
and should be harder for the model to use as that kind of shortcut.

Output: data/processed/pgk2_physchem.npz (dense float32 array, row-aligned
with PGK2_selection_features.parquet).
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

FEATURES_PARQUET = "data/processed/PGK2_selection_features.parquet"
OUTPUT_NPZ = "data/processed/pgk2_physchem.npz"

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3",
]


def _compute_batch(smiles_batch):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    out = np.zeros((len(smiles_batch), len(DESCRIPTOR_NAMES)), dtype=np.float32)
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
        ]
    return out


def main(chunk_size: int = 200_000, n_jobs: int = None):
    print("=" * 70)
    print("[*] PGK2 PHYSCHEM DESCRIPTORS (row-aligned with features parquet)", flush=True)
    print("=" * 70)

    n_jobs = n_jobs or os.cpu_count() or 4
    smiles = pq.read_table(FEATURES_PARQUET, columns=["SMILES"]).column("SMILES").to_pylist()
    n = len(smiles)
    print(f"[*] {n:,} compounds to describe, {n_jobs} workers.", flush=True)

    blocks = []
    t0 = time.time()
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for start in range(0, n, chunk_size):
            chunk = smiles[start:start + chunk_size]
            sub_batches = np.array_split(chunk, n_jobs)
            results = parallel(delayed(_compute_batch)(list(b)) for b in sub_batches if len(b) > 0)
            blocks.append(np.vstack(results))
            done = start + len(chunk)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            print(f"    {done:,}/{n:,} | {rate:.0f} mol/s | ETA {eta / 60:.1f} min", flush=True)

    X = np.vstack(blocks)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ, X=X, names=np.array(DESCRIPTOR_NAMES))
    print(f"[*] Done in {(time.time() - t0) / 60:.1f} min. Wrote {X.shape} to {OUTPUT_NPZ}", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
