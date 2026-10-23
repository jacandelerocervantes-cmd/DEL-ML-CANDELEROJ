"""
Domain-shift feature mask: keeps only the ECFP4 bits that also occur in the
official ASMS validation+test pools at a comparable frequency to the DEL
library. Bits that are common in DEL but nearly absent outside it are
building-block/linker signatures of the DEL library (feature-importance
inspection of the first model showed the top bits were exactly these), so a
model that leans on them cannot say anything about the external pool.

Output: data/processed/pgk2_bit_mask.npz  (keep: int array of kept column idx)
"""

import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scripts.train_pgk2_full import featurize_parallel

ECFP4_CACHE = "data/processed/pgk2_ecfp4_sparse.npz"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
OUT = "data/processed/pgk2_bit_mask.npz"
MIN_DEL_FREQ = 0.002   # ignore bits too rare to matter
MAX_RATIO = 0.25       # drop if pool_freq < MAX_RATIO * del_freq


def presence_freq(X):
    Xb = X.copy().tocsr()
    Xb.data[:] = 1
    return np.asarray(Xb.mean(axis=0)).ravel()


def main():
    X = sp.load_npz(ECFP4_CACHE)
    rng = np.random.RandomState(0)
    rows = np.sort(rng.choice(X.shape[0], 400_000, replace=False))
    del_freq = presence_freq(X[rows])
    del X

    pools = []
    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        for name in ("PGK2_Validation_split.csv", "PGK2_Test_split.csv"):
            with z.open(f"Val-Test-set/{name}") as f:
                pools.append(pd.read_csv(f))
    smiles = pd.concat(pools)["SMILES"].tolist()
    fps, _ = featurize_parallel(smiles)
    P = sp.csr_matrix(np.stack([fp for fp in fps if fp is not None]))
    pool_freq = presence_freq(P)

    drop = (del_freq >= MIN_DEL_FREQ) & (pool_freq < MAX_RATIO * del_freq)
    keep = np.where(~drop)[0]
    np.savez(OUT, keep=keep, del_freq=del_freq, pool_freq=pool_freq)
    print(f"[*] pool compounds: {P.shape[0]:,} | bits kept {len(keep)} / 2048, dropped {int(drop.sum())}", flush=True)
    for b in (1549, 1311, 1517, 1883):
        print(f"    bit {b}: del_freq={del_freq[b]:.4f} pool_freq={pool_freq[b]:.4f} dropped={bool(drop[b])}", flush=True)


if __name__ == "__main__":
    main()
