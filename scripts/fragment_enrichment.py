"""
Fragment-enrichment scoring + series-diverse selection for the PGK2 pool.

Instead of a whole-molecule classifier (which saturates on the external ASMS
pool because DEL and pool molecules share fragments, not molecules), each
ECFP4 bit gets a naive-Bayes-style log-enrichment weight: how much more often
it appears in DEL hits than in DEL non-hits. Only bits that exist in the
official pools are eligible. A compound's score is the sum of weights of the
bits it carries.

The leaderboard ranks first by the number of distinct chemical series among
the 50 picks, so the final 50 are taken ~5 each from ~10 clusters of the
top-scoring pool compounds, not as a flat top-50.

Weights are fit on the scaffold-split train rows and checked on the held-out
rows (ROC/PR-AUC) before scoring the official validation pool.
"""

import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.train_pgk2_full import featurize_parallel, load_metadata

RDLogger.DisableLog("rdApp.*")

ECFP4_CACHE = "data/processed/pgk2_ecfp4_sparse.npz"
SPLIT_CACHE = "data/processed/pgk2_scaffold_split_idx.npz"
MASK = "data/processed/pgk2_bit_mask.npz"
VAL_TEST_ZIP = "data/raw/Val-Test-set.zip"
ALPHA = 1.0
MIN_POS_COUNT = 20
MIN_POOL_FREQ = 0.0005
MIN_WEIGHT = 0.5
MAX_WEIGHT = 4.0
POOL_TOP_N = 3000
N_SERIES = 10
PER_SERIES = 5
SUBMISSION_INDEX = 6
GENERIC_PENALTY = 1.0
LABELED_PARQUET = "data/processed/PGK2_selection_labeled.parquet"


def binarize(X):
    Xb = X.tocsr().copy()
    Xb.data[:] = 1
    return Xb.astype(np.float32)


def main():
    import pyarrow.parquet as pq
    meta = load_metadata()
    y = meta["Label"].to_numpy()
    aux = pq.read_table(LABELED_PARQUET, columns=["RandomID", "count_PGK2", "count_NTC", "historic_hits"]).to_pandas()
    meta = meta.merge(aux, on="RandomID", how="left")
    y_gen = ((meta["count_PGK2"] >= 3) & (meta["count_NTC"] == 0) & (meta["historic_hits"] >= 5)).to_numpy().astype(int)
    print(f"[*] promiscuous 'generic DEL binder' set (historic_hits>=5): {int(y_gen.sum()):,} compounds", flush=True)
    idx = np.load(SPLIT_CACHE)
    train_idx, val_idx = idx["train_idx"], idx["val_idx"]

    X = binarize(sp.load_npz(ECFP4_CACHE))
    Xtr, ytr, ygtr = X[train_idx], y[train_idx], y_gen[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    del X

    n1 = np.asarray(Xtr[ytr == 1].sum(axis=0)).ravel()
    n_all = np.asarray(Xtr.sum(axis=0)).ravel()
    n0 = n_all - n1
    N1, N0 = ytr.sum(), (ytr == 0).sum()
    w = np.log((n1 + ALPHA) / (N1 + ALPHA)) - np.log((n0 + ALPHA) / (N0 + ALPHA))
    g1 = np.asarray(Xtr[ygtr == 1].sum(axis=0)).ravel()
    g0 = n_all - g1
    G1, G0 = ygtr.sum(), (ygtr == 0).sum()
    w_gen = np.log((g1 + ALPHA) / (G1 + ALPHA)) - np.log((g0 + ALPHA) / (G0 + ALPHA))
    w_raw = w.copy()
    w = w - GENERIC_PENALTY * np.maximum(w_gen, 0)
    print(f"[*] bits whose weight dropped by >1 after removing generic-DEL enrichment: "
          f"{int(((w_raw - w) > 1).sum())}", flush=True)
    del Xtr

    with zipfile.ZipFile(VAL_TEST_ZIP) as z:
        pools = []
        for name in ("PGK2_Validation_split.csv", "PGK2_Test_split.csv"):
            with z.open(f"Val-Test-set/{name}") as f:
                pools.append(pd.read_csv(f))
    val_pool = pools[0]
    all_smiles = pd.concat(pools)["SMILES"].tolist()
    fps_all, _ = featurize_parallel(all_smiles)
    pool_freq = (np.stack([f for f in fps_all if f is not None]) > 0).mean(axis=0)
    del fps_all

    eligible = (n1 >= MIN_POS_COUNT) & (pool_freq >= MIN_POOL_FREQ) & (w >= MIN_WEIGHT)
    w_use = np.where(eligible, np.minimum(w, MAX_WEIGHT), 0.0).astype(np.float32)
    print(f"[*] eligible bits: {int(eligible.sum())} / 2048", flush=True)

    top_bits = np.argsort(-w_use)[:12]
    for b in top_bits:
        print(f"    bit {b}: w_specific={w_use[b]:.2f} (raw {w_raw[b]:.2f}, generic {max(w_gen[b], 0):.2f}) "
              f"pos_count={int(n1[b])} pool_freq={pool_freq[b]:.4f}", flush=True)

    sva = Xva @ w_use
    print(f"[*] held-out DEL split: ROC-AUC={roc_auc_score(yva, sva):.4f} "
          f"PR-AUC={average_precision_score(yva, sva):.4f}", flush=True)

    fps, _ = featurize_parallel(val_pool["SMILES"].tolist())
    ok = [f is not None for f in fps]
    pool = val_pool[ok].reset_index(drop=True)
    Xp = np.stack([f for f in fps if f is not None])
    pool["Score"] = ((Xp > 0).astype(np.float32) @ w_use)
    pool = pool.sort_values("Score", ascending=False).reset_index(drop=True)
    print(f"[*] official pool score: max={pool.Score.max():.2f} median={pool.Score.median():.2f} "
          f"n(score>0)={(pool.Score > 0).sum():,} | distinct score values={pool.Score.nunique():,}", flush=True)

    top = pool.head(POOL_TOP_N).copy()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top["SMILES"]]
    dists = []
    for i in range(1, len(bvs)):
        sims = DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i])
        dists.extend(1 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(bvs), 0.65, isDistData=True)
    top["Cluster"] = -1
    for cid, members in enumerate(clusters):
        top.loc[list(members), "Cluster"] = cid
    cl = (top.groupby("Cluster").agg(best=("Score", "max"), size=("Score", "size"))
          .query("size >= 3").sort_values("best", ascending=False))
    print(f"[*] {len(clusters)} clusters in top {POOL_TOP_N}; {len(cl)} with >=3 members", flush=True)

    picks = []
    for cid in cl.index[:N_SERIES]:
        picks.append(top[top.Cluster == cid].sort_values("Score", ascending=False).head(PER_SERIES))
    chosen = pd.concat(picks)
    if len(chosen) < 50:
        rest = top[~top.CatalogID.isin(chosen.CatalogID)].head(50 - len(chosen))
        chosen = pd.concat([chosen, rest])
    chosen = chosen.head(50)
    assert chosen.CatalogID.nunique() == 50

    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/Team_candeleroj_{SUBMISSION_INDEX}.txt"
    chosen["CatalogID"].to_csv(path, index=False, header=False)
    chosen.to_csv(f"submissions/Team_candeleroj_{SUBMISSION_INDEX}_detail.csv", index=False)
    print(f"[*] wrote {path}; series used: {chosen.Cluster.nunique()}", flush=True)
    print(chosen[["CatalogID", "Cluster", "Score", "SMILES"]].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
