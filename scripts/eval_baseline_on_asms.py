"""
Honest evaluation of the official-baseline recipe against real ASMS feedback.

Why this design. The 1,609 confirmed negatives and 5 confirmed hits are NOT training
data -- with 5 positives, training on them would teach nothing and any cross-validated
metric would be noise. They are something more valuable: real assay outcomes on the
Validation split that no model here has ever seen. So we train the official baseline
recipe on the DEL data (the challenge's actual training signal) and use the confirmed
outcomes purely as a held-out test set.

Official baseline, quoted in the rules: "a baseline LightGBM model trained on ECFP4
using all DEL-hits and 2 times that number of negatives" -- it scored 5 hits / 3 series
in the Validation top-50 and 1 hit in the Test top-50. That is the number to beat.

The DEL library and the ASMS library are different collections, so this evaluation is
genuinely out-of-distribution by construction -- which is exactly the gap that matters
for the Test split, and is a stronger test than any scaffold split within one library.
"""
import numpy as np, pandas as pd, scipy.sparse as sp, lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog("rdApp.*")
RNG = np.random.default_rng(0)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def log(m): print(m, flush=True)

def featurize(smiles):
    rows, cols, vals = [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for k, v in GEN.GetCountFingerprint(m).GetNonzeroElements().items():
            rows.append(i); cols.append(k); vals.append(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(smiles), 2048), dtype=np.float32)

log("Loading DEL labels and precomputed ECFP4...")
lab = pd.read_parquet("data/processed/PGK2_selection_labeled.parquet", columns=["Label"])
X = sp.load_npz("data/processed/pgk2_ecfp4_sparse_local.npz").tocsr()
y = lab.Label.values
pos = np.flatnonzero(y == 1)
neg = RNG.choice(np.flatnonzero(y == 0), size=2 * len(pos), replace=False)
log(f"  official ratio: {len(pos):,} DEL-hits + {len(neg):,} negatives")
idx = np.concatenate([pos, neg])
Xtr, ytr = X[idx].astype(np.float32), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

log("Training LightGBM on ECFP4...")
clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                         n_jobs=-1, verbose=-1, random_state=0)
clf.fit(Xtr, ytr)

log("Scoring the confirmed-outcome evaluation set...")
val = pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv")
conf = pd.read_csv("data/processed/confirmed_submission_outcomes.csv")
conf = conf[conf.confirmed_label.notna()]
ev = val.merge(conf[["CatalogID", "confirmed_label"]], on="CatalogID")
log(f"  {len(ev):,} compounds with a known ASMS label "
    f"({int((ev.confirmed_label==1).sum())} hits, {int((ev.confirmed_label==0).sum())} negatives)")
s_ev = clf.predict_proba(featurize(ev.SMILES.tolist()))[:, 1]
yv = ev.confirmed_label.values.astype(int)

def boot(fn, n=2000):
    v = []
    pi, ni = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    for _ in range(n):
        b = np.concatenate([RNG.choice(pi, len(pi)), RNG.choice(ni, len(ni))])
        if len(set(yv[b])) < 2: continue
        v.append(fn(yv[b], s_ev[b]))
    return np.percentile(v, [2.5, 97.5])

for name, fn in [("ROC-AUC", roc_auc_score), ("PR-AUC", average_precision_score)]:
    lo, hi = boot(fn)
    log(f"  {name}: {fn(yv, s_ev):.4f}   IC95 bootstrap [{lo:.4f}, {hi:.4f}]")
log(f"  PR-AUC de un clasificador aleatorio = prevalencia = {yv.mean():.4f}")

log("Scoring the full 244K Validation split (where do the 5 hits rank?)...")
s_all = clf.predict_proba(featurize(val.SMILES.tolist()))[:, 1]
val["score"] = s_all
val["rank"] = val.score.rank(ascending=False, method="min")
hits = val[val.CatalogID.isin(conf[conf.confirmed_label == 1].CatalogID)]
for _, r in hits.iterrows():
    log(f"    {r.CatalogID:<14} rank {int(r['rank']):>7,} / {len(val):,}  "
        f"(percentil {100*r['rank']/len(val):.2f}%)")
top50 = val.nsmallest(50, "rank")
log(f"  hits confirmados dentro del top-50 del modelo: "
    f"{int(top50.CatalogID.isin(hits.CatalogID).sum())}  (baseline oficial: 5)")
val[["CatalogID", "score"]].to_csv("data/processed/baseline_lgbm_validation_scores.csv", index=False)
log("Wrote data/processed/baseline_lgbm_validation_scores.csv")
