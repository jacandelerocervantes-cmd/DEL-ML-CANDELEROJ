"""exp69: does adding the validation split's identified hits (and inactives) as extra training data help recover held-out TEST hits from other chemical series?
Protocol of exp21b (cluster-held-out CV on the 6,580 Active Learning labels, groups = Butina at ECFP4 Tanimoto 0.3, StratifiedGroupKFold(5), seeds 0-2, logistic regression C=0.05 balanced on ECFP6 log counts,
30,000 background negatives from the unlabelled pool); held-out hits are ranked among the unlabelled pool minus the background negatives; cutoffs scaled to 178,052.
Variants: V0 baseline (AL train folds + background negatives); V1 + the 32 validation hits as extra positives; V2 + 32 validation hits + the 2,592 validation inactives;
V3 + 32 validation hits with weight 3 (they are a different chemotype and few). Inputs: validation hits/inactives (validation results, not published). RDKit venv. Output: data/processed/gp/exp69_add_validation_hits.json"""
import numpy as np, pandas as pd, json, os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
g6 = G.GetMorganGenerator(radius=3, fpSize=2048); g4 = G.GetMorganGenerator(radius=2, fpSize=2048)
fx = lambda g, Sm: np.minimum(np.array([g.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in Sm]), 255).astype(np.uint8)
al = pd.read_csv("data/raw/active_learning_data.csv"); y = al.Label.values; n = len(al)
P = pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc = [c for c in P.columns if 'smi' in c.lower()][0]
U = P[~P.CatalogID.isin(al.CatalogID)].reset_index(drop=True); nu = len(U)
A = fx(g6, al.RDKit_SMILES)
XUp = "data/processed/XU.npy"
if os.path.exists(XUp): XU = np.load(XUp)
else: XU = np.vstack([fx(g6, U[sc].iloc[i:i + 20000]) for i in range(0, nu, 20000)]); np.save(XUp, XU)
V = pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); vsc = [c for c in V.columns if 'smi' in c.lower()][0]; vs = dict(zip(V.CatalogID, V[vsc]))
lab = pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv")
VH = fx(g6, [vs[c] for c in lab[lab.confirmed_label == 1].CatalogID]); VN = fx(g6, [vs[c] for c in lab[lab.confirmed_label == 0].CatalogID])
print("pool", nu, "| hits validation", len(VH), "| inactivos validation", len(VN), flush=True)
bv = [g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist = []
for i in range(1, n): dist += [1 - s for s in DataStructs.BulkTanimotoSimilarity(bv[i], bv[:i])]
grp = np.zeros(n, int)
for k, c in enumerate(Butina.ClusterData(dist, n, 0.7, isDistData=True)): grp[list(c)] = k
L = lambda X: np.log1p(X.astype(np.float32)); V_ = ["V0", "V1", "V2", "V3"]; R = {k: [] for k in V_}
for seed in range(3):
    rng = np.random.default_rng(seed); extra = rng.choice(nu, 30000, replace=False); keep = np.ones(nu, bool); keep[extra] = False; XK = XU[keep]
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(A, y, grp):
        base_X = np.vstack([A[tr], XU[extra]]); base_y = np.r_[y[tr], np.zeros(len(extra))]
        for v in V_:
            X, yy, w = base_X, base_y, np.ones(len(base_y))
            if v in ("V1", "V2", "V3"):
                X = np.vstack([X, VH]); yy = np.r_[yy, np.ones(len(VH))]; w = np.r_[w, np.full(len(VH), 3.0 if v == "V3" else 1.0)]
            if v == "V2": X = np.vstack([X, VN]); yy = np.r_[yy, np.zeros(len(VN))]; w = np.r_[w, np.ones(len(VN))]
            m = LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000).fit(L(X), yy, sample_weight=w)
            su = np.sort(np.concatenate([m.decision_function(L(XK[i:i + 20000])) for i in range(0, len(XK), 20000)]))
            sh = m.decision_function(L(A[te][y[te] == 1])); R[v] += list(len(su) - np.searchsorted(su, sh, side='right') + 1)
    print("semilla", seed, "lista", flush=True)
res = {}
for v in V_:
    r = np.array(R[v]).reshape(3, -1); sc_ = len(XK) / 178052
    res[v] = {f"top{k}_equiv178k": [int((r[s] <= k * sc_).sum()) for s in range(3)] for k in (500, 1000, 2000, 5000)}; res[v]["rango_mediano"] = [int(np.median(r[s])) for s in range(3)]
print(json.dumps(res, indent=1)); json.dump(res, open("data/processed/gp/exp69_add_validation_hits.json", "w"), indent=1)
