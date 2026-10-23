"""exp54: does an ENSEMBLE of Chemprop + logistic regression beat either alone? Uses the per-fold Chemprop predictions saved by kaggle/pgk2_route2/chemprop_cv.py (preds_s{seed}_f{fold}.npz),
and reproduces locally, on the SAME folds, extras and pool, the logistic regression (ECFP6 counts, C=0.05, balanced, trained on the train folds + the same 30,000 background negatives).
Held-out hits are ranked among the unlabelled pool minus the extras; combinations are z-scored over the pool then blended: w*Chemprop + (1-w)*LR. Cutoffs scaled to 178,052. RDKit venv."""
import glob, os, sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
def load_pool_features(path):
    """ECFP6 count fingerprints (2048 bits, counts capped at 255) of the unlabelled pool = test split minus the labelled compounds, in file order (as in kaggle/pgk2_route2/chemprop_cv.py). Cached in path."""
    import os
    if os.path.exists(path): return np.load(path)
    pool=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); lab=pd.read_csv("data/raw/active_learning_data.csv"); U=pool[~pool.CatalogID.isin(lab.CatalogID)].reset_index(drop=True)
    scol=[c for c in U.columns if 'smi' in c.lower()][0]; g=G.GetMorganGenerator(radius=3,fpSize=2048)
    X=np.vstack([np.minimum(g.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(x)),255).astype(np.uint8) for x in U[scol]]); np.save(path,X); return X
D=sys.argv[1]; S="data/processed"
al=pd.read_csv("data/raw/active_learning_data.csv"); grp=pd.read_csv("kaggle/pgk2_route2/al_groups.csv"); assert (al.CatalogID.values==grp.CatalogID.values).all()
y=al.Label.values; groups=grp.group.values; n=len(al); XU=load_pool_features(S+"/XU.npy"); nu=len(XU); N_EXTRA=30000
g6=G.GetMorganGenerator(radius=3,fpSize=2048); XA=np.log1p(np.minimum(np.array([g6.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]),255).astype(np.float32))
L=lambda X:np.log1p(X.astype(np.float32)); W=[0,0.25,0.5,0.75,1.0]; ranks={w:[] for w in W}; seeds=sorted({int(os.path.basename(f).split('_s')[1].split('_')[0]) for f in glob.glob(D+"/preds_s*_f*.npz")}); print("semillas disponibles:",seeds)
for seed in seeds:
    rng=np.random.default_rng(seed); extra=np.sort(rng.choice(nu,N_EXTRA,replace=False)); keep=np.setdiff1d(np.arange(nu),extra)
    for fi in range(5):
        f=f"{D}/preds_s{seed}_f{fi}.npz"
        if not os.path.exists(f): continue
        z=np.load(f); assert (z["keep_idx"]==keep).all(), "los extras no coinciden con los de Kaggle"
        te=z["te"]; tr=np.setdiff1d(np.arange(n),te)   # folds tomados del npz (sklearn de Kaggle da otra particion que el local)
        m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=2000).fit(np.vstack([XA[tr],L(XU[extra])]),np.r_[y[tr],np.zeros(N_EXTRA)])
        hits=z["hits_te"]; lr_pool=m.decision_function(L(XU[keep])); lr_hits=m.decision_function(XA[hits]); cp_pool=z["sc_pool"].astype(np.float64); cp_hits=z["sc_hits"].astype(np.float64)
        zl=lambda a,ref:(a-ref.mean())/ref.std(); 
        for w in W:
            pool=w*zl(cp_pool,cp_pool)+(1-w)*zl(lr_pool,lr_pool); hh=w*zl(cp_hits,cp_pool)+(1-w)*zl(lr_hits,lr_pool); su=np.sort(pool)
            ranks[w]+=list(len(su)-np.searchsorted(su,hh,side="right")+1)
        print(f"semilla {seed} fold {fi} lista",flush=True)
scale=len(keep)/178052
print("\npeso de Chemprop | hits retenidos | top500 | top1000 | top2000 | top5000 | rango mediano   (promedio por semilla)")
for w in W:
    r=np.array(ranks[w]); k=len(seeds)
    print(f"  {w:4.2f} ({'solo LR' if w==0 else 'solo Chemprop' if w==1 else 'ensamble'})  | {len(r)//k:3d} | "+" | ".join(f"{(r<=c*scale).sum()/k:5.1f}" for c in (500,1000,2000,5000))+f" | {int(np.median(r))}")
