"""exp21 (ruta 1): do random unlabelled-pool compounds used as extra negatives (positive-unlabeled style) improve held-out-hit recall in the pool?
Same cluster-held-out CV as exp20 (Butina sim>=0.3, StratifiedGroupKFold, seeds 0-2). For each fold and seed:
  - sample N extra negatives from the unlabelled pool (N in 0, 30000), train logistic regression (ECFP6 counts, balanced) on AL train folds + extras,
  - rank the held-out hits among the unlabelled pool EXCLUDING the sampled extras (same ranking pool for N=0 and N>0, so the comparison is fair).
Reports hits of 39 in the top 500 / 1000 / 2000 / 5000 of that pool. Seeded."""
import numpy as np, pandas as pd, json, os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
S="data/processed"
g6=G.GetMorganGenerator(radius=3,fpSize=2048); g4=G.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda g,Sm:np.minimum(np.array([g.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in Sm]),255).astype(np.uint8)
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; n=len(al)
P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
U=P[~P.CatalogID.isin(al.CatalogID)].reset_index(drop=True); nu=len(U)
A=fx(g6,al.RDKit_SMILES)
if os.path.exists(S+"/XU.npy"): XU=np.load(S+"/XU.npy")
else: XU=np.vstack([fx(g6,U[sc].iloc[i:i+20000]) for i in range(0,nu,20000)]); np.save(S+"/XU.npy",XU)
print("pool",nu,"fingerprints listos",flush=True)
bv=[g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
L=lambda X:np.log1p(X.astype(np.float32)); NS=[0,30000]; R={k:[] for k in NS}
for seed in range(3):
    rng=np.random.default_rng(seed); extra=rng.choice(nu,30000,replace=False); keep=np.ones(nu,bool); keep[extra]=False; XK=XU[keep]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(A,y,g):
        for N in NS:
            X=A[tr]; yy=y[tr]
            if N: X=np.vstack([X,XU[extra[:N]]]); yy=np.r_[yy,np.zeros(N)]
            m=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(L(X),yy)
            su=np.sort(np.concatenate([m.decision_function(L(XK[i:i+20000])) for i in range(0,len(XK),20000)]))
            sh=m.decision_function(L(A[te][y[te]==1])); R[N]+=list(len(su)-np.searchsorted(su,sh,side='right')+1)
    print("semilla",seed,"listo",flush=True)
res={}
for N in NS:
    r=np.array(R[N]).reshape(3,-1); res[f"N_extra={N}"]={f"top{k}":[int((r[s]<=k).sum()) for s in range(3)] for k in (500,1000,2000,5000)}; res[f"N_extra={N}"]["rango_mediano"]=[int(np.median(r[s])) for s in range(3)]
print(json.dumps(res,indent=1)); json.dump(res,open("data/processed/gp/exp21_background_negatives.json","w"),indent=1)
