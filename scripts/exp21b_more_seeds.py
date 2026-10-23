"""exp21b_more_seeds: extension of exp21b (same cluster-held-out cross-validation, same code) to seeds 3-7 and to N = 0 and 100,000 extra background negatives only, so that together with seeds 0-2 of exp21b there are 8 seeds. Run in Python 3.12.0, scikit-learn 1.9.1 for the 8-seed figures of the writeups (seeds 0-2 are the rerun of exp21b in that environment). Output: data/processed/gp/exp21b_seeds_3to7.json. Reads data/processed/XU.npy if present (written by exp21b). Seeded."""
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
L=lambda X:np.log1p(X.astype(np.float32)); NS=[0,100000]; R={k:[] for k in NS}
for seed in range(3,8):
    rng=np.random.default_rng(seed); extra=rng.choice(nu,100000,replace=False); keep=np.ones(nu,bool); keep[extra]=False; XK=XU[keep]
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
    r=np.array(R[N]).reshape(5,-1); res[f"N_extra={N}"]={f"top{k}_of_178k_equiv":[int((r[s]<=k*len(XK)/178052).sum()) for s in range(5)] for k in (500,1000,2000,5000)}; res[f"N_extra={N}"]["rango_mediano"]=[int(np.median(r[s])) for s in range(5)]
print(json.dumps(res,indent=1)); json.dump(res,open("data/processed/gp/exp21b_seeds_3to7.json","w"),indent=1)
