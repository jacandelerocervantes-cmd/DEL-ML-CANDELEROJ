"""exp20: target metric for GPU routes. Cluster-held-out CV on the 6,580 AL labels (Butina sim>=0.3, StratifiedGroupKFold, seeds 0-2).
For each fold: train on the other folds' labels, score ALL unlabelled test-pool compounds (178k) AND the held-out AL compounds with the same model;
rank each held-out hit among the unlabelled pool. Report how many of the 39 hits land in the pool's top 500 / 1000 / 2000 / 5000 (pool ~178k).
Model: logistic regression on ECFP6 counts (best single model in exp13). Seeded."""
import numpy as np, pandas as pd, json
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
g6=G.GetMorganGenerator(radius=3,fpSize=2048); g4=G.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda g,S:np.minimum(np.array([g.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S]),255).astype(np.uint8)
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; n=len(al)
P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
U=P[~P.CatalogID.isin(al.CatalogID)].reset_index(drop=True); print("pool sin etiqueta:",len(U),flush=True)
A=fx(g6,al.RDKit_SMILES); XU=np.vstack([fx(g6,U[sc].iloc[i:i+20000]) for i in range(0,len(U),20000)]); print("fingerprints listos",flush=True)
bv=[g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
L=lambda X:np.log1p(X.astype(np.float32)); ranks=[]
for seed in range(3):
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(A,y,g):
        m=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(L(A[tr]),y[tr])
        su=np.concatenate([m.decision_function(L(XU[i:i+20000])) for i in range(0,len(XU),20000)]); su.sort()
        sh=m.decision_function(L(A[te][y[te]==1]))
        ranks+=list(len(su)-np.searchsorted(su,sh,side='right')+1)
    print("semilla",seed,"listo",flush=True)
r=np.array(ranks).reshape(3,-1); res={"pool":len(U)}
for k in (500,1000,2000,5000,10000): res[f"hits_en_top{k}_de_39"]=[int((r[s]<=k).sum()) for s in range(3)]
res["rango_mediano"]=[int(np.median(r[s])) for s in range(3)]; print(json.dumps(res,indent=1)); json.dump(res,open("data/processed/gp/exp20_baseline_pool_rank.json","w"),indent=1)
