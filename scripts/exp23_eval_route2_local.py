"""exp23: evaluate the Kaggle route-2 embeddings locally (the notebook trained the DEL network and wrote the embeddings, but its evaluation stage failed).
Embeddings: kaggle output emb_al.npy (6,580 x 512) and emb_pool.npy (178,052 x 512, same row order as the unlabelled pool U used in exp21b).
Protocol identical to exp21b/route2_multitask.py: cluster-held-out CV (Butina sim>=0.3, StratifiedGroupKFold, seeds 0-2), 100k background negatives,
logistic regression on standardized features, held-out hits ranked among the unlabelled pool minus the extras; cutoffs scaled to 178,052. Seeded."""
import numpy as np, pandas as pd, json
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog('rdApp.*')
S="data/processed/kaggle_route2"
EA=np.load(S+"/k_out/emb_al.npy"); EU=np.load(S+"/k_out/emb_pool.npy"); XU=np.load(S+"/XU.npy")
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; n=len(al); nu=len(EU); assert EU.shape[0]==XU.shape[0]==nu and EA.shape[0]==n, (EU.shape,XU.shape,EA.shape)
g6=G.GetMorganGenerator(radius=3,fpSize=2048); g4=G.GetMorganGenerator(radius=2,fpSize=2048)
A6=np.minimum(np.array([g6.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]),255).astype(np.uint8)
bv=[g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
L6=lambda X:np.log1p(X.astype(np.float32)); EA=EA.astype(np.float32); EU=EU.astype(np.float32)
variants={"ecfp6 (linea base)":(L6(A6),L6(XU)),"emb DEL":(EA,EU),"ecfp6 + emb DEL":(np.hstack([L6(A6),EA]),np.hstack([L6(XU),EU]))}
res={k:[] for k in variants}
for seed in range(3):
    rng=np.random.default_rng(seed); extra=rng.choice(nu,100000,replace=False); keep=np.ones(nu,bool); keep[extra]=False
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(A6,y,g):
        for name,(FA,FU) in variants.items():
            mu=np.vstack([FA[tr],FU[extra]]).mean(0); sd=np.vstack([FA[tr],FU[extra]]).std(0)+1e-6
            X=(np.vstack([FA[tr],FU[extra]])-mu)/sd; yy=np.r_[y[tr],np.zeros(len(extra))]
            m=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(X,yy)
            su=np.sort(np.concatenate([m.decision_function((FU[keep][i:i+20000]-mu)/sd) for i in range(0,keep.sum(),20000)]))
            sh=m.decision_function((FA[te][y[te]==1]-mu)/sd); res[name]+=list(len(su)-np.searchsorted(su,sh,side='right')+1)
    print("semilla",seed,"lista",flush=True)
out={}
for name,r in res.items():
    r=np.array(r).reshape(3,-1); out[name]={f"top{k}_equiv178k":[int((r[s]<=k*(nu-100000)/178052).sum()) for s in range(3)] for k in (500,1000,2000,5000)}; out[name]["rango_mediano"]=[int(np.median(r[s])) for s in range(3)]
print(json.dumps(out,indent=1,ensure_ascii=False)); json.dump(out,open("data/processed/gp/exp23_route2_local_eval.json","w"),indent=1,ensure_ascii=False)
