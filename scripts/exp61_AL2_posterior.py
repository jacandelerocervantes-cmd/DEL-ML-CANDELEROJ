import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*'); g=G.GetMorganGenerator(radius=2,fpSize=2048)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
S=pd.read_csv("data/processed/gp/exp60_selection_AL2_newgroups.csv"); ids=list(S.CatalogID); n=50
fp=[g.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in ids]; d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(d,n,1-0.32,isDistData=True)): cl[list(c)]=k
C=cl.max()+1; new=(S.origen=="grupo_nuevo").values
print("grupos:",C,"| tamanos:",sorted(np.bincount(cl).tolist(),reverse=True),"| grupos con nuevos:",len(set(cl[new])),"| grupos con alta_prob:",len(set(cl[~new])),"| compartidos:",len(set(cl[new])&set(cl[~new])))
rng=np.random.default_rng(60)
for r in (1.0,0.5,0.25):   # prob relativa de un 'nuevo' vs 'alta prob'
    p0=np.where(new,r,1.0); p=p0*12/p0.sum(); acc=[]; 
    for _ in range(16):
        X=rng.random((250000,n))<p; X=X[X.sum(1)==12]
        M=np.stack([X[:,cl==k].any(1) for k in range(C)],1); acc.append(X[M.sum(1)==5])
    A=np.vstack(acc); 
    print(f"r={r}: muestras 12 hits y 5 series: {len(A)} | hits esperados entre nuevos {A[:,new].sum(1).mean():.1f} de 18, entre alta prob {A[:,~new].sum(1).mean():.1f} de 32" if len(A) else f"r={r}: sin muestras")
