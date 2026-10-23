import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*'); g=G.GetMorganGenerator(radius=2,fpSize=2048)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
D="submissions/active_learning_candidates/"
def top(f): d=pd.read_csv(D+f); return list(d[d.Sel_50==1].CatalogID) if "Sel_50" in d else list(d.CatalogID[:50])
fp=lambda ids:[g.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in ids]
A1=top("Team_candeleroj_AL1_20261005.csv"); fa=fp(A1)
def groups(f):
    n=len(f); d=[]
    for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(f[i],f[:i])]
    return len(Butina.ClusterData(d,n,1-0.32,isDistData=True))
print("AL1: grupos",groups(fa))
for name in ["Team_candeleroj_AL2_20261005.csv","Team_candeleroj_AL2_DELforward_20261005.csv","Team_candeleroj_AL2_newgroups_20261005.csv"]:
    ids=top(name); f=fp(ids)
    near=[max(DataStructs.BulkTanimotoSimilarity(x,fa))>=0.32 for x in f]
    new=[i for i,b in enumerate(near) if not b]
    print(name,"| ids:",len(ids),"| grupos propios:",groups(f),"| en grupo de AL1:",sum(near),"| fuera de los grupos de AL1:",len(new),"| grupos nuevos:",groups([f[i] for i in new]) if len(new)>1 else len(new))
