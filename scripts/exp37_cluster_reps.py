"""exp37: one representative hit (Butina centroid) per validation-hit cluster at Tanimoto>=0.35 (the threshold calibrated in exp36) + confirmed negatives -> submissions/Team_candeleroj_154.txt.
Expected if the clustering matches Synapse's series: N_hits = N_clusters = number of clusters. Run with the RDKit venv."""
import os, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); NEG=sorted(o[o.confirmed_label==0].CatalogID)
gen=G.GetMorganGenerator(radius=2,fpSize=2048); f=[gen.GetFingerprint(Chem.MolFromSmiles(vs[x])) for x in VH]; n=len(f); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(f[i],f[:i])]
cl=Butina.ClusterData(d,n,1-0.35,isDistData=True); rows=[]
for k,c in enumerate(sorted(cl,key=len,reverse=True)):
    for j,i in enumerate(c): rows.append(dict(cluster=k+1,tam=len(c),representante=(j==0),CatalogID=VH[i],smiles=vs[VH[i]]))
T=pd.DataFrame(rows); T.to_csv("data/processed/gp/exp37_validation_hit_clusters.csv",index=False)
reps=list(T[T.representante].CatalogID); print("clusters:",len(cl),"| representantes:",len(reps))
rng=np.random.default_rng(154); L=reps+list(rng.choice(NEG,50-len(reps),replace=False)); assert len(set(L))==50
p="submissions/Team_candeleroj_154.txt"; assert not os.path.exists(p); open(p,"w").write("\n".join(L)+"\n")
print("tamano de cada cluster:",sorted(T.groupby('cluster').tam.first().tolist(),reverse=True))
pd.set_option('display.width',200); print(T[T.representante][["cluster","tam","CatalogID","smiles"]].to_string(index=False,max_colwidth=70))
