"""exp16: continuous score for ALL 184,632 test-pool compounds (no hard 0s) for the full-library PR-AUC/ROC-AUC criterion.
score = rank-avg(ASMS LR on ECFP6 counts, ASMS RF on ECFP4 counts) [0.9] + DEL LR [0.1], multiplied by a SOFT property penalty
(1 / (1 + total normalized distance outside the min-max window of the 39 hits)). Labelled AL compounds get their true label as score (1/0 extremes, flagged).
Selection: top-500 by score among unlabeled, 1 per Butina cluster (sim>=0.6), Sel_50. Output: data/processed/gp/exp16_pool_scores.csv (CatalogID, Sel_50, Score)."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
RDLogger.DisableLog('rdApp.*')
g4=G.GetMorganGenerator(radius=2,fpSize=2048); g6=G.GetMorganGenerator(radius=3,fpSize=2048)
fx=lambda g,S:np.array([g.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S],dtype=np.float32)
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values
P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
cols=['MW','logP','HBD','HBA','TPSA','aroR','rotB']
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as D
pr=lambda s:(lambda m:(Descriptors.MolWt(m),Crippen.MolLogP(m),D.CalcNumHBD(m),D.CalcNumHBA(m),D.CalcTPSA(m),D.CalcNumAromaticRings(m),D.CalcNumRotatableBonds(m)))(Chem.MolFromSmiles(s))
hp=pd.DataFrame([pr(s) for s in al.RDKit_SMILES[y==1]],columns=cols)
lr=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(np.log1p(fx(g6,al.RDKit_SMILES)),y)
rf=RandomForestClassifier(500,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(fx(g4,al.RDKit_SMILES),y)
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
de=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(np.log1p(np.vstack([fx(g4,pos.SMILES),fx(g4,neg.SMILES)])),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
a=[];b=[];c=[]
for i in range(0,len(P),20000):
    S=P[sc].iloc[i:i+20000]; X4=fx(g4,S); X6=fx(g6,S)
    a.append(lr.predict_proba(np.log1p(X6))[:,1]); b.append(rf.predict_proba(X4)[:,1]); c.append(de.predict_proba(np.log1p(X4))[:,1]); print(i,flush=True)
a,b,c=map(np.concatenate,(a,b,c)); r=lambda v:rankdata(v)/len(v)
base=0.9*(0.5*r(a)+0.5*r(b))+0.1*r(c)
dist=sum(np.maximum(hp[k].min()-P[k],0)/(hp[k].max()-hp[k].min()+1e-9)+np.maximum(P[k]-hp[k].max(),0)/(hp[k].max()-hp[k].min()+1e-9) for k in cols)
P['Score']=base/(1+dist); lab=dict(zip(al.CatalogID,al.Label)); P['lab']=P.CatalogID.map(lab)
print("sin empates exactos:",P.Score.nunique(),"valores únicos de",len(P),"| compuestos fuera de ventana con score>0:",int((dist>0).sum()))
U=P[P.lab.isna()].sort_values('Score',ascending=False).head(500).reset_index(drop=True)
bv=[g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in U[sc]]; n=len(bv); dd=[]
for i in range(1,n): dd+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
gg=np.zeros(n,int)
for k,cc in enumerate(Butina.ClusterData(dd,n,0.4,isDistData=True)): gg[list(cc)]=k
sel=U.assign(cluster=gg).drop_duplicates('cluster').head(50); assert len(sel)==50
P['Sel_50']=P.CatalogID.isin(sel.CatalogID).astype(int)
P.loc[P.lab.notna(),'Score']=P.loc[P.lab.notna(),'Score']*0.0+P.loc[P.lab.notna(),'lab']*1e-9   # labelled: tiny, they are known
P[['CatalogID','Sel_50','Score']].to_csv("data/processed/gp/exp16_pool_scores.csv",index=False)
sel[['CatalogID',sc,'Score','cluster']].to_csv("data/processed/gp/exp16_top50_detail.csv",index=False); print(sel[['CatalogID',sc]].head(8).to_string(index=False))
