"""exp11: Active-learning candidate list. Pool = PGK2_Test_split (184,632). Steps (seeds fixed):
1) property pre-filter = full min-max window of the 39 real hits (exp10); already-labelled 6,580 cpds excluded;
2) score = rank-average of ASMS logistic + ASMS RF (trained on all 6,580 labels, 39 hits) with 0.1 weight DEL logistic (exp8);
3) top-50 chosen greedily from the top-500 with max 1 compound per Butina cluster (sim>=0.6).
Output: data/processed/gp/exp11_active_learning_scores.csv (CatalogID, Sel_50, Score) + exp11_top50_detail.csv"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
RDLogger.DisableLog('rdApp.*')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda S:np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S],dtype=np.float32)
al=pd.read_csv("data/raw/active_learning_data.csv"); P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet")
cols=['MW','logP','HBD','HBA','TPSA','aroR','rotB']
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as D
pr=lambda s:(lambda m:(Descriptors.MolWt(m),Crippen.MolLogP(m),D.CalcNumHBD(m),D.CalcNumHBA(m),D.CalcTPSA(m),D.CalcNumAromaticRings(m),D.CalcNumRotatableBonds(m)))(Chem.MolFromSmiles(s))
al[cols]=[pr(s) for s in al.RDKit_SMILES]; h=al[al.Label==1]
ok=np.all([P[c].between(h[c].min(),h[c].max()) for c in cols],axis=0)&~P.CatalogID.isin(al.CatalogID).values
C=P[ok].reset_index(drop=True); print("pool",len(P),"| pasa filtro y sin etiqueta:",len(C))
A=np.log1p(fx(al.RDKit_SMILES)); y=al.Label.values; Cx=fx(C.SMILES)
lr=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(A,y).predict_proba(np.log1p(Cx))[:,1]
rf=RandomForestClassifier(500,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(np.expm1(A),y).predict_proba(Cx)[:,1]
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
Xd=np.log1p(np.vstack([fx(pos.SMILES),fx(neg.SMILES)])); yd=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
de=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(Xd,yd).predict_proba(np.log1p(Cx))[:,1]
r=lambda v:rankdata(v)/len(v)
C['score']=0.9*(0.5*r(lr)+0.5*r(rf))+0.1*r(de); C['p_lr']=lr; C['p_rf']=rf; C['p_del']=de
top=C.sort_values('score',ascending=False).head(500).reset_index(drop=True)
bv=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.SMILES]; n=len(bv); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.4,isDistData=True)): g[list(c)]=k
top['cluster']=g; sel=top.drop_duplicates('cluster').head(50)
print("top500 en",len(set(g)),"clusters; seleccionados",len(sel),"| posiciones en el ranking:",sel.index.min(),"-",sel.index.max())
S=P[['CatalogID']].copy(); S['Score']=0.0; S=S.merge(C[['CatalogID','score']],on='CatalogID',how='left'); S['Score']=S.score.fillna(0.0); S['Sel_50']=S.CatalogID.isin(sel.CatalogID).astype(int)
S[['CatalogID','Sel_50','Score']].to_csv("data/processed/gp/exp11_active_learning_scores.csv",index=False)
sel[['CatalogID','SMILES','score','p_lr','p_rf','p_del','cluster']].to_csv("data/processed/gp/exp11_top50_detail.csv",index=False)
print(sel[['CatalogID','score','p_lr','p_rf','p_del']].round(3).head(10).to_string(index=False))
