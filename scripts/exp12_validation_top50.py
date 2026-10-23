"""exp12: generalization probe -- top-50 of the active-learning model over the VALIDATION pool (scaffold-disjoint from the AL/test data).
Same recipe as exp11 (property window of the 39 hits, 0.9*ASMS LR/RF rank-avg + 0.1*DEL LR, 1 per Butina cluster from the top-500).
Excludes every compound already present in submissions/Team_candeleroj_1..112.txt. Output: submissions/Team_candeleroj_113.txt (validation only)."""
import glob, numpy as np, pandas as pd
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, rdMolDescriptors as D
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
RDLogger.DisableLog('rdApp.*')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda S:np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S],dtype=np.float32)
cols=['MW','logP','HBD','HBA','TPSA','aroR','rotB']
pr=lambda s:(lambda m:(Descriptors.MolWt(m),Crippen.MolLogP(m),D.CalcNumHBD(m),D.CalcNumHBA(m),D.CalcTPSA(m),D.CalcNumAromaticRings(m),D.CalcNumRotatableBonds(m)))(Chem.MolFromSmiles(s))
al=pd.read_csv("data/raw/active_learning_data.csv"); al[cols]=[pr(s) for s in al.RDKit_SMILES]; h=al[al.Label==1]
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[c for c in v.columns if 'smi' in c.lower()][0]
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=int(f.split('_')[-1][:-4]) if f.split('_')[-1][:-4].isdigit() else 0
    if 1<=k<=112: tested|={l.strip() for l in open(f) if l.strip()}
v=v[~v.CatalogID.isin(tested)].reset_index(drop=True); print("validation pool",len(v),"(tras excluir",len(tested),"ya enviados)")
v[cols]=[pr(s) for s in v[sc]]
C=v[np.all([v[c].between(h[c].min(),h[c].max()) for c in cols],axis=0)].reset_index(drop=True); print("pasan filtro:",len(C))
A=np.log1p(fx(al.RDKit_SMILES)); y=al.Label.values; Cx=fx(C[sc])
lr=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(A,y).predict_proba(np.log1p(Cx))[:,1]
rf=RandomForestClassifier(500,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(np.expm1(A),y).predict_proba(Cx)[:,1]
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
Xd=np.log1p(np.vstack([fx(pos.SMILES),fx(neg.SMILES)])); yd=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
de=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(Xd,yd).predict_proba(np.log1p(Cx))[:,1]
r=lambda a:rankdata(a)/len(a); C['score']=0.9*(0.5*r(lr)+0.5*r(rf))+0.1*r(de)
top=C.sort_values('score',ascending=False).head(500).reset_index(drop=True)
bv=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sc]]; n=len(bv); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.4,isDistData=True)): g[list(c)]=k
sel=top.assign(cluster=g).drop_duplicates('cluster').head(50); assert len(sel)==50 and sel.CatalogID.is_unique
open("submissions/Team_candeleroj_113.txt","w").write("\n".join(sel.CatalogID)+"\n")
sel[['CatalogID',sc,'score','cluster']].to_csv("data/processed/gp/exp12_val_top50_detail.csv",index=False)
print(sel[['CatalogID',sc]].head(10).to_string(index=False))
