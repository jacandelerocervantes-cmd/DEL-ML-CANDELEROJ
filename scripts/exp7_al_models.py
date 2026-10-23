"""exp7: models trained on the real ASMS-labelled active-learning set (6,580 cpds, 39 hits), evaluated with
cluster-disjoint CV (Butina 0.6 Tanimoto, StratifiedGroupKFold, 5 seeds). sklearn only (LightGBM is blocked by Windows app control here).
Metrics are pooled out-of-fold: ROC-AUC, PR-AUC, hits in top 1% (66 cpds), enrichment factor @1%. Seeds fixed."""
import numpy as np, pandas as pd, json
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
RDLogger.DisableLog('rdApp.*')
d=pd.read_csv("data/raw/active_learning_data.csv"); y=d.Label.values; ms=[Chem.MolFromSmiles(s) for s in d.RDKit_SMILES]
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
bv=[gen.GetFingerprint(m) for m in ms]; X=np.array([gen.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32)
n=len(ms); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
cl=Butina.ClusterData(dist,n,0.4,isDistData=True)   # distance cutoff 0.4 -> sim>=0.6 same cluster
g=np.zeros(n,int)
for k,c in enumerate(cl): g[list(c)]=k
print("clusters",len(cl),"hits en",len(set(g[y==1])),"clusters")
names=['rdkit_desc']
D=np.array([[f(m) for _,f in Descriptors.descList] for m in ms],dtype=np.float64); D=np.nan_to_num(np.clip(D,-1e6,1e6))
def maxsim(tr,te):
    H=[bv[i] for i in tr if y[i]==1]; return np.array([max(DataStructs.BulkTanimotoSimilarity(bv[i],H)) for i in te])
M={
 'random':lambda tr,te,r:r.random(len(te)),
 'maxsim_to_train_hits':lambda tr,te,r:maxsim(tr,te),
 'ecfp4_logreg':lambda tr,te,r:LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(np.log1p(X[tr]),y[tr]).predict_proba(np.log1p(X[te]))[:,1],
 'ecfp4_rf':lambda tr,te,r:RandomForestClassifier(500,min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1],
 'ecfp4_hgb':lambda tr,te,r:HistGradientBoostingClassifier(max_depth=4,learning_rate=0.05,max_iter=200,class_weight='balanced',random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1],
 'desc_rf':lambda tr,te,r:RandomForestClassifier(500,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(D[tr],y[tr]).predict_proba(D[te])[:,1],
}
K=int(round(0.01*n)); res={}
for name,fn in M.items():
    rows=[]
    for seed in range(5):
        oof=np.zeros(n); r=np.random.default_rng(seed)
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,g): oof[te]=fn(tr,te,r)
        top=np.argsort(-oof)[:K]; rows.append((roc_auc_score(y,oof),average_precision_score(y,oof),y[top].sum()))
    a=np.array(rows); res[name]=dict(auc=a[:,0].mean().round(3),auc_sd=a[:,0].std().round(3),prauc=a[:,1].mean().round(3),hits_top1pct=a[:,2].mean().round(1),EF1=round(a[:,2].mean()/(y.sum()*K/n),1))
    print(name,res[name],flush=True)
json.dump(res,open("data/processed/gp/exp7_al_models.json","w"),indent=1)
