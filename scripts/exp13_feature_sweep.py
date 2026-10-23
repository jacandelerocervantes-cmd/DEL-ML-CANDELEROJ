"""exp13: feature-set / model sweep on the 6,580 active-learning labels. Strict cluster CV (Butina sim>=0.3), StratifiedGroupKFold, 5 seeds.
Metrics pooled OOF: ROC-AUC, PR-AUC, hits in top 1% (66). Logistic regression C=0.05 balanced unless stated; RF 300 trees."""
import numpy as np, pandas as pd, json
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, MACCSkeys
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
RDLogger.DisableLog('rdApp.*')
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; ms=[Chem.MolFromSmiles(s) for s in al.RDKit_SMILES]; n=len(ms)
cnt=lambda g:np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32)
F={'ecfp4_cnt':cnt(G.GetMorganGenerator(radius=2,fpSize=2048)),'ecfp6_cnt':cnt(G.GetMorganGenerator(radius=3,fpSize=2048)),'ecfp2_cnt':cnt(G.GetMorganGenerator(radius=1,fpSize=2048)),
 'fcfp4_cnt':cnt(G.GetMorganGenerator(radius=2,fpSize=2048,atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen())),
 'atompair_cnt':cnt(G.GetAtomPairGenerator(fpSize=2048)),'torsion_cnt':cnt(G.GetTopologicalTorsionGenerator(fpSize=2048)),
 'rdkit_path':cnt(G.GetRDKitFPGenerator(maxPath=5,fpSize=2048)),'maccs':np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in ms],dtype=np.float32)}
F['ecfp4+fcfp4+ap']=np.hstack([F['ecfp4_cnt'],F['fcfp4_cnt'],F['atompair_cnt']])
bv=[G.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(m) for m in ms]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
K=66
def cv(X,mk,log=False):
    out=[]
    for seed in range(5):
        oof=np.zeros(n)
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,g):
            Xa=np.log1p(X) if log else X; oof[te]=mk().fit(Xa[tr],y[tr]).predict_proba(Xa[te])[:,1]
        out.append(oof)
    return out
def met(oofs): a=np.array([[roc_auc_score(y,o),average_precision_score(y,o),y[np.argsort(-o)[:K]].sum()] for o in oofs]).mean(0); return dict(auc=round(a[0],3),prauc=round(a[1],3),hits1pct=round(a[2],1))
LR=lambda:LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000); RF=lambda:RandomForestClassifier(300,class_weight='balanced_subsample',n_jobs=-1,random_state=0)
res={}; oofs={}
for fn,X in F.items():
    oofs[(fn,'lr')]=cv(X,LR,True); res[f"{fn}|lr"]=met(oofs[(fn,'lr')]); print(fn,'LR',res[f"{fn}|lr"],flush=True)
    if fn in('ecfp4_cnt','fcfp4_cnt','atompair_cnt','ecfp6_cnt'): oofs[(fn,'rf')]=cv(X,RF); res[f"{fn}|rf"]=met(oofs[(fn,'rf')]); print(fn,'RF',res[f"{fn}|rf"],flush=True)
def ens(keys): return [np.mean([rankdata(oofs[k][s]) for k in keys],axis=0) for s in range(5)]
for nm,keys in {'ens ecfp4 LR+RF':[('ecfp4_cnt','lr'),('ecfp4_cnt','rf')],'ens ecfp4+fcfp4+atompair LR':[('ecfp4_cnt','lr'),('fcfp4_cnt','lr'),('atompair_cnt','lr')],
 'ens 4 repr LR+RF':[('ecfp4_cnt','lr'),('ecfp4_cnt','rf'),('fcfp4_cnt','lr'),('fcfp4_cnt','rf'),('atompair_cnt','lr'),('atompair_cnt','rf')]}.items():
    res[nm]=met(ens(keys)); print(nm,res[nm],flush=True)
json.dump(res,open("data/processed/gp/exp13_feature_sweep.json","w"),indent=1)
