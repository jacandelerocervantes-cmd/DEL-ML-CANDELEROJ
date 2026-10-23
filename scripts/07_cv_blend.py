"""Step 7: scaffold-grouped 5-fold CV on the 6,580 AL labels. AL model: LR on Morgan r3 counts trained on the other folds. DEL model: LR (positives count_PGK2>=5, NTC==0, historic_hits<=1) trained on DEL only.
Evaluates z(DEL)*wd + z(AL)*wa blends (OOF pooled): AUC, PR-AUC, hits in top 5%. Also 'stack' = AL LR with the DEL score as an extra feature. Output results/cv_blend.csv and results/del_historic_scores_labelled.npy."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
F=lambda s,g:np.vstack([cnt([Chem.MolFromSmiles(x) for x in s[i:i+20000]],g) for i in range(0,len(s),20000)])
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; sm=list(al.RDKit_SMILES); X2=F(sm,g2); X3=F(sm,g3)
grp=pd.factorize([Chem.MolToSmiles(MS.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in sm])[0]
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)&(t.historic_hits<=1)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(F(list(pos.SMILES)+list(neg.SMILES),g2),np.r_[np.ones(len(pos)),np.zeros(len(neg))]); d=dm.decision_function(X2)
z=lambda a:(a-a.mean())/a.std(); oof=np.zeros(len(y)); oofs=np.zeros(len(y)); folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X3,y,grp))
for tr,te in folds:
    oof[te]=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(X3[tr],y[tr]).decision_function(X3[te])
    Xs=np.c_[X3,np.repeat(z(d)[:,None],10,1)]; oofs[te]=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(Xs[tr],y[tr]).decision_function(Xs[te])
def ev(n,s): 
    top=np.argsort(-s)[:int(.05*len(s))]; top1=np.argsort(-s)[:66]; r=dict(model=n,AUC=round(roc_auc_score(y,s),3),PRAUC=round(average_precision_score(y,s),3),top5=int(y[top].sum()),top1=int(y[top1].sum())); print(r,flush=True); return r
rows=[ev("DEL only",d),ev("AL only (oof)",oof)]
for wd in (0.25,0.5,1,2): rows.append(ev(f"z(DEL)*{wd}+z(AL)",wd*z(d)+z(oof)))
rows.append(ev("stack LR (DEL as feature)",oofs)); pd.DataFrame(rows).to_csv("results/cv_blend.csv",index=False)
