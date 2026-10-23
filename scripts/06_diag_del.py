"""Step 6: diagnostic of the DEL-ML model: how well does the DEL score rank the 6,580 AL-labelled compounds (39 hits; labels NOT used in DEL training)? Compares positive definitions and models. Output printed + results/del_variants.csv."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
F=lambda s:np.vstack([cnt([Chem.MolFromSmiles(x) for x in s[i:i+20000]]) for i in range(0,len(s),20000)])
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; XL=F(list(al.RDKit_SMILES))
t=pd.read_pickle("data/processed/del_sel.pkl"); cand=t[(t.count_PGK2>=3)&(t.count_NTC==0)].reset_index(drop=True); negs=t[t.count_PGK2<=1].sample(100000,random_state=0).reset_index(drop=True)
XC=F(list(cand.SMILES)); XN=F(list(negs.SMILES)); rows=[]
def ev(name,s):
    top=np.argsort(-s)[:int(len(s)*0.05)]; r=dict(variant=name,AUC=round(roc_auc_score(y,s),3),PRAUC=round(average_precision_score(y,s),3),top5pct=int(y[top].sum())); rows.append(r); print(r,flush=True)
ratio=(cand.count_PGK2_with_inhibitor/cand.count_PGK2)
defs={"c>=5":cand.count_PGK2>=5,"c>=3":cand.count_PGK2>=3,"c>=5 competed(inh<=0.5c)":(cand.count_PGK2>=5)&(ratio<=0.5),"c>=3 competed":(cand.count_PGK2>=3)&(ratio<=0.5),
 "c>=5 historic<=1":(cand.count_PGK2>=5)&(cand.historic_hits<=1),"c>=5 z-high":(cand.count_PGK2>=5)&(cand.zscore_PGK2>=cand.zscore_PGK2.quantile(.5))}
for nm,mk in defs.items():
    m=mk.values
    if m.sum()<100: print(nm,"too few",m.sum()); continue
    X=np.vstack([XC[m],XN[:50000]]); yy=np.r_[np.ones(m.sum()),np.zeros(50000)]
    ev(f"LR C=.05 {nm} n={m.sum()}",LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(X,yy).decision_function(XL))
    if nm in("c>=5","c>=3","c>=5 competed(inh<=0.5c)"):
        X=np.vstack([XC[m],XN]); yy=np.r_[np.ones(m.sum()),np.zeros(len(XN))]
        g=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,scale_pos_weight=len(XN)/m.sum(),verbose=-1,n_jobs=-1).fit(X,yy); ev(f"LGBM {nm} n={m.sum()}",g.predict_proba(XL)[:,1])
pd.DataFrame(rows).to_csv("results/del_variants.csv",index=False)
