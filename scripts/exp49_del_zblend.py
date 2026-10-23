"""exp49: does adding the DEL logistic score (z-scored blend, small weight) hurt or help ranking within the SAME split? Repeated stratified 5-fold on the 6,580 AL labels (random folds: in the real
task unlabelled analogs of known hits are in the same split, so within-split CV is the relevant regime), logistic regression ECFP6 +/- w*DEL. Metrics: PR-AUC, hits in the top 1% (66). RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
RDLogger.DisableLog('rdApp.*'); g3=G.GetMorganGenerator(radius=3,fpSize=2048); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms,g:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; ms=[Chem.MolFromSmiles(s) for s in al.RDKit_SMILES]; X=cnt(ms,g3)
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
XD=cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES)+list(neg.SMILES)],g2)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(XD,np.r_[np.ones(len(pos)),np.zeros(len(neg))]); d=dm.decision_function(cnt(ms,g2)); d=(d-d.mean())/d.std()
W=[0,0.05,0.1,0.2,0.3,0.5]; R={w:[] for w in W}; K=66
for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=0).split(X,y):
    m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(X[tr],y[tr]); s=m.decision_function(X[te]); z=(s-s.mean())/s.std()
    for w in W:
        b=z+w*d[te]; R[w].append((average_precision_score(y[te],b),roc_auc_score(y[te],b),y[te][np.argsort(-b)[:max(1,K//5)]].sum(),y[te].sum()))
print("peso DEL (z) | PR-AUC | ROC-AUC | hits en el top 1% de cada fold (de los hits del fold)")
for w in W: a=np.array(R[w]).mean(0); print(f"  {w:4.2f}        | {a[0]:.3f}  | {a[1]:.3f}   | {a[2]:.2f} de {a[3]:.2f}")
