"""exp8: DEL -> ASMS transfer. Train on PGK2 DEL (positives: count_PGK2>=5 & count_NTC==0; negatives: random count_PGK2<=1),
score the 6,580 active-learning compounds (39 real ASMS hits). Seeds fixed. No ASMS label is used for training."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda S:np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S],dtype=np.float32)
t=pd.read_pickle("data/processed/del_sel.pkl")
pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
X=np.vstack([fx(pos.SMILES),fx(neg.SMILES)]); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
al=pd.read_csv("data/raw/active_learning_data.csv"); AX=fx(al.RDKit_SMILES); ay=al.Label.values; K=66
def ev(n,s):
    top=np.argsort(-s)[:K]; print(f"{n}: AUC {roc_auc_score(ay,s):.3f}  PR-AUC {average_precision_score(ay,s):.3f}  hits top1% {ay[top].sum()}/39  EF1 {ay[top].sum()/(39*K/len(ay)):.1f}",flush=True)
ev("DEL logística",LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(np.log1p(X),y).predict_proba(np.log1p(AX))[:,1])
ev("DEL RF",RandomForestClassifier(300,class_weight='balanced_subsample',n_jobs=-1,random_state=0).fit(X,y).predict_proba(AX)[:,1])
