"""Step 1: scores the 178,052-compound pool with (a) the DEL-ML model and (b) the Active Learning (AL) model. Run from the repository root after 00_prepare_data.py.
DEL-ML: L2 logistic regression on Morgan radius-2 count fingerprints (2,048 bits, log1p), trained on PGK2_selection.parquet (positives count_PGK2 >= 5 and count_NTC == 0; 50,000 random negatives with count_PGK2 <= 1).
AL model: L2 logistic regression on Morgan radius-3 count fingerprints, trained on the 6,580 released labels plus 50,000 background negatives per half, cross-fitted in two halves of the pool.
Output: results/pool_scores.csv (CatalogID, SMILES, del, al). RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
F=lambda smis,g:np.vstack([cnt([Chem.MolFromSmiles(s) for s in smis[i:i+20000]],g) for i in range(0,len(smis),20000)])
LR=lambda:LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000)
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
U=P[~P.CatalogID.isin(set(al.CatalogID))].reset_index(drop=True).rename(columns={sc:"SMILES"})[["CatalogID","SMILES"]]
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LR().fit(F(list(pos.SMILES)+list(neg.SMILES),g2),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
U["del"]=np.concatenate([dm.decision_function(F(list(U.SMILES[i:i+20000]),g2)) for i in range(0,len(U),20000)])
XU=F(list(U.SMILES),g3); XL=F(list(al.RDKit_SMILES),g3); rng=np.random.default_rng(48); half=rng.permutation(len(XU)); H1,H2=half[:len(XU)//2],half[len(XU)//2:]; s=np.zeros(len(XU),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False); s[te]=LR().fit(np.vstack([XL,XU[ex]]),np.r_[y,np.zeros(len(ex))]).decision_function(XU[te])
U["al"]=s; U.to_csv("results/pool_scores.csv",index=False); print("scored",len(U),"compounds; Spearman DEL-AL:",round(float(U["del"].rank().corr(U.al.rank())),3))
