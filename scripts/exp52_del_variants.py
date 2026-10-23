"""exp52: how much real signal do DEL-trained models carry on the 6,580 labelled TEST compounds (39 hits)? Variants of the DEL 'hit' definition, each a logistic regression on ECFP4 counts
(60k balanced sample), evaluated on the AL labels without training on them: ROC-AUC, PR-AUC, hits in the top 1% (66). The organizers require DEL to be a KEY component, so we need the DEL model that really helps. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda smis:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smis],dtype=np.float32))
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; XA=cnt(al.RDKit_SMILES); K=66
t=pd.read_pickle("data/processed/del_sel.pkl"); t=t[t.SMILES.notna()&(t.SMILES.str.len()>0)]
neg_pool=t[t.count_PGK2<=1]; neg=neg_pool.sample(50000,random_state=0)
inh_drop=lambda d:d.count_PGK2_with_inhibitor<=d.count_PGK2/4
V={"A: count>=5 y NTC=0 (actual)":t[(t.count_PGK2>=5)&(t.count_NTC==0)],
   "B: count>=3 y NTC=0":t[(t.count_PGK2>=3)&(t.count_NTC==0)],
   "C: count>=10 y NTC=0":t[(t.count_PGK2>=10)&(t.count_NTC==0)],
   "D: A y compite con el inhibidor (sitio ATP)":(lambda a:a[inh_drop(a)])(t[(t.count_PGK2>=5)&(t.count_NTC==0)]),
   "E: A y NO compite (otro sitio)":(lambda a:a[~inh_drop(a)])(t[(t.count_PGK2>=5)&(t.count_NTC==0)]),
   "F: historic_hits>=1 y count>=5":t[(t.historic_hits>=1)&(t.count_PGK2>=5)&(t.count_NTC==0)]}
print(f"{'variante':46s} positivos | ROC-AUC | PR-AUC | hits en el top 1% (de 39)")
res={}
for name,pos in V.items():
    if len(pos)<50: print(name,"muy pocos positivos:",len(pos)); continue
    pos=pos.sample(min(len(pos),20000),random_state=0); X=np.vstack([cnt(pos.SMILES),cnt(neg.SMILES)]); yy=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(X,yy); s=m.decision_function(XA); res[name]=s
    print(f"{name:46s} {len(pos):8d} |  {roc_auc_score(y,s):.3f}  | {average_precision_score(y,s):.3f}  | {int(y[np.argsort(-s)[:K]].sum())}",flush=True)
pd.DataFrame(res).assign(Label=y,CatalogID=al.CatalogID).to_csv("data/processed/gp/exp52_del_variants_scores.csv",index=False)
