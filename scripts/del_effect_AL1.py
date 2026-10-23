"""del_effect_AL1: how much does the DEL term (weight 0.1) change the AL1 ranking? Compares z(AL model) with z(AL model) + 0.1*z(DEL) over the 178,052 pool compounds,
with the same DEL model as exp51 (logistic regression C=0.05 balanced, ECFP4 radius-2 count fingerprints log1p, positives count_PGK2>=5 and count_NTC==0, 50,000 negatives with count_PGK2<=1, seed 0).
Reads the exp48 output (data/processed/gp/exp48_scores_A6.csv, exp48_selection_A6.csv), so run it after exp48. The 50 selected are lifted to the same base Score in that file, so the comparison is printed
twice: as in the files and excluding the 50 selected. Reported: DEL z range, Spearman(AL, DEL), Spearman(blend, AL), overlap of the top 50/100/1,000/10,000 and rank shifts. RDKit venv. Prints only."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
ids=list(pd.read_csv("data/raw/active_learning_submission_example.csv").CatalogID)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES)+list(neg.SMILES)]),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
ms=[Chem.MolFromSmiles(smi[c]) for c in ids]; sd=np.concatenate([dm.decision_function(cnt(ms[i:i+20000])) for i in range(0,len(ms),20000)])
base=pd.read_csv("data/processed/gp/exp48_scores_A6.csv").set_index("CatalogID").Score.reindex(ids).values
sel=set(pd.read_csv("data/processed/gp/exp48_selection_A6.csv").CatalogID); z=lambda a:(a-a.mean())/a.std(); zb,zd=z(base),z(sd); bl=zb+0.1*zd
print(f"DEL z-score: min {zd.min():.2f}, max {zd.max():.2f}, p99 {np.percentile(zd,99):.2f} | Spearman(AL, DEL) {spearmanr(zb,zd)[0]:.3f}")
ex=np.array([c not in sel for c in ids])
for lab,m in (("as in the files (the 50 selected lifted in the base Score)",np.ones(len(ids),bool)),("excluding the 50 selected",ex)):
    a,b=zb[m],bl[m]; ra,rb=rankdata(-a,method="min"),rankdata(-b,method="min"); top=lambda v,k:set(np.argsort(-v)[:k])
    print(f"{lab}: Spearman(blend, AL) {spearmanr(a,b)[0]:.4f} | overlap top 50/100/1000/10000: {[len(top(a,k)&top(b,k)) for k in (50,100,1000,10000)]}")
    for k in (50,100,1000): s=np.abs(rb[np.argsort(-a)[:k]]-ra[np.argsort(-a)[:k]]); print(f"   top {k} of the AL model: median |rank shift| {np.median(s):.0f}, mean {s.mean():.1f}")
