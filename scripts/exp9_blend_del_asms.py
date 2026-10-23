"""exp9: rank-blend of ASMS logistic (cluster-CV out-of-fold) with the DEL-trained logistic score (exp8; uses no ASMS labels).
Butina cutoff 0.7 (strict), StratifiedGroupKFold, 5 seeds."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
fx=lambda S:np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in S],dtype=np.float32)
t=pd.read_pickle("data/processed/del_sel.pkl")
pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
X=np.log1p(np.vstack([fx(pos.SMILES),fx(neg.SMILES)])); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
al=pd.read_csv("data/raw/active_learning_data.csv"); A=np.log1p(fx(al.RDKit_SMILES)); ay=al.Label.values; n=len(al)
del_s=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(X,y).predict_proba(A)[:,1]
bv=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
K=66; W=[0,0.1,0.2,0.3,0.5,1.0]; R={w:[] for w in W}
for seed in range(5):
    oof=np.zeros(n)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(A,ay,g):
        oof[te]=LogisticRegression(C=0.05,class_weight='balanced',max_iter=2000).fit(A[tr],ay[tr]).predict_proba(A[te])[:,1]
    ra,rd=rankdata(oof)/n,rankdata(del_s)/n
    for w in W:
        s=(1-w)*ra+w*rd; R[w].append((roc_auc_score(ay,s),average_precision_score(ay,s),ay[np.argsort(-s)[:K]].sum()))
for w in W: a=np.array(R[w]).mean(0); print(f"peso DEL {w:.1f}: AUC {a[0]:.3f}  PR-AUC {a[1]:.3f}  hits top1% {a[2]:.1f}/39")
