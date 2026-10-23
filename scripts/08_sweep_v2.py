"""Step 8: DEL v2 (positives: count_PGK2>=5, NTC==0, historic_hits<=1; CV AUC 0.786 on AL labels vs 0.770) scores the pool; sweeps pool% x DEL weight x diversity.
Reports overlap with the earlier AL1/AL2 files (comparison only), the DEL ablation (picks changed when DEL removed) and the share of labelled hits kept inside the DEL pool. Output results/pool_scores_v2.csv, results/sweep_v2.csv."""
import itertools, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
F=lambda s:np.vstack([cnt([Chem.MolFromSmiles(x) for x in s[i:i+20000]]) for i in range(0,len(s),20000)])
U=pd.read_csv("results/pool_scores.csv"); al=pd.read_csv("data/raw/active_learning_data.csv")
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)&(t.historic_hits<=1)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(F(list(pos.SMILES)+list(neg.SMILES)),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
U["del2"]=np.concatenate([dm.decision_function(F(list(U.SMILES[i:i+20000]))) for i in range(0,len(U),20000)]); U.to_csv("results/pool_scores_v2.csv",index=False)
dl=dm.decision_function(F(list(al.RDKit_SMILES))); hit=al.Label.values==1
z=lambda a:(a-a.mean())/a.std(); U["zdel"]=z(U.del2); U["zal"]=z(U.al)
d="active_learning_candidates/"; T1=set(pd.read_csv(d+"Team_candeleroj_AL1_20261005.csv").query("Sel_50==1").CatalogID); T2=set(pd.read_csv(d+"Team_candeleroj_AL2_20261005.csv").query("Sel_50==1").CatalogID)
fps={}
def pick(df,score,div,cand=6000):
    o=[];f_=[]
    for r in df.assign(_s=score).sort_values("_s",ascending=False).head(cand).itertuples():
        if r.CatalogID not in fps: fps[r.CatalogID]=g2.GetFingerprint(Chem.MolFromSmiles(r.SMILES))
        f=fps[r.CatalogID]
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=div: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=50: break
    return o
rows=[]; base={dv:pick(U,U.zal,dv) for dv in (0.5,0.6,0.7)}
for pct,wd,dv in itertools.product((20,35,50,100),(0.25,0.5,0.75,1.0),(0.6,0.7)):
    pool=U[U.del2.rank(ascending=False)<=pct/100*len(U)]; l1=pick(pool,wd*pool.zdel+pool.zal,dv); keep=(dl>=np.quantile(dl,1-pct/100))[hit].mean()
    rows.append(dict(pct=pct,w_del=wd,div=dv,ov_AL1=len(set(l1)&T1),ov_AL2=len(set(l1)&T2),ablation=50-len(set(l1)&set(base[dv])),hits_kept_in_pool=round(float(keep),2))); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/sweep_v2.csv",index=False)
