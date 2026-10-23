"""Step 4: design sweep for step 2. For each (DEL_POOL_PCT, W_AL, DIVERSITY) it builds list 1 (DEL-first) and reports (a) overlap with the earlier AL1/AL2 files (comparison only, never an input to scoring)
and (b) the DEL ablation (list 1 changed when DEL is removed). Pick a design where the ablation is large AND the overlap is acceptable. Output: results/sweep.csv."""
import itertools, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
U=pd.read_csv("results/pool_scores.csv"); z=lambda a:(a-a.mean())/a.std(); U["zdel"]=z(U["del"]); U["zal"]=z(U.al)
d="active_learning_candidates/"; T1=set(pd.read_csv(d+"Team_candeleroj_AL1_20261005.csv").query("Sel_50==1").CatalogID); T2=set(pd.read_csv(d+"Team_candeleroj_AL2_20261005.csv").query("Sel_50==1").CatalogID)
fps={}
def fp(r):
    if r.CatalogID not in fps: fps[r.CatalogID]=g2.GetFingerprint(Chem.MolFromSmiles(r.SMILES))
    return fps[r.CatalogID]
def pick(df,score,div,cand=6000,exclude=(),n=50):
    o=[];f_=[]
    for r in df[~df.CatalogID.isin(exclude)].assign(_s=score).sort_values("_s",ascending=False).head(cand).itertuples():
        f=fp(r)
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=div: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=n: break
    return o
rows=[]
for pct,w,div in itertools.product((5,10,20,30),(0.5,1,2,4),(0.5,0.6,0.7)):
    pool=U[U["del"].rank(ascending=False)<=pct/100*len(U)]; l1=pick(pool,pool.zdel+w*pool.zal,div); r1=pick(U,U.zal,div)
    rows.append(dict(pct=pct,w_al=w,div=div,ov_AL1=len(set(l1)&T1),ov_AL2=len(set(l1)&T2),ablation_changed=50-len(set(l1)&set(r1)))); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/sweep.csv",index=False)
