"""Step 9: alternative DEL-involving formulations (rank-sum, min-rank/consensus, DEL-gate + AL) vs overlap with earlier AL1 and DEL ablation. Comparison only; the earlier list is never an input."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
U=pd.read_csv("results/pool_scores_v2.csv"); n=len(U); rd=U.del2.rank(pct=True); ra=U.al.rank(pct=True)
T1=set(pd.read_csv("active_learning_candidates/Team_candeleroj_AL1_20261005.csv").query("Sel_50==1").CatalogID)
print("targets: DEL pct (v2) min/median/max",U[U.CatalogID.isin(T1)].del2.rank(pct=True).describe()[["min","50%","max"]].round(3).tolist() if False else rd[U.CatalogID.isin(T1)].describe()[["min","50%","max"]].round(3).tolist())
fps={}
def pick(score,div=0.6):
    o=[];f_=[]
    for r in U.assign(_s=score).sort_values("_s",ascending=False).head(6000).itertuples():
        if r.CatalogID not in fps: fps[r.CatalogID]=g2.GetFingerprint(Chem.MolFromSmiles(r.SMILES))
        f=fps[r.CatalogID]
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=div: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=50: break
    return o
base=pick(ra)
F={"rank-sum":rd+ra,"rank-product":np.log(rd)+np.log(ra),"min-rank (consenso)":np.minimum(rd,ra),"min-rank + 0.1*AL":np.minimum(rd,ra)+0.1*ra}
for g in (0.8,0.9,0.95): F[f"DEL gate >= p{int(g*100)} then AL"]=np.where(rd>=g,ra,-1)
for k,s in F.items():
    l=pick(s); print(f"{k:28s} overlap AL1 {len(set(l)&T1):2d}  ablation(changed w/o DEL) {50-len(set(l)&set(base)):2d}")
