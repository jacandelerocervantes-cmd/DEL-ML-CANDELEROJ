"""Step 2: DEL-first selection with the AL model as correction, plus the ablation required by the organizers (2026-10-22 posts).
Design: pool = top DEL_POOL_PCT % of the pool by DEL-ML score; score = z(DEL) + W_AL * z(AL), z over the whole pool; 50 compounds greedy with Tanimoto < DIVERSITY (Morgan r2, 2,048 bits) among the best CAND candidates.
List 2 = the next 50 under the same rule, excluding list 1. Ablation = the same lists with the AL score only (no DEL); the number of list-1 compounds that change is reported.
Edit the CONFIG line to change the design; every run is printed. Optional comparison with the 100 compounds of the earlier submissions in reference/earlier_workflow_selection_ids.csv (a comparison only, not an input).
Output: results/selection_list1.csv, results/selection_list2.csv, results/ablation.json. RDKit venv."""
import json, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
CONFIG=dict(DEL_POOL_PCT=20, W_AL=1.0, DIVERSITY=0.6, CAND=6000)
U=pd.read_csv("results/pool_scores.csv"); z=lambda a:(a-a.mean())/a.std(); U["zdel"]=z(U["del"]); U["zal"]=z(U.al)
fps={}
def pick(df,score,exclude=(),n=50):
    o=[];f_=[]
    for r in df[~df.CatalogID.isin(exclude)].assign(_s=score).sort_values("_s",ascending=False).head(CONFIG["CAND"]).itertuples():
        if r.CatalogID not in fps: fps[r.CatalogID]=g2.GetFingerprint(Chem.MolFromSmiles(r.SMILES))
        f=fps[r.CatalogID]
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=CONFIG["DIVERSITY"]: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=n: break
    return o
def two(df,score): a=pick(df,score); return a,pick(df,score,exclude=set(a))
pool=U[U["del"].rank(ascending=False)<=CONFIG["DEL_POOL_PCT"]/100*len(U)]
l1,l2=two(pool,pool.zdel+CONFIG["W_AL"]*pool.zal); r1,r2=two(U,U.zal)
out=dict(config=CONFIG,pool=len(pool),list1_changed_without_DEL=50-len(set(l1)&set(r1)),list2_changed_without_DEL=50-len(set(l2)&set(r2)))
try:
    ref=pd.read_csv("reference/earlier_workflow_selection_ids.csv"); old=set(ref.CatalogID); out["of_the_earlier_workflow_100_kept"]=len((set(l1)|set(l2))&old); out["ablation_version_of_the_earlier_workflow_100_kept"]=len((set(r1)|set(r2))&old)
except FileNotFoundError: pass
pd.DataFrame({"CatalogID":l1}).to_csv("results/selection_list1.csv",index=False); pd.DataFrame({"CatalogID":l2}).to_csv("results/selection_list2.csv",index=False); json.dump(out,open("results/ablation.json","w"),indent=1); print(json.dumps(out,indent=1))
