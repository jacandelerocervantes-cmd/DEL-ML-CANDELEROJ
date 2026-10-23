"""exp71: separate analysis done after AL1 and AL2 were submitted (it changes neither). Question: with workflows that give the DEL-ML model a significant role (organizers' rules of 2026-10-22: DEL must not be a small filter or a small blended weight;
pairing an AL model with a DEL-ML model is admissible; the check is an ablation on the 50 selected), how many of the 100 compounds already submitted (AL1 + AL2) would still be selected?
Grid fixed in advance and reported in full (no design was dropped). Each design gives list 1 (50) and list 2 (next 50, excluding list 1, diversity 0.6). Ablation = the same list 1 built with the AL score only.
Needs data/processed/gp/exp70_pool_scores.csv (written by exp70). Output: results/exp71_del_designs_overlap.json. RDKit venv."""
import json, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
U=pd.read_csv("data/processed/gp/exp70_pool_scores.csv").merge(P[["CatalogID",sc]].rename(columns={sc:"smi"}),on="CatalogID")
z=lambda a:(a-a.mean())/a.std(); U["zdel"]=z(U["del"]); U["zal"]=z(U.al); N=len(U); U["rdel"]=U["del"].rank(ascending=False); U["ral"]=U.al.rank(ascending=False)
A1=set(pd.read_csv("data/processed/gp/exp48_selection_A6.csv").CatalogID); A2=set(pd.read_csv("data/processed/gp/exp60_selection_AL2_newgroups.csv").CatalogID); OLD=A1|A2
fps={}
def fp(c,s):
    if c not in fps: fps[c]=g2.GetFingerprint(Chem.MolFromSmiles(s))
    return fps[c]
def pick(df,score,exclude=(),n=50,th=0.6,head=6000):
    o=[];f_=[]
    for r in df[~df.CatalogID.isin(exclude)].assign(_s=score).sort_values("_s",ascending=False).head(head).itertuples():
        f=fp(r.CatalogID,r.smi)
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=th: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=n: break
    return o
def two(df,score):
    l1=pick(df,score); l2=pick(df,score,exclude=set(l1)); return l1,l2
ref1,ref2=two(U,U.zal)
designs=[("AL only (reference, no DEL)",U,U.zal)]
for p in (1,2,5,10,20): S=U[U.rdel<=p/100*N]; designs.append((f"DEL pool top {p}% + z(DEL)+z(AL)",S,S.zdel+S.zal))
for p in (5,20): S=U[U.rdel<=p/100*N]; designs.append((f"DEL pool top {p}%, DEL as main + 0.5*z(AL)",S,S.zdel+0.5*S.zal))
for K in (5000,10000,20000,40000): S=U[(U.rdel<=K)&(U.ral<=K)]; designs.append((f"intersection AL top {K} and DEL top {K}, z(DEL)+z(AL)",S,S.zdel+S.zal))
rk=lambda a:a.rank(pct=True)
designs.append(("rank average 0.5 AL + 0.5 DEL, whole pool",U,rk(U.al)+rk(U["del"])))
for w in (1,2,3): designs.append((f"z(AL)+{w}*z(DEL), whole pool",U,U.zal+w*U.zdel))
rows=[]
for nm,df,s in designs:
    l1,l2=two(df,s); un=set(l1)|set(l2); ch=50-len(set(l1)&set(ref1))
    rows.append(dict(design=nm,pool=len(df),list1_changed_without_DEL=ch,list1_in_AL1=len(set(l1)&A1),list1_in_AL2=len(set(l1)&A2),list2_in_AL1=len(set(l2)&A1),list2_in_AL2=len(set(l2)&A2),of_the_100_kept=len(un&OLD),sig_ablation_ge25=ch>=25))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_json("results/exp71_del_designs_overlap.json",orient="records",indent=1)
