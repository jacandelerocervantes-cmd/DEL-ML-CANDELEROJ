"""exp60: AL2, a selection oriented to NEW groups relative to AL1 (clusters metric). Score = z(model) + 0.3*z(DEL), same blend as exp53; the 50 compounds of AL1 are excluded before ranking.
Candidates: the 6,000 highest-scoring compounds, grouped by Butina clustering at Tanimoto 0.32. Step 1 picks 18 compounds whose maximum similarity to AL1's 50 is below 0.32 (best blend per group, 18 best groups).
Step 2 picks 32 high-scoring compounds from the remaining candidates, at most 2 per group (the 18 do not count towards the cap); the 32 have no similarity filter against AL1 or the 18. Output in the organizers' format. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); W=0.3; g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
tmpl=pd.read_csv(r"data/raw/active_learning_submission_example.csv"); ids=list(tmpl.CatalogID)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES)+list(neg.SMILES)]),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
ms=[Chem.MolFromSmiles(smi[c]) for c in ids]; sd=np.concatenate([dm.decision_function(cnt(ms[i:i+20000])) for i in range(0,len(ms),20000)]); print("DEL listo",flush=True)
base=pd.read_csv("data/processed/gp/exp48_scores_A6.csv").set_index("CatalogID").Score.reindex(ids).values
z=lambda a:(a-a.mean())/a.std(); blend=z(base)+W*z(sd); df=pd.DataFrame({"CatalogID":ids,"blend":blend}); df["smiles"]=[smi[c] for c in ids]
al1=set(pd.read_csv("data/processed/gp/exp48_selection_A6.csv").CatalogID); old_b6=set(pd.read_csv("data/processed/gp/exp48_selection_B6.csv").CatalogID)
M=pd.read_csv("data/processed/gp/exp43_unlabelled_series_members.csv").merge(df[["CatalogID","blend"]],on="CatalogID"); R=pd.read_csv("data/processed/gp/exp43_extended_series.csv"); multi=R[(R.mcs!="singleton")&(R.hits>=2)]
quota={int(r.serie):3 if r.sin_etiqueta>=25 else 2 for r in multi.itertuples()}; mem=[]
for k,q in quota.items(): mem+=list(M[(M.serie==k)&(~M.CatalogID.isin(al1|set(mem)))].sort_values("blend",ascending=False).head(q).CatalogID)

fa=[g2.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in al1]
top=df[~df.CatalogID.isin(al1)].sort_values("blend",ascending=False).head(6000).reset_index(drop=True); fp=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.smiles]
top["sim_al1"]=[max(DataStructs.BulkTanimotoSimilarity(f,fa)) for f in fp]
n=len(fp); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(d,n,1-0.32,isDistData=True)): cl[list(c)]=k
top["g"]=cl
new=top[top.sim_al1<0.32]; newpick=list(new.groupby("g").head(1).sort_values("blend",ascending=False).head(18).CatalogID)
rest=top[~top.CatalogID.isin(newpick)]; cap={}; hp=[]; seen=set()
for r in rest.itertuples():
    if len(hp)>=50-len(newpick): break
    if cap.get(r.g,0)>=2: continue
    k=(MolToInchiKey(Chem.MolFromSmiles(r.smiles)) or "")[:14]
    if k in seen: continue
    cap[r.g]=cap.get(r.g,0)+1; hp.append(r.CatalogID); seen.add(k)
sel=set(newpick+hp); assert len(sel)==50 and not (sel&al1)
S=(blend-blend.min())/(blend.max()-blend.min()); m=np.array([c in sel for c in ids]); s=S.copy(); s[m]=np.maximum(s[m],0.999)
o=pd.DataFrame({"CatalogID":ids,"Score":np.round(s,6),"Sel_50":m.astype(int)}).sort_values("Score",ascending=False)
p="submissions/active_learning_candidates/Team_candeleroj_AL2_newgroups_20261005.csv"; o.to_csv(p,index=False); chk=pd.read_csv(p)
assert list(chk.columns)==["CatalogID","Score","Sel_50"] and len(chk)==178052 and chk.Sel_50.sum()==50 and chk.Score.between(0,1).all() and chk.Score.notna().all()
sub=top[top.CatalogID.isin(sel)]
print("newgroups:",len(newpick),"nuevos +",len(hp),"alta prob | grupos 0.32 entre las 50:",sub.g.nunique(),"| con sim<0.32 a AL1:",int((sub.sim_al1<0.32).sum()),"| comparte con B6:",len(sel&old_b6),"| blend medio nuevos",round(top[top.CatalogID.isin(newpick)].blend.mean(),2),"alta prob",round(top[top.CatalogID.isin(hp)].blend.mean(),2),"| blend medio AL2 B6 ref (top50 del blend)",round(top.blend.head(50).mean(),2))
pd.DataFrame({"CatalogID":newpick+hp,"origen":["grupo_nuevo"]*len(newpick)+["alta_prob"]*len(hp)}).to_csv("data/processed/gp/exp60_selection_AL2_newgroups.csv",index=False)
