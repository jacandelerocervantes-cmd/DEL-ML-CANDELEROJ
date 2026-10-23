"""exp53: AL2 with the DEL as a real component of BOTH the score and the selection (organizers: DEL must be a key component of the final workflow).
blend = z(model) + W*z(DEL logistic)   with W = 0.3 (random-CV on the 6,580 labels: PR-AUC 0.433 -> 0.449, ROC-AUC unchanged, top-1% hits 4.20 -> 4.00 of 7.8; exp49).
z(model) is recovered from exp48's rescaled score (a linear map of the raw decision function). Selection recipe as exp48 but ranked by the blend, DISJOINT from AL1's 50:
~33 model picks (diversified Tanimoto<0.6) + 17 members of the extended series (MCS) of multi-hit series (2-3 per series). Output CSV in the organizers' format (CatalogID, Score, Sel_50). RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
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
mem=list(dict.fromkeys(mem))[:17]
top=df[~df.CatalogID.isin(al1|set(mem))].sort_values("blend",ascending=False).head(4000).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.smiles]; pick=[];pf=[];seen=set()
for i,r in top.iterrows():
    if len(pick)>=50-len(mem): break
    k=(MolToInchiKey(Chem.MolFromSmiles(r.smiles)) or "")[:14]
    if k in seen or (pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=0.6): continue
    pick.append(r.CatalogID);pf.append(fps[i]);seen.add(k)
sel=set(mem+pick); assert len(sel)==50 and not (sel&al1)
S=(blend-blend.min())/(blend.max()-blend.min()); m=np.array([c in sel for c in ids]); s=S.copy(); s[m]=np.maximum(s[m],0.999)
o=pd.DataFrame({"CatalogID":ids,"Score":np.round(s,6),"Sel_50":m.astype(int)}).sort_values("Score",ascending=False)
p="submissions/active_learning_candidates/Team_candeleroj_AL2_DELforward_20261005.csv"; o.to_csv(p,index=False); chk=pd.read_csv(p)
assert list(chk.columns)==["CatalogID","Score","Sel_50"] and len(chk)==178052 and chk.Sel_50.sum()==50 and chk.Score.between(0,1).all() and chk.Score.notna().all()
print("AL2 (DEL al",W,"en puntaje y seleccion):",len(mem),"de series ampliadas +",len(pick),"del modelo | comparte con AL1:",len(sel&al1),"| comparte con el B6 anterior:",len(sel&old_b6),"de 50")
pd.DataFrame({"CatalogID":mem+pick,"origen":["serie_ampliada"]*len(mem)+["modelo_blend"]*len(pick)}).to_csv("data/processed/gp/exp53_selection_AL2_DELforward.csv",index=False)
