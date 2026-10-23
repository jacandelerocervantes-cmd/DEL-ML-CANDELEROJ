"""exp48: (a) TEST lists A6/B6 with the recipe supported by validation results 156-165: ~33 model picks (LR ECFP6 trained on the 39 hits + 6,541 negatives + background negatives, cross-fitted,
diversified Tanimoto<0.6) + ~17 members of the extended series (MCS) of multi-hit series (2-3 per series, best model score), NO analogs of singleton hits (163: 0/10). B6 = next ones.
(b) VALIDATION submission 166: same model blended with the DEL logistic score (0.9 rank LR + 0.1 rank DEL) to check that the DEL term does not hurt. RDKit venv."""
import os, numpy as np, pandas as pd, time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:4.0f}s]",*a,flush=True)
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g=g3:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
def feats(smis): 
    ms=[Chem.MolFromSmiles(s) for s in smis]; return ms,np.vstack([cnt(ms[i:i+20000]) for i in range(0,len(ms),20000)])
def crossfit(XL,yl,XU,seed):
    rng=np.random.default_rng(seed); half=rng.permutation(len(XU)); H1,H2=half[:len(XU)//2],half[len(XU)//2:]; sc=np.zeros(len(XU),dtype=np.float32)
    for tr,te in ((H1,H2),(H2,H1)):
        ex=rng.choice(tr,50000,replace=False); m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XU[ex]]),np.r_[yl,np.zeros(len(ex))]); sc[te]=m.decision_function(XU[te])
    return sc
def diversified(df,smicol,score,th,n=50,exclude=(),maxcand=4000):
    top=df.assign(_s=score).sort_values("_s",ascending=False).head(maxcand).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[smicol]]
    pick=[];pf=[];seen=set()
    for i,r in top.iterrows():
        if len(pick)>=n: break
        if r.CatalogID in exclude: continue
        k=(MolToInchiKey(Chem.MolFromSmiles(r[smicol])) or "")[:14]
        if k in seen: continue
        if pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=th: continue
        pick.append(r.CatalogID);pf.append(fps[i]);seen.add(k)
    return pick
# ---------------- (a) TEST
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
lab=set(al.CatalogID); U=P[~P.CatalogID.isin(lab)].reset_index(drop=True); _,XU=feats(U[sc]); _,XL=feats(al.RDKit_SMILES); U["score"]=crossfit(XL,y,XU,48); log("test: modelo listo")
M=pd.read_csv("data/processed/gp/exp43_unlabelled_series_members.csv").merge(U[["CatalogID","score"]],on="CatalogID"); R=pd.read_csv("data/processed/gp/exp43_extended_series.csv"); multi=R[(R.mcs!="singleton")&(R.hits>=2)]
quota={int(r.serie):3 if r.sin_etiqueta>=25 else 2 for r in multi.itertuples()}; used=set(); out={}
for nm in ("A6","B6"):
    mem=[]
    for k,q in quota.items():
        g=M[(M.serie==k)&(~M.CatalogID.isin(used))].sort_values("score",ascending=False).head(q); mem+=list(g.CatalogID); used|=set(g.CatalogID)
    mem=list(dict.fromkeys(mem))[:17]; used|=set(mem)
    mod=diversified(U,sc,U.score.values,0.6,n=50-len(mem),exclude=used|set(mem)); used|=set(mod); pick=mem+mod
    assert len(pick)==50 and len(set(pick))==50 and not (set(pick)&lab); out[nm]=pick
    pd.DataFrame({"CatalogID":pick,"origen":["serie_ampliada"]*len(mem)+["modelo"]*len(mod)}).to_csv(f"data/processed/gp/exp48_selection_{nm}.csv",index=False); print(nm,"serie ampliada",len(mem),"| modelo",len(mod))
print("solape A6/B6:",len(set(out['A6'])&set(out['B6'])))
s=pd.read_csv("data/processed/gp/exp16_pool_scores.csv")[["CatalogID"]]; s["lab"]=s.CatalogID.map(dict(zip(al.CatalogID,al.Label))); sm=dict(zip(U.CatalogID,U.score)); unl=s.lab.isna(); raw=s.CatalogID.map(sm); lo,hi=raw[unl].min(),raw[unl].max()
for nm in ("A6","B6"):
    o=pd.DataFrame({"CatalogID":s.CatalogID,"Sel_50":s.CatalogID.isin(set(out[nm])).astype(int)}); o["Score"]=np.where(unl,0.001+0.998*(raw-lo)/(hi-lo),np.where(s.lab==1,1.0,0.0))
    o.loc[o.Sel_50==1,"Score"]=o.loc[o.Sel_50==1,"Score"].clip(lower=0.999); o["Score"]=o.Score.round(6); assert len(o)==184632 and o.Sel_50.sum()==50 and o.Score.notna().all(); o.to_csv(f"data/processed/gp/exp48_scores_{nm}.csv",index=False)
# ---------------- (b) VALIDATION con mezcla DEL
import sys
if not (os.path.exists("data/processed/confirmed_submission_outcomes_v4.csv") and os.path.exists("submissions/Team_candeleroj_165.txt")):
    print("parte (b) omitida: requiere los resultados de validation (no se publican)"); sys.exit(0)
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sv=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sv]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID); labeled=set(o.CatalogID)
W=v[~v.CatalogID.isin(labeled)].reset_index(drop=True); _,XW=feats(W[sv]); _,XV=feats([vs[x] for x in VH+VN]); sw=crossfit(XV,np.r_[np.ones(len(VH)),np.zeros(len(VN))],XW,47); log("validation: modelo listo")
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
_,XD=feats(list(pos.SMILES)+list(neg.SMILES)); dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(XD,np.r_[np.ones(len(pos)),np.zeros(len(neg))]); sd=dm.decision_function(XW); log("DEL listo")
r=lambda a:rankdata(a)/len(a); blend=0.9*r(sw)+0.1*r(sd)
p166=diversified(W,sv,blend,0.6); p165=[l.strip() for l in open("submissions/Team_candeleroj_165.txt") if l.strip()]; prev=set()
for e in (156,157,158,159): prev|={l.strip() for l in open(f"submissions/Team_candeleroj_{e}.txt") if l.strip()}
pth="submissions/Team_candeleroj_166.txt"; assert not os.path.exists(pth) and len(p166)==50; open(pth,"w").write("\n".join(p166)+"\n")
print("166 (mezcla 0.9 modelo + 0.1 DEL): comparte con 165:",len(set(p166)&set(p165)),"| ya probados en 156-159:",len(set(p166)&prev),"| correlacion de rango modelo-DEL:",round(float(np.corrcoef(r(sw),r(sd))[0,1]),3))
