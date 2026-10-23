"""exp46: replicate on the TEST split the recipe that gave 35/50, 17/50, 14/50, 13/50 hits in validation (envios 156-159): 
(a) one analog per known series (Tanimoto>=0.45 to a series member; series = Butina 0.32 of the 39 hits) and (b) the rest from a logistic regression (ECFP6 counts, C=0.05, balanced)
trained on the 39 hits + 6,541 negatives + background negatives from the unlabelled pool, diversified (1 per Butina cluster, sim>=0.6 -> distance 0.4), InChIKey14 dedup.
CROSS-FITTING: the pool is split in two halves; background negatives come from one half and the other half is scored (and vice versa), so EVERY unlabelled compound gets a score.
A5 = best analogs + top model ranks, B5 = next ones (disjoint). RDKit venv. Outputs data/processed/gp/exp46_*."""
import numpy as np, pandas as pd, time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:4.0f}s]",*a,flush=True)
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values; P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
lab=set(al.CatalogID); U=P[~P.CatalogID.isin(lab)].reset_index(drop=True); mols=[Chem.MolFromSmiles(s) for s in U[sc]]; log("pool sin etiqueta:",len(U))
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g3.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
XU=np.vstack([cnt(mols[i:i+20000]) for i in range(0,len(mols),20000)]); XL=cnt([Chem.MolFromSmiles(s) for s in al.RDKit_SMILES]); log("fingerprints listos")
rng=np.random.default_rng(46); half=rng.permutation(len(U)); H1,H2=half[:len(U)//2],half[len(U)//2:]; score=np.zeros(len(U),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False)
    m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XU[ex]]),np.r_[y,np.zeros(len(ex))])
    score[te]=m.decision_function(XU[te]); log("mitad lista")
# analogos: serie de los 39 hits
H=al[al.Label==1].reset_index(drop=True); HF=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in H.RDKit_SMILES]; n=len(HF); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(HF[i],HF[:i])]
ser=np.zeros(n,int)
for k,c in enumerate(sorted(Butina.ClusterData(d,n,1-0.32,isDistData=True),key=len,reverse=True)): ser[list(c)]=k
fps=[g2.GetFingerprint(m) for m in mols]; SU=np.array([DataStructs.BulkTanimotoSimilarity(f,HF) for f in fps],dtype=np.float32); log("series de los hits:",ser.max()+1)
U["simmax"]=SU.max(1); U["serie"]=ser[SU.argmax(1)]; U["score"]=score; U["ik"]=[(MolToInchiKey(m) or "")[:14] for m in mols]
top=U.sort_values("score",ascending=False).head(3000).reset_index(drop=True); tf=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sc]]; nn=len(tf); dd=[]
for i in range(1,nn): dd+=[1-x for x in DataStructs.BulkTanimotoSimilarity(tf[i],tf[:i])]
cl=np.zeros(nn,int)
for k,c in enumerate(Butina.ClusterData(dd,nn,0.4,isDistData=True)): cl[list(c)]=k
top["cl"]=cl; ranked=top.drop_duplicates("cl")
used=set(); out={}
for nm in ("A5","B5"):
    pick=[];orig=[];seen=set()
    for k in range(ser.max()+1):
        g=U[(U.serie==k)&(U.simmax>=0.45)&(~U.CatalogID.isin(used))].sort_values("simmax",ascending=False)
        for _,r in g.iterrows():
            if r.ik in seen: continue
            pick.append(r.CatalogID);orig.append(f"analogo_serie_{k}");seen.add(r.ik);used.add(r.CatalogID);break
    for _,r in ranked.iterrows():
        if len(pick)>=50: break
        if r.CatalogID in used or r.ik in seen: continue
        pick.append(r.CatalogID);orig.append("modelo");seen.add(r.ik);used.add(r.CatalogID)
    assert len(pick)==50 and len(set(pick))==50 and not (set(pick)&lab); out[nm]=(pick,orig)
    pd.DataFrame({"CatalogID":pick,"origen":orig}).to_csv(f"data/processed/gp/exp46_selection_{nm}.csv",index=False)
    print(nm,pd.Series(orig).str.replace(r"_\d+$","",regex=True).value_counts().to_dict(),"| puntaje del modelo (percentil en el pool) de los del modelo:",round(float((U.score.rank(pct=True)[U.CatalogID.isin([p for p,o in zip(pick,orig) if o=='modelo'])]).median()),4))
print("solape A5/B5:",len(set(out['A5'][0])&set(out['B5'][0])))
A4=set(pd.read_csv("data/processed/gp/exp44_selection_A4.csv").CatalogID)|set(pd.read_csv("data/processed/gp/exp44_selection_B4.csv").CatalogID)
for nm in ("A5","B5"): print(nm,"comparte con A4/B4:",len(set(out[nm][0])&A4),"de 50")
s=pd.read_csv("data/processed/gp/exp16_pool_scores.csv")[["CatalogID"]]; s["lab"]=s.CatalogID.map(dict(zip(al.CatalogID,al.Label))); sm=dict(zip(U.CatalogID,U.score)); unl=s.lab.isna()
raw=s.CatalogID.map(sm); lo,hi=raw[unl].min(),raw[unl].max()
for nm in ("A5","B5"):
    sel=set(out[nm][0]); o=pd.DataFrame({"CatalogID":s.CatalogID,"Sel_50":s.CatalogID.isin(sel).astype(int)}); o["Score"]=np.where(unl,0.001+0.998*(raw-lo)/(hi-lo),np.where(s.lab==1,1.0,0.0))
    o.loc[o.Sel_50==1,"Score"]=o.loc[o.Sel_50==1,"Score"].clip(lower=0.999); o["Score"]=o.Score.round(6); assert len(o)==184632 and o.Sel_50.sum()==50 and o.Score.notna().all(); o.to_csv(f"data/processed/gp/exp46_scores_{nm}.csv",index=False)
