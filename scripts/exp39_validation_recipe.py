"""exp39: test the A2/B2 RECIPE in the validation split (where we know 32 hits / 15 official series and 2,592 negatives).
Recipe = per known series the best unlabelled analog (Tanimoto>=0.45 to a series member, 1 per series in A2v, next one in B2v) + the rest from a logistic model trained on the
validation labels (ECFP6 counts, 100k background negatives) with diversity (1 per Butina cluster, sim>=0.6) and InChIKey14 dedup. Excludes every compound already in submissions 1-155.
Outputs submissions/Team_candeleroj_156.txt (A2v) and _157.txt (B2v) + data/processed/gp/exp39_*.csv. Run with the RDKit venv."""
import glob, os, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID)
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=155: tested|={l.strip() for l in open(f) if l.strip()}
U=v[~v.CatalogID.isin(tested)].reset_index(drop=True); print("validation sin enviar:",len(U),"(enviados:",len(tested),")",flush=True)
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
mol=lambda s:Chem.MolFromSmiles(s); HF=[g2.GetFingerprint(mol(vs[x])) for x in VH]; n=len(HF); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(HF[i],HF[:i])]
ser=np.zeros(n,int)
for k,c in enumerate(sorted(Butina.ClusterData(d,n,1-0.32,isDistData=True),key=len,reverse=True)): ser[list(c)]=k
print("series de validation (0.32):",ser.max()+1,flush=True)
fps=[g2.GetFingerprint(mol(s)) for s in U[sc]]; SU=np.array([DataStructs.BulkTanimotoSimilarity(f,HF) for f in fps],dtype=np.float32)
U["simmax"]=SU.max(1); U["serie"]=ser[SU.argmax(1)]; print("analogos >=0.45:",int((U.simmax>=0.45).sum()),"| >=0.6:",int((U.simmax>=0.6).sum()),flush=True)
# modelo con etiquetas de validation
cnt=lambda S:np.log1p(np.array([g3.GetCountFingerprintAsNumPy(mol(s)) for s in S],dtype=np.float32))
rng=np.random.default_rng(39); extra=rng.choice(len(U),100000,replace=False); keep=np.setdiff1d(np.arange(len(U)),extra)
Xl=cnt([vs[x] for x in VH+VN]); yl=np.r_[np.ones(len(VH)),np.zeros(len(VN))]; Xe=cnt(U[sc].iloc[extra])
m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([Xl,Xe]),np.r_[yl,np.zeros(len(extra))])
U["score"]=np.nan; U.loc[keep,"score"]=np.concatenate([m.decision_function(cnt(U[sc].iloc[keep[i:i+20000]])) for i in range(0,len(keep),20000)]); print("modelo listo",flush=True)
U["ik"]=[MolToInchiKey(mol(s))[:14] if mol(s) else "" for s in U[sc]] if False else ""
picks={"A2v":[],"B2v":[]}; used=set(); ikseen={"A2v":set(),"B2v":set()}
cand=U[U.simmax>=0.45].copy(); cand["ik"]=[(MolToInchiKey(mol(s)) or "")[:14] for s in cand[sc]]; ikmap=dict(zip(cand.CatalogID,cand.ik))
for name in ("A2v","B2v"):
    for k in range(ser.max()+1):
        g=cand[(cand.serie==k)&(~cand.CatalogID.isin(used))&(~cand.ik.isin(ikseen[name]))].sort_values("simmax",ascending=False)
        if len(g): r=g.iloc[0]; picks[name].append((r.CatalogID,"analogo",k,round(float(r.simmax),2))); used.add(r.CatalogID); ikseen[name].add(r.ik)
top=U[(~U.CatalogID.isin(used))&U.score.notna()].sort_values("score",ascending=False).head(2000).reset_index(drop=True)
tf=[g2.GetFingerprint(mol(s)) for s in top[sc]]; nn=len(tf); dd=[]
for i in range(1,nn): dd+=[1-x for x in DataStructs.BulkTanimotoSimilarity(tf[i],tf[:i])]
cl=np.zeros(nn,int)
for k,c in enumerate(Butina.ClusterData(dd,nn,0.4,isDistData=True)): cl[list(c)]=k
top["cl"]=cl; top["ik"]=[(MolToInchiKey(mol(s)) or "")[:14] for s in top[sc]]; seen_cl=set()
for name in ("A2v","B2v"):
    for _,r in top.iterrows():
        if len(picks[name])>=50: break
        if r.CatalogID in used or r.cl in seen_cl or r.ik in ikseen[name]: continue
        picks[name].append((r.CatalogID,"modelo",-1,round(float(r.score),2))); used.add(r.CatalogID); seen_cl.add(r.cl); ikseen[name].add(r.ik)
for name,num in (("A2v",156),("B2v",157)):
    L=[p[0] for p in picks[name]]; assert len(set(L))==50 and set(L)<=set(v.CatalogID) and not (set(L)&tested), name
    pth=f"submissions/Team_candeleroj_{num}.txt"; assert not os.path.exists(pth); open(pth,"w").write("\n".join(L)+"\n")
    pd.DataFrame(picks[name],columns=["CatalogID","origen","serie","sim_o_score"]).assign(smiles=[vs[x] for x in L]).to_csv(f"data/processed/gp/exp39_{name}.csv",index=False)
    na=sum(p[1]=="analogo" for p in picks[name]); print(f"{name} -> envio {num}: {na} analogos (1 por serie) + {50-na} del modelo; series cubiertas por analogos: {sorted(p[2] for p in picks[name] if p[1]=='analogo')}")
print("solape A2v/B2v:",len({p[0] for p in picks['A2v']}&{p[0] for p in picks['B2v']}))
