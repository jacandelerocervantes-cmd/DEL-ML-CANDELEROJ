"""exp50: clean paired test of the DEL z-blend in VALIDATION. Candidates = validation compounds NEVER sent (excluded: everything in submissions 1-166 and the labelled ones).
168 = top-50 by the model alone (diversified Tanimoto<0.6); 167 = top-50 by z(model) + 0.1*z(DEL) (same diversification). Same candidate set, same depth: the only difference is the DEL term.
Compounds picked by both count equally in both lists, so the difference in hits comes from the compounds picked by only one. RDKit venv."""
import glob, os, numpy as np, pandas as pd, time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:4.0f}s]",*a,flush=True)
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g=g3:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sv=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sv]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID); labeled=set(o.CatalogID)
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=166: tested|={l.strip() for l in open(f) if l.strip()}
W=v[~v.CatalogID.isin(labeled)].reset_index(drop=True); ms=[Chem.MolFromSmiles(s) for s in W[sv]]; XW=np.vstack([cnt(ms[i:i+20000]) for i in range(0,len(ms),20000)])
XL=cnt([Chem.MolFromSmiles(vs[x]) for x in VH+VN]); yl=np.r_[np.ones(len(VH)),np.zeros(len(VN))]; rng=np.random.default_rng(50)
half=rng.permutation(len(W)); H1,H2=half[:len(W)//2],half[len(W)//2:]; sw=np.zeros(len(W),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False); m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XW[ex]]),np.r_[yl,np.zeros(len(ex))]); sw[te]=m.decision_function(XW[te])
log("modelo listo")
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
XD=cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES)+list(neg.SMILES)],g2); dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(XD,np.r_[np.ones(len(pos)),np.zeros(len(neg))])
sd=np.concatenate([dm.decision_function(cnt(ms[i:i+20000],g2)) for i in range(0,len(ms),20000)]); log("DEL listo")
z=lambda a:(a-a.mean())/a.std(); W["sw"]=sw; W["blend"]=z(sw)+0.1*z(sd); N=W[~W.CatalogID.isin(tested)].reset_index(drop=True); print("candidatos nunca enviados:",len(N))
def diversified(col,n=50):
    top=N.sort_values(col,ascending=False).head(4000).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sv]]; pick=[];pf=[];seen=set()
    for i,r in top.iterrows():
        if len(pick)>=n: break
        k=(MolToInchiKey(Chem.MolFromSmiles(r[sv])) or "")[:14]
        if k in seen or (pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=0.6): continue
        pick.append(r.CatalogID);pf.append(fps[i]);seen.add(k)
    return pick
p168=diversified("sw"); p167=diversified("blend")
for num,p in ((167,p167),(168,p168)):
    pth=f"submissions/Team_candeleroj_{num}.txt"; assert len(set(p))==50 and not os.path.exists(pth) and not (set(p)&tested) and set(p)<=set(v.CatalogID); open(pth,"w").write("\n".join(p)+"\n")
pc=lambda ids:float(N[N.CatalogID.isin(ids)].sw.rank(pct=True).median()) if False else float(np.median([(N.sw<x).mean() for x in N[N.CatalogID.isin(ids)].sw]))
print("167 (modelo + 0.1 DEL) y 168 (solo modelo): comparten",len(set(p167)&set(p168)),"de 50 | solo en 167:",len(set(p167)-set(p168)),"| solo en 168:",len(set(p168)-set(p167)))
print("percentil medio del modelo entre los candidatos: 167",round(pc(p167),4),"| 168",round(pc(p168),4))
