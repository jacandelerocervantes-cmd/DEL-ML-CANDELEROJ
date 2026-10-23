"""exp55: one 'active learning round' in VALIDATION. Re-train the model with the aggregate outcome of the earlier lists used as SOFT labels, then pick 50 never-tested compounds.
Soft labels (probability of being a hit) from the known results: 156: analogs 6/15=0.40, first 17 model picks 15/17=0.88, remaining 18 model picks 14/18=0.78; 157: 17/50=0.34 (uniform);
158: MCS-series 6/9=0.67, singleton-analogs 0/10 (true negatives), model 8/31=0.26; 159: 13/50=0.26. Each soft compound enters twice (positive with weight p, negative with weight 1-p).
Control: 168 (the unadjusted model, 12 hits in 50) on an equivalent candidate set. Output submissions/Team_candeleroj_169.txt. RDKit venv."""
import glob, os, numpy as np, pandas as pd, time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:4.0f}s]",*a,flush=True)
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g3.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sv=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sv]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID); labeled=set(o.CatalogID)
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=168: tested|={l.strip() for l in open(f) if l.strip()}
soft={}                                                          # compuesto -> probabilidad de hit
a=pd.read_csv("data/processed/gp/exp39_A2v.csv"); mod=a[a.origen=="modelo"].CatalogID.tolist()
for c in a[a.origen=="analogo"].CatalogID: soft[c]=6/15
for i,c in enumerate(mod): soft[c]=15/17 if i<17 else 14/18
for c in [l.strip() for l in open("submissions/Team_candeleroj_157.txt") if l.strip()]: soft.setdefault(c,17/50)
b=pd.read_csv("data/processed/gp/exp45_A4v.csv")
for r in b.itertuples(): soft[r.CatalogID]=6/9 if r.origen.startswith("serie_ampliada") else (0.0 if r.origen=="analogo_hit_suelto" else 8/31)
for c in [l.strip() for l in open("submissions/Team_candeleroj_159.txt") if l.strip()]: soft.setdefault(c,13/50)
soft={c:p for c,p in soft.items() if c not in labeled}; log("compuestos con etiqueta blanda:",len(soft),"| p media",round(np.mean(list(soft.values())),3))
W=v[~v.CatalogID.isin(labeled)].reset_index(drop=True); ms=[Chem.MolFromSmiles(s) for s in W[sv]]; XW=np.vstack([cnt(ms[i:i+20000]) for i in range(0,len(ms),20000)]); idx={c:i for i,c in enumerate(W.CatalogID)}
XL=cnt([Chem.MolFromSmiles(vs[x]) for x in VH+VN]); yl=np.r_[np.ones(len(VH)),np.zeros(len(VN))]
si=[idx[c] for c in soft if c in idx]; sp=np.array([soft[W.CatalogID[i]] for i in si]); XS=XW[si]
def crossfit(use_soft,seed=55):
    rng=np.random.default_rng(seed); half=rng.permutation(len(W)); H1,H2=half[:len(W)//2],half[len(W)//2:]; sc=np.zeros(len(W),dtype=np.float32)
    for tr,te in ((H1,H2),(H2,H1)):
        ex=rng.choice(tr,50000,replace=False); X=[XL,XW[ex]]; y=[yl,np.zeros(len(ex))]; w=[np.ones(len(yl)),np.ones(len(ex))]
        if use_soft: X+= [XS,XS]; y+=[np.ones(len(sp)),np.zeros(len(sp))]; w+=[sp,1-sp]
        m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack(X),np.concatenate(y),sample_weight=np.concatenate(w)); sc[te]=m.decision_function(XW[te])
    return sc
s_old=crossfit(False); s_new=crossfit(True); log("modelos listos")
W["old"]=s_old; W["new"]=s_new; N=W[~W.CatalogID.isin(tested)].reset_index(drop=True); print("candidatos nunca enviados:",len(N))
def diversified(col,n=50):
    top=N.sort_values(col,ascending=False).head(4000).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sv]]; pick=[];pf=[];seen=set()
    for i,r in top.iterrows():
        if len(pick)>=n: break
        k=(MolToInchiKey(Chem.MolFromSmiles(r[sv])) or "")[:14]
        if k in seen or (pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=0.6): continue
        pick.append(r.CatalogID);pf.append(fps[i]);seen.add(k)
    return pick
p_new=diversified("new"); p_old=diversified("old"); pth="submissions/Team_candeleroj_169.txt"; assert len(set(p_new))==50 and not os.path.exists(pth) and not (set(p_new)&tested); open(pth,"w").write("\n".join(p_new)+"\n")
print("169 (modelo ajustado con etiquetas blandas): comparte con el modelo original sobre el mismo conjunto:",len(set(p_new)&set(p_old)),"de 50 | solo en 169:",len(set(p_new)-set(p_old)))
print("correlacion de puntajes viejo/nuevo en los candidatos:",round(float(np.corrcoef(N.old,N.new)[0,1]),3))
