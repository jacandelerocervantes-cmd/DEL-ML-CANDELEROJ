"""exp58: three cluster-oriented validation submissions on NEVER-TESTED compounds (excluded: submissions 1-173 and the labelled ones).
176 (a) chemical regions: Butina regions (Tanimoto>=0.5) over the model's top-8000; best-scored compound of each of the 50 best regions.
178 (b) binding sites: DEL logistic trained ONLY on DEL positives that do NOT compete with the inhibitor (non-ATP-site, 538 positives) -> top-50 (diversified 0.6).
179 (c) known-series regions: never-tested compounds with Tanimoto>=0.4 to a hit of each of the 15 validation series (Butina 0.32 of the 32 hits), best model score first, up to 3 per series, then fill by similarity.
RDKit venv."""
import glob, os, numpy as np, pandas as pd, time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:4.0f}s]",*a,flush=True)
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g=g3:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sv=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sv]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID); labeled=set(o.CatalogID)
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=173: tested|={l.strip() for l in open(f) if l.strip()}
W=v[~v.CatalogID.isin(labeled)].reset_index(drop=True); ms=[Chem.MolFromSmiles(s) for s in W[sv]]; XW=np.vstack([cnt(ms[i:i+20000]) for i in range(0,len(ms),20000)])
XL=cnt([Chem.MolFromSmiles(vs[x]) for x in VH+VN]); yl=np.r_[np.ones(len(VH)),np.zeros(len(VN))]; rng=np.random.default_rng(58)
half=rng.permutation(len(W)); H1,H2=half[:len(W)//2],half[len(W)//2:]; sw=np.zeros(len(W),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False); m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XW[ex]]),np.r_[yl,np.zeros(len(ex))]); sw[te]=m.decision_function(XW[te])
W["sw"]=sw; log("modelo listo")
N=W[~W.CatalogID.isin(tested)].reset_index(drop=True); nm=[ms[i] for i in np.where(~W.CatalogID.isin(tested).values)[0]]; print("candidatos nunca enviados:",len(N))
ik=lambda m:(MolToInchiKey(m) or "")[:14]
def write(num,ids,note):
    ids=list(dict.fromkeys(ids)); assert len(ids)==50 and set(ids)<=set(v.CatalogID) and not (set(ids)&tested), (num,len(ids))
    p=f"submissions/Team_candeleroj_{num}.txt"; assert not os.path.exists(p); open(p,"w").write(chr(10).join(ids)+chr(10)); print(num,note)
# ---- (a) regiones quimicas
top=N.sort_values("sw",ascending=False).head(8000); tix=list(top.index); tf=[g2.GetFingerprint(nm[i]) for i in tix]; n=len(tf); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(tf[i],tf[:i])]
cl=Butina.ClusterData(d,n,0.5,isDistData=True); log("regiones (Butina 0.5) en el top-8000:",len(cl))
reps=sorted([(float(N.sw[tix[c[0]]]),N.CatalogID[tix[c[0]]],len(c)) for c in cl],reverse=True)[:50]   # el primer miembro de cada cluster es su centroide; se usa el de mayor puntaje
best=[]
for c in cl:
    j=max(c,key=lambda t:N.sw[tix[t]]); best.append((float(N.sw[tix[j]]),N.CatalogID[tix[j]]))
best=sorted(best,reverse=True)[:50]; write(176,[b[1] for b in best],f"(a) 50 regiones quimicas distintas | rango de puntaje del modelo {best[-1][0]:.2f}..{best[0][0]:.2f}")
# ---- (b) sitio no competitivo
t=pd.read_pickle("data/processed/del_sel.pkl"); t=t[t.SMILES.notna()&(t.SMILES.str.len()>0)]
pa=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; pe=pa[~(pa.count_PGK2_with_inhibitor<=pa.count_PGK2/4)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(cnt([Chem.MolFromSmiles(s) for s in list(pe.SMILES)+list(neg.SMILES)],g2),np.r_[np.ones(len(pe)),np.zeros(len(neg))])
sd=np.concatenate([dm.decision_function(cnt(nm[i:i+20000],g2)) for i in range(0,len(nm),20000)]); log("DEL no competitivo listo (positivos:",len(pe),")")
N["sd"]=sd; top=N.sort_values("sd",ascending=False).head(3000).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sv]]; pick=[];pf=[];seen=set()
for i,r in top.iterrows():
    if len(pick)>=50: break
    k=ik(Chem.MolFromSmiles(r[sv]))
    if k in seen or (pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=0.6): continue
    pick.append(r.CatalogID);pf.append(fps[i]);seen.add(k)
write(178,pick,"(b) DEL de sitio no competitivo (otro sitio distinto al ATP)")
# ---- (c) regiones de series conocidas
HF=[g2.GetFingerprint(Chem.MolFromSmiles(vs[h])) for h in VH]; nh=len(HF); dd=[]
for i in range(1,nh): dd+=[1-x for x in DataStructs.BulkTanimotoSimilarity(HF[i],HF[:i])]
ser=np.zeros(nh,int)
for k,c in enumerate(Butina.ClusterData(dd,nh,1-0.32,isDistData=True)): ser[list(c)]=k
SU=np.array([DataStructs.BulkTanimotoSimilarity(g2.GetFingerprint(m),HF) for m in nm],dtype=np.float32); N["simmax"]=SU.max(1); N["serie"]=ser[SU.argmax(1)]; log("series:",ser.max()+1)
pick=[];seen=set()
for rnd in range(3):
    for k in range(ser.max()+1):
        g=N[(N.serie==k)&(N.simmax>=0.4)&(~N.CatalogID.isin(pick))].sort_values("sw",ascending=False)
        for _,r in g.iterrows():
            kk=ik(Chem.MolFromSmiles(r[sv]))
            if kk in seen: continue
            pick.append(r.CatalogID);seen.add(kk);break
if len(pick)<50:
    for _,r in N[~N.CatalogID.isin(pick)].sort_values("simmax",ascending=False).iterrows():
        if len(pick)>=50: break
        kk=ik(Chem.MolFromSmiles(r[sv]))
        if kk not in seen: pick.append(r.CatalogID);seen.add(kk)
write(179,pick[:50],f"(c) vecinos (Tanimoto>=0.4) de las {ser.max()+1} series conocidas | con vecinos >=0.4: {int((N.simmax>=0.4).sum())}")
print("solapes: 176/178",len(set(open('submissions/Team_candeleroj_176.txt').read().split())&set(open('submissions/Team_candeleroj_178.txt').read().split())),"| 176/179",len(set(open('submissions/Team_candeleroj_176.txt').read().split())&set(open('submissions/Team_candeleroj_179.txt').read().split())),"| 178/179",len(set(open('submissions/Team_candeleroj_178.txt').read().split())&set(open('submissions/Team_candeleroj_179.txt').read().split())))
