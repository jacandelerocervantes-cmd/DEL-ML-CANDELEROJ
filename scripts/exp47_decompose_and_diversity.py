"""exp47: six validation submissions (160-165).
Decomposition of the winners: 160 = analogs of 156 (15); 161 = the first 17 MODEL picks of 156 (rank order); 162 = MCS-series picks of 158 (9); 163 = singleton-hit analogs of 158 (10).
Diversity test with the SAME kind of model (LR on the validation labels, cross-fitted so every compound is scored; candidates may include compounds already tested in 156-159):
165 = top-50 diversified at Tanimoto<0.6 (as in 156); 164 = top-50 with pairwise Tanimoto < 0.32 (one official-like series each). RDKit venv."""
import os, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*')
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc])); vset=set(v.CatalogID)
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID); labeled=set(o.CatalogID)
rng=np.random.default_rng(47)
def write(num,ids,note):
    ids=list(ids); pad=[x for x in VN if x not in ids]; L=ids+list(rng.choice(pad,50-len(ids),replace=False)); assert len(set(L))==50 and set(L)<=vset
    p=f"submissions/Team_candeleroj_{num}.txt"; assert not os.path.exists(p); open(p,"w").write("\n".join(L)+"\n"); print(num,note,f"({len(ids)} compuestos + {50-len(ids)} negativos confirmados)")
a=pd.read_csv("data/processed/gp/exp39_A2v.csv"); b=pd.read_csv("data/processed/gp/exp45_A4v.csv")
write(160,a[a.origen=="analogo"].CatalogID,"156: analogos por Tanimoto")
write(161,a[a.origen=="modelo"].CatalogID.head(17),"156: primeros 17 del modelo")
write(162,b[b.origen.str.startswith("serie_ampliada")].CatalogID,"158: series por MCS")
write(163,b[b.origen=="analogo_hit_suelto"].CatalogID,"158: analogos de hits sueltos")
# ---- modelo con validacion cruzada por mitades sobre TODO el pool no etiquetado
g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
U=v[~v.CatalogID.isin(labeled)].reset_index(drop=True); mols=[Chem.MolFromSmiles(s) for s in U[sc]]
cnt=lambda ms:np.log1p(np.array([g3.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
XU=np.vstack([cnt(mols[i:i+20000]) for i in range(0,len(mols),20000)]); XL=cnt([Chem.MolFromSmiles(vs[x]) for x in VH+VN]); yl=np.r_[np.ones(len(VH)),np.zeros(len(VN))]
half=rng.permutation(len(U)); H1,H2=half[:len(U)//2],half[len(U)//2:]; score=np.zeros(len(U),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False); m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XU[ex]]),np.r_[yl,np.zeros(len(ex))]); score[te]=m.decision_function(XU[te])
U["score"]=score; top=U.sort_values("score",ascending=False).head(3000).reset_index(drop=True); fps=[g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top[sc]]
top["ik"]=[(MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14] for s in top[sc]]
# 165: diversificado a Tanimoto<0.6 (como 156): codicioso por puntaje
def greedy(th):
    pick=[];pf=[];seen=set()
    for i,r in top.iterrows():
        if len(pick)>=50: break
        if r.ik in seen: continue
        if pf and max(DataStructs.BulkTanimotoSimilarity(fps[i],pf))>=th: continue
        pick.append(r.CatalogID);pf.append(fps[i]);seen.add(r.ik)
    return pick,i
p165,d165=greedy(0.6); p164,d164=greedy(0.32)
write(165,p165,f"modelo diversificado a Tanimoto<0.6 (profundidad del ranking: {d165+1})")
write(164,p164,f"modelo con pares Tanimoto<0.32 (profundidad del ranking: {d164+1})")
prev=set()
for e in (156,157,158,159): prev|={l.strip() for l in open(f"submissions/Team_candeleroj_{e}.txt") if l.strip()}
for n,p in ((164,p164),(165,p165)): print(f"  {n}: {len(set(p)&prev)} de sus 50 ya estaban en 156-159")
print("solape 164/165:",len(set(p164)&set(p165)))
