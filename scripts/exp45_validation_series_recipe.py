"""exp45: the A4/B4 recipe applied to the VALIDATION split (where we know 32 hits): organizers' series logic (exp42 series + MCS) instead of plain Tanimoto.
Slots: (1) unsent validation compounds that contain the MCS of a known series (3 per series in A4v, next 3 in B4v); (2) best analog (Tanimoto>=0.45) of each singleton hit;
(3) fill with the model picks already computed in exp39 (LR trained on validation labels). Excludes everything in submissions 1-157. Writes submissions/Team_candeleroj_158.txt (A4v) and _159.txt (B4v). RDKit venv."""
import glob, os, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=157: tested|={l.strip() for l in open(f) if l.strip()}
U=v[~v.CatalogID.isin(tested)].reset_index(drop=True); print("validation sin enviar (excluyendo 1-157):",len(U),flush=True)
SER=pd.read_csv("data/processed/gp/exp42_validation_series.csv"); ext=SER[SER.mcs!="singleton"]; single=[h for r in SER[SER.mcs=="singleton"].itertuples() for h in str(r.ids).split(";")]
print("series con MCS:",len(ext),"| hits sueltos:",len(single),flush=True)
mols=[Chem.MolFromSmiles(s) for s in U[sc]]; ok=[m is not None for m in mols]; U=U[ok].reset_index(drop=True); mols=[m for m in mols if m is not None]
ik=lambda m:(MolToInchiKey(m) or "")[:14]
members={}
for r in ext.itertuples():
    q=Chem.MolFromSmarts(r.mcs); idx=[i for i,m in enumerate(mols) if m.HasSubstructMatch(q)]; members[r.serie]=idx
    print(f"  serie {r.serie} ({r.hits} hits): {len(idx)} compuestos sin enviar contienen su MCS",flush=True)
gen=G.GetMorganGenerator(radius=2,fpSize=2048); SF=[gen.GetFingerprint(Chem.MolFromSmiles(vs[h])) for h in single]
best={h:[] for h in single}
for i,m in enumerate(mols):
    sims=DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),SF); j=int(np.argmax(sims))
    if sims[j]>=0.45: best[single[j]].append((sims[j],i))
print("analogos de hits sueltos (>=0.45):",sum(len(x) for x in best.values()),flush=True)
# modelo propio (LR sobre las etiquetas de validation + 100k negativos de fondo), diversificado por Butina; reparte A4v y B4v por turnos
from sklearn.linear_model import LogisticRegression
from rdkit.ML.Cluster import Butina
g3=G.GetMorganGenerator(radius=3,fpSize=2048)
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID); VN=sorted(o[o.confirmed_label==0].CatalogID)
cnt=lambda ms:np.log1p(np.array([g3.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
rng=np.random.default_rng(45); extra=rng.choice(len(U),100000,replace=False); keep=np.setdiff1d(np.arange(len(U)),extra)
Xl=cnt([Chem.MolFromSmiles(vs[x]) for x in VH+VN]); Xe=cnt([mols[i] for i in extra])
lr=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([Xl,Xe]),np.r_[np.ones(len(VH)),np.zeros(len(VN)),np.zeros(len(extra))])
sc_=np.concatenate([lr.decision_function(cnt([mols[i] for i in keep[a:a+20000]])) for a in range(0,len(keep),20000)]); print("modelo listo",flush=True)
top=keep[np.argsort(-sc_)[:2500]]; tf=[gen.GetFingerprint(mols[i]) for i in top]; nn=len(tf); dd=[]
for a in range(1,nn): dd+=[1-x for x in DataStructs.BulkTanimotoSimilarity(tf[a],tf[:a])]
cl=np.zeros(nn,int)
for k,c in enumerate(Butina.ClusterData(dd,nn,0.4,isDistData=True)): cl[list(c)]=k
seen_cl=set(); ranked=[]
for a,i in enumerate(top):
    if cl[a] in seen_cl: continue
    seen_cl.add(cl[a]); ranked.append(U.CatalogID[i])
mod={"A2v":ranked[0::2],"B2v":ranked[1::2]}
res={}; used=set(); idmap=dict(zip(U.CatalogID,range(len(U))))
for nm,mk in (("A4v","A2v"),("B4v","B2v")):
    pick=[];origin=[];seen=set()
    def add(i,o):
        k=ik(mols[i]); c=U.CatalogID[i]
        if c in used or k in seen: return False
        pick.append(c);origin.append(o);seen.add(k);used.add(c); return True
    for s,idx in members.items():
        n=0
        for i in idx:
            if n>=3: break
            if add(i,f"serie_ampliada_{s}"): n+=1
    for h in single:
        for _,i in sorted(best[h],reverse=True):
            if add(i,"analogo_hit_suelto"): break
    for c in mod[mk]:
        if len(pick)>=50: break
        i=idmap.get(c)
        if i is not None: add(i,"modelo")
    assert len(pick)==50 and len(set(pick))==50 and set(pick)<=set(v.CatalogID) and not (set(pick)&tested), nm
    res[nm]=(pick,origin)
for (nm,num) in (("A4v",158),("B4v",159)):
    pick,origin=res[nm]; p=f"submissions/Team_candeleroj_{num}.txt"; assert not os.path.exists(p); open(p,"w").write("\n".join(pick)+"\n")
    pd.DataFrame({"CatalogID":pick,"origen":origin,"smiles":[vs[c] for c in pick]}).to_csv(f"data/processed/gp/exp45_{nm}.csv",index=False)
    print(nm,"-> envio",num,pd.Series(origin).str.replace(r"_\d+$","",regex=True).value_counts().to_dict())
print("solape A4v/B4v:",len(set(res['A4v'][0])&set(res['B4v'][0])))
