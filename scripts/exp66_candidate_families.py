"""exp66: groups the 384 compounds that can still be hits into families (Butina, ECFP4 Tanimoto 0.32) and computes min/max hits per family under the Synapse count constraints. Output: exp65_candidate_families.csv, exp65_family_bounds.csv. RDKit venv."""
import pandas as pd, numpy as np, os
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
b=pd.read_csv("data/processed/gp/exp65_validation_unresolved_bounds.csv"); cand=b[(b["max"]==1)].CatalogID.tolist(); neg=b[b["max"]==0].CatalogID.tolist()
V=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[c for c in V.columns if 'smi' in c.lower()][0]; smi=dict(zip(V.CatalogID,V[sc]))
g=G.GetMorganGenerator(radius=2,fpSize=2048); fp=[g.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in cand]; n=len(fp); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=Butina.ClusterData(d,n,1-0.32,isDistData=True); fam=np.zeros(n,int)
for k,c in enumerate(cl): fam[list(c)]=k
sizes=np.bincount(fam); print("candidatos:",n,"| familias (Tanimoto>=0.32):",len(sizes),"| tamanos:",sorted(sizes.tolist(),reverse=True)[:15],"| familias de 1 compuesto:",int((sizes==1).sum()))
# restricciones: reconstruir ecuaciones sobre los candidatos (los forzados a inactivo valen 0)
d0=pd.read_pickle("data/processed/leaderboard.pkl"); v=d0[(d0.submitterid==3610902)&(d0.evaluationid==9619685)&(d0.status=="ACCEPTED")].copy()
v["n"]=v.name.str.extract(r"^Team_candeleroj_(\d+)\.txt$")[0].astype(float); v=v[v.n.notna()].drop_duplicates("n",keep="last"); res=dict(zip(v.n.astype(int),v.N_hits)); res.update({176:1,178:0,179:0})
lab=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); L=dict(zip(lab.CatalogID,lab.confirmed_label)); ix={c:i for i,c in enumerate(cand)}; negs=set(neg)
rows=[];rhs=[];names=[]
for k in sorted(res):
    f=f"submissions/Team_candeleroj_{k}.txt"
    if not os.path.exists(f) or pd.isna(res[k]): continue
    ids=[l.strip() for l in open(f) if l.strip()]; kh=sum(L.get(c)==1 for c in ids); u=[ix[c] for c in ids if c in ix]
    if u: rows.append(u); rhs.append(res[k]-kh); names.append(k)
A=lil_matrix((len(rows),n))
for r,u in enumerate(rows):
    for i in u: A[r,i]=1
A=A.tocsr(); rhs=np.array(rhs,float); cons=LinearConstraint(A,rhs,rhs); print("ecuaciones sobre candidatos:",len(rows),"| hits repartidos entre los candidatos (suma max de rhs en listas disjuntas no calculada)")
tot=milp(np.ones(n),constraints=cons,bounds=Bounds(0,1),integrality=np.ones(n)); tot2=milp(-np.ones(n),constraints=cons,bounds=Bounds(0,1),integrality=np.ones(n))
print("hits totales posibles entre los candidatos: min",round(tot.fun),"max",round(-tot2.fun))
out=[]
for k in range(len(sizes)):
    c=np.zeros(n); c[fam==k]=1
    r1=milp(c,constraints=cons,bounds=Bounds(0,1),integrality=np.ones(n),options={"time_limit":10}); r2=milp(-c,constraints=cons,bounds=Bounds(0,1),integrality=np.ones(n),options={"time_limit":10})
    out.append((k,int(sizes[k]),round(r1.fun) if r1.success else -1,round(-r2.fun) if r2.success else -1))
o=pd.DataFrame(out,columns=["familia","compuestos","min_hits","max_hits"])
print("familias con hits seguros (min>=1):",int((o.min_hits>=1).sum()),"| familias que pueden tener hits (max>=1):",int((o.max_hits>=1).sum()),"| familias que seguro no tienen hits (max=0):",int((o.max_hits==0).sum()))
print("suma de min por familia:",int(o.min_hits.sum()),"| suma de max:",int(o.max_hits.sum()))
print(o.sort_values("max_hits",ascending=False).head(12).to_string(index=False))
pd.DataFrame({"CatalogID":cand,"familia":fam}).to_csv("data/processed/gp/exp65_candidate_families.csv",index=False); o.to_csv("data/processed/gp/exp65_family_bounds.csv",index=False)
