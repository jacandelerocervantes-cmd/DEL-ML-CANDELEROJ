"""exp70: bounds on the number of Tanimoto-0.32 families that contain hits among the 384 candidate compounds, under all Synapse count constraints (lists 156-181). MILP with scipy: min and max of the number of families with at least one hit. Needs the exp65/exp66 outputs. Not published (validation labels)."""
import pandas as pd, numpy as np, os
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, hstack, vstack, csr_matrix
d0=pd.read_pickle("data/processed/leaderboard.pkl"); v=d0[(d0.submitterid==3610902)&(d0.evaluationid==9619685)&(d0.status=="ACCEPTED")].copy()
v["n"]=v.name.str.extract(r"^Team_candeleroj_(\d+)\.txt$")[0].astype(float); v=v[v.n.notna()].drop_duplicates("n",keep="last"); res=dict(zip(v.n.astype(int),v.N_hits)); res.update({176:1,178:0,179:0,180:2,181:1})
lab=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); L=dict(zip(lab.CatalogID,lab.confirmed_label))
cf=pd.read_csv("data/processed/gp/exp65_candidate_families.csv"); cand=cf.CatalogID.tolist(); fam=cf.familia.values; ix={c:i for i,c in enumerate(cand)}; n=len(cand); F=fam.max()+1
b=pd.read_csv("data/processed/gp/exp65_validation_unresolved_bounds.csv"); neg=set(b[b["max"]==0].CatalogID)
rows=[];rhs=[]
for k in sorted(res):
    f=f"submissions/Team_candeleroj_{k}.txt"
    if not os.path.exists(f) or pd.isna(res[k]): continue
    ids=[l.strip() for l in open(f) if l.strip()]; kh=sum(L.get(c)==1 for c in ids); u=[ix[c] for c in ids if c in ix]
    if u: rows.append(u); rhs.append(res[k]-kh)
A=lil_matrix((len(rows),n+F))
for r,u in enumerate(rows):
    for i in u: A[r,i]=1
A=A.tocsr(); rhs=np.array(rhs,float)
# z_f <= sum_{i in f} x_i   ->  z_f - sum x_i <= 0
G=lil_matrix((F,n+F))
for i in range(n): G[fam[i],i]=-1
for f in range(F): G[f,n+f]=1
G=G.tocsr()
cons=[LinearConstraint(A,rhs,rhs),LinearConstraint(G,-np.inf,0)]
c=np.r_[np.zeros(n),np.ones(F)]
lo=Bounds(np.zeros(n+F),np.ones(n+F)); integ=np.ones(n+F)
mx=milp(-c,constraints=cons,bounds=lo,integrality=integ,options={"time_limit":120}); 
# minimo: z_f >= x_i para cada i en f -> z_f - x_i >= 0
H=lil_matrix((n,n+F))
for i in range(n): H[i,i]=-1; H[i,n+fam[i]]=1
cons2=[LinearConstraint(A,rhs,rhs),LinearConstraint(H.tocsr(),0,np.inf)]
mn=milp(c,constraints=cons2,bounds=lo,integrality=integ,options={"time_limit":120})
print("ecuaciones:",len(rows),"| familias candidatas:",F)
print("numero de familias (Tanimoto 0.32) con hits entre los candidatos: minimo",round(mn.fun),"| maximo",round(-mx.fun))
