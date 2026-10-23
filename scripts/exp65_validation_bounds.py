"""exp65: which compounds of the tested validation lists are already forced to be inactive or hit by the Synapse counts? MILP bounds per unresolved compound over all lists with a result. Output: data/processed/gp/exp65_validation_unresolved_bounds.csv. RDKit-free; scipy only."""
import pandas as pd, numpy as np, re, os
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
d=pd.read_pickle("data/processed/leaderboard.pkl")
v=d[(d.submitterid==3610902)&(d.evaluationid==9619685)&(d.status=="ACCEPTED")].copy()
v["n"]=v.name.str.extract(r"^Team_candeleroj_(\d+)\.txt$")[0].astype(float); v=v[v.n.notna()].drop_duplicates("n",keep="last")
res=dict(zip(v.n.astype(int),v.N_hits)); res.update({176:1,178:0,179:0})
lab=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); L=dict(zip(lab.CatalogID,lab.confirmed_label))
lists={}
for n in sorted(res):
    f=f"submissions/Team_candeleroj_{n}.txt"
    if os.path.exists(f) and not pd.isna(res[n]): lists[n]=[l.strip() for l in open(f) if l.strip()]
unk=sorted({c for ids in lists.values() for c in ids if c not in L}); ix={c:i for i,c in enumerate(unk)}
print("envios con resultado:",len(lists),"| compuestos distintos sin identificar en esas listas:",len(unk))
rows=[];rhs=[]
for n,ids in lists.items():
    kh=sum(L.get(c)==1 for c in ids); u=[ix[c] for c in ids if c not in L]
    if u: rows.append(u); rhs.append(res[n]-kh)
A=lil_matrix((len(rows),len(unk)))
for r,u in enumerate(rows):
    for i in u: A[r,i]=1
A=A.tocsr(); rhs=np.array(rhs,float); print("ecuaciones:",len(rows),"| rhs total no negativo:",bool((rhs>=0).all()))
cons=LinearConstraint(A,rhs,rhs); bnd=Bounds(0,1); integ=np.ones(len(unk))
mn=np.zeros(len(unk)); mx=np.zeros(len(unk))
for i in range(len(unk)):
    c=np.zeros(len(unk)); c[i]=1
    r1=milp(c,constraints=cons,bounds=bnd,integrality=integ,options={"time_limit":10}); r2=milp(-c,constraints=cons,bounds=bnd,integrality=integ,options={"time_limit":10})
    mn[i]=round(r1.fun) if r1.success else np.nan; mx[i]=round(-r2.fun) if r2.success else np.nan
ok=~np.isnan(mn)&~np.isnan(mx)
print("compuestos con cota resuelta:",int(ok.sum()),"de",len(unk))
print("forzados a hit (min=1):",int((mn[ok]==1).sum()),"| forzados a inactivo (max=0):",int((mx[ok]==0).sum()),"| indeterminados:",int(((mn[ok]==0)&(mx[ok]==1)).sum()))
pd.DataFrame({"CatalogID":unk,"min":mn,"max":mx}).to_csv("data/processed/gp/exp65_validation_unresolved_bounds.csv",index=False)
