"""Constraint solver for unidentified Validation hits (PGK2).
Each real round with N_hits gives: sum(x_i over round compounds with unknown label) = N - (known hits in round).
Reports forced-hit / forced-negative compounds and min/max total hits. Read-only; writes data/processed/gp/exp6_*.csv"""
import ast, os, numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
src=open("scripts/build_confirmed_outcomes.py",encoding="utf8").read()
i=src.index("ROUND_OUTCOMES = {"); j=src.index("\n}",i)
R=ast.literal_eval(src[i+len("ROUND_OUTCOMES = "):j+2])
R.update({"Team_candeleroj_101.txt":3,"Team_candeleroj_102.txt":2,"Team_candeleroj_103.txt":0,"Team_candeleroj_112.txt":5,"Team_candeleroj_116.txt":0, "Team_candeleroj_117.txt":0, "Team_candeleroj_118.txt":0, "Team_candeleroj_119.txt":0, "Team_candeleroj_120.txt":0, "Team_candeleroj_121.txt":1, "Team_candeleroj_122.txt":0, "Team_candeleroj_123.txt":1, "Team_candeleroj_124.txt":1, "Team_candeleroj_125.txt":0, "Team_candeleroj_126.txt":0, "Team_candeleroj_127.txt":1, "Team_candeleroj_128.txt":1, "Team_candeleroj_129.txt":1, "Team_candeleroj_130.txt":0, "Team_candeleroj_131.txt":4})  # results pasted by user 2026-10-02
o=pd.read_csv(os.environ.get("OUTCOMES","data/processed/confirmed_submission_outcomes.csv"))
lab=dict(zip(o.CatalogID,o.confirmed_label))
rows=[]; unk=set()
for f,n in R.items():
    p="submissions/"+f
    if not os.path.exists(p): continue
    ids=[l.strip() for l in open(p) if l.strip()]
    if len(ids)!=50: print("WARN",f,len(ids))
    known=sum(lab.get(c)==1 for c in ids)
    u=[c for c in ids if c not in lab or pd.isna(lab[c])]
    unk|=set(u); rows.append((f,n,known,u))
unk=sorted(unk); ix={c:k for k,c in enumerate(unk)}
print("variables",len(unk),"not in table:",sum(c not in lab for c in unk))
A=[];lo=[]
for f,n,k,u in rows:
    if not u:
        if n!=k: print("INCONSISTENT",f,n,k)
        continue
    r=np.zeros(len(unk)); r[[ix[c] for c in u]]=1; A.append(r); lo.append(n-k)
A=np.array(A); lo=np.array(lo,float)
cons=LinearConstraint(A,lo,lo); bnd=Bounds(0,1); integ=np.ones(len(unk))
def solve(c):
    r=milp(c,constraints=cons,bounds=bnd,integrality=integ); return None if r.status!=0 else r.x
t=np.ones(len(unk)); a=solve(t); b=solve(-t)
print("feasible:",a is not None)
print("min total hits",int(round(a.sum())),"max total hits",int(round(b.sum())))
res=[]
for c,k in ix.items():
    e=np.zeros(len(unk)); e[k]=1
    mn=solve(e); mx=solve(-e); res.append((c,int(round(mn[k])),int(round(mx[k]))))
d=pd.DataFrame(res,columns=['CatalogID','min','max'])
d['estado']=np.where(d['min']==1,'HIT_FORZADO',np.where(d['max']==0,'NEG_FORZADO','incierto'))
print(d.estado.value_counts().to_dict())
r99=set(next(u for f,n,k,u in rows if f=="Team_candeleroj_99.txt"))
x=d[d.CatalogID.isin(r99)]; print("ronda 99 (36):",x.estado.value_counts().to_dict())
os.makedirs("data/processed/gp",exist_ok=True); d.to_csv("data/processed/gp/exp6_unidentified_constraints.csv",index=False)
print(d[d.estado=='HIT_FORZADO'].CatalogID.tolist())
