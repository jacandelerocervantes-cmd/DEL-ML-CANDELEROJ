import numpy as np, pandas as pd
from math import lgamma, log
exec(open("scripts/exp26_group_unidentified.py",encoding="utf8").read().split("cs=pd.read_csv")[0])
def bounds(idx):
    c=np.zeros(len(unk)); c[idx]=1; return (int(round(float(c@solve(c)))),int(round(float(c@solve(-c)))))
lc=lambda n,k:(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1))/log(2)
cs=pd.read_csv("data/processed/gp/exp6_unidentified_constraints.csv"); unc=set(cs[cs.estado=="incierto"].CatalogID)
D=pd.read_csv("data/processed/gp/exp26_unidentified_families.csv"); D=D[D.CatalogID.isin(unc)]
print("inciertos:",len(unc),"| de ellos en grupos del analisis:",len(D))
tot=0
for g,sub in D.groupby("grupo_diseno"):
    mn,mx=bounds([ix[c] for c in sub.CatalogID]); n=len(sub); b=lc(n,mn) if mn==mx else float('nan'); tot+=0 if mn!=mx else b
    print(f"  {g:26s} {n:3d} candidatos | hits {mn}-{mx} | bits {b:.1f}")
print(f"suma de bits (grupos con conteo exacto): {tot:.1f}")
fuera=[c for c in unc if c not in set(D.CatalogID)]; print("inciertos fuera de esos grupos:",len(fuera))
