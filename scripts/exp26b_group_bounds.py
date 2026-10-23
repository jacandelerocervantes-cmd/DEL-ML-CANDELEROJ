import ast, os, numpy as np, pandas as pd
exec(open("scripts/exp26_group_unidentified.py",encoding="utf8").read().split("cs=pd.read_csv")[0])   # reutiliza el solver y las restricciones
D=pd.read_csv("data/processed/gp/exp26_unidentified_families.csv")
def bounds(idx):
    c=np.zeros(len(unk)); c[idx]=1; return (int(round(float(c@solve(c)))),int(round(float(c@solve(-c)))))
print("=== por MOTIVO quimico (194 inciertos)")
for m,g in D.groupby("motivo"):
    mn,mx=bounds([ix[c] for c in g.CatalogID]); print(f"  {m:24s} {len(g):3d} compuestos | hits posibles: {mn} a {mx}")
dd=pd.read_csv("docs/DATOS_D_hits_sin_identificar.csv"); gm=dict(zip(dd.CatalogID,dd.grupo))
D["grupo_diseno"]=D.CatalogID.map(gm).fillna("(fuera de DATOS_D)")
print("\n=== por GRUPO DE DISENO original (DATOS_D, lo que se envio junto)")
for m,g in D.groupby("grupo_diseno"):
    mn,mx=bounds([ix[c] for c in g.CatalogID]); print(f"  {m:26s} {len(g):3d} compuestos | hits posibles: {mn} a {mx}")
print("\n=== por RONDA en la que aparecieron (las que aun tienen hits sin identificar)")
for (nres,u),f in zip(rows,[f for f,n in R.items() if os.path.exists('submissions/'+f) and any(c not in lab or pd.isna(lab[c]) for c in [l.strip() for l in open('submissions/'+f) if l.strip()])]):
    uu=[c for c in u if c in set(D.CatalogID)]
    print(f"  {f.replace('Team_candeleroj_','r').replace('.txt',''):6s} hits sin identificar en la ronda: {nres} | inciertos en ella: {len(uu)}")
D.to_csv("data/processed/gp/exp26_unidentified_families.csv",index=False)
