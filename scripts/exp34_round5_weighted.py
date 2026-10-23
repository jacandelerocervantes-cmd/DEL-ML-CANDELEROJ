"""exp34: round 5. fenilo: uniform max-entropy pool (exp31). sueltos: max-entropy pool under SOFT weights that favour hit sets whose two hits inside pool 142
share more structure (envio 142 = 2 hits / 1 cluster => same series). weight = exp(0.7*(MCS_atoms-10)) clipped to [1,40]; no hypothesis is excluded."""
import numpy as np, pandas as pd
exec(open("scripts/exp33_same_series_pairs.py",encoding="utf8").read().split("g=\"sueltos\"")[0])
T=pd.read_csv("data/processed/gp/exp33_sueltos_pairs.csv"); W={tuple(sorted((IDS["sueltos"].index(a),IDS["sueltos"].index(b)))):float(np.clip(np.exp(0.7*(m-10)),1,40)) for a,b,m in zip(T.A,T.B,T.atomos_MCS)}
g="sueltos"; ids=IDS[g]; n=len(ids); c=hyps(g,history_from_files(g)); Pl=pd.read_csv(PLAN); pool142=set(Pl[Pl.envio==142].miembros.iloc[0].split(";")); inpool=np.array([x in pool142 for x in ids])
w=np.array([W.get(tuple(sorted(i for i in h if inpool[i])),1.0) for h in c]); w/=w.sum()
def Hw(counts,w):
    p=np.bincount(counts,weights=w); p=p[p>0]; return -(p*np.log2(p)).sum()
rng=np.random.default_rng(34); best=None;be=-1
for _ in range(2500):
    s=rng.integers(max(1,n//6),n//2+2); p=np.zeros(n,bool); p[rng.choice(n,s,replace=False)]=True; e=Hw(p[c].sum(1),w)
    if e>be: be,best=e,p
pool=[x for x,b in zip(ids,best) if b]; print(f"sueltos: {len(c)} combinaciones; entropia ponderada del pool {be:.2f} bits ({len(pool)} compuestos); sin pesos uniformes seria ~{Hw(best[c].sum(1),np.ones(len(c))/len(c)):.2f}")
# fenilo uniforme
gf="fenilo"; cf=hyps(gf,history_from_files(gf)); nf=len(IDS[gf]); pf,ef=best_pool(cf,nf,rng); poolf=[x for x,b in zip(IDS[gf],pf) if b]
nxt=max(Pl.envio)+1; rows=[]
for gg,pl,e in ((gf,poolf,ef),(g,pool,be)):
    pad=list(rng.choice([x for x in NEG if x not in pl],50-len(pl),replace=False)); L=pl+pad; assert len(set(L))==50
    path=f"submissions/Team_candeleroj_{nxt}.txt"; assert not os.path.exists(path); open(path,"w").write("\n".join(L)+"\n")
    print(f"  envio {nxt}: {gg} | {len(pl)} compuestos del grupo | entropia {e:.2f} bits"); rows.append(dict(envio=nxt,grupo=gg,n_en_pool=len(pl),miembros=";".join(pl))); nxt+=1
pd.concat([Pl,pd.DataFrame(rows)],ignore_index=True).to_csv(PLAN,index=False)
