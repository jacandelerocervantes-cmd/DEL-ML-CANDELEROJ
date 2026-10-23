"""exp30: adaptive greedy (max-entropy pool) simulation, one group at a time with exact enumeration of the consistent hit sets.
Groups (exp29): sueltos 4/57, fenilo 3/38, quinoxalina 2/15. Seeded. Reports submissions needed to pin the hit set (count answers, no noise)."""
import itertools, numpy as np
rng=np.random.default_rng(30)
def H(c):
    p=np.bincount(c)/len(c); p=p[p>0]; return -(p*np.log2(p)).sum()
def sim(n,k,reps,tries=250):
    cs=np.array(list(itertools.combinations(range(n),k)),dtype=np.int8); out=[]
    for _ in range(reps):
        truth=cs[rng.integers(len(cs))]; c=cs; q=0
        while len(c)>1:
            best=None;be=-1
            for _ in range(tries):
                s=rng.integers(max(1,n//6),n//2+2); p=np.zeros(n,bool); p[rng.choice(n,s,replace=False)]=True
                e=H(p[c].sum(1))
                if e>be: be,best=e,p
            c=c[best[c].sum(1)==best[truth].sum()]; q+=1
        out.append(q)
    return out
tot=0
for nm,n,k,reps in (("quinoxalina",15,2,30),("fenilo",38,3,15),("sueltos",57,4,4)):
    r=sim(n,k,reps); tot+=np.mean(r); print(f"{nm:12s} {k} hits en {n}: envios {np.mean(r):.1f} (min {min(r)}, max {max(r)}) sobre {reps} simulaciones",flush=True)
print(f"TOTAL por separado, adaptivo: ~{tot:.0f} envios")
