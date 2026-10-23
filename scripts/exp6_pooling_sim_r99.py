"""Adaptive group-testing simulation: find 5 hits among the 36 unidentified compounds of round 99.
Each query = subset of the 36 (padded to 50 with confirmed negatives); answer = number of hits in subset.
Greedy: choose pool maximizing entropy of the answer over still-consistent 5-subsets. Seeded."""
import itertools, numpy as np
rng=np.random.default_rng(0)
N,K=36,5
combos=np.array(list(itertools.combinations(range(N),K)),dtype=np.int8)  # 376,992 x 5
def count(pool,cs): return pool[cs].sum(1)
def entropy(c):
    p=np.bincount(c)/len(c); p=p[p>0]; return -(p*np.log2(p)).sum()
def best_pool(cs,tries=300):
    best=None;be=-1
    for _ in range(tries):
        s=rng.integers(max(1,N//4),N//2+3)
        pool=np.zeros(N,bool); pool[rng.choice(N,s,replace=False)]=True
        e=entropy(count(pool,cs))
        if e>be: be,best=e,pool
    return best,be
res=[]
for t in range(20):
    truth=combos[rng.integers(len(combos))]; cs=combos; q=0
    while len(cs)>1:
        pool,e=best_pool(cs); ans=pool[truth].sum(); cs=cs[count(pool,cs)==ans]; q+=1
        if q==1 and t==0: print("1er pool: tamaño",pool.sum(),"entropía",round(e,2),"bits")
    res.append(q)
print("candidatos iniciales",len(combos),"log2",round(np.log2(len(combos)),1))
print("envíos hasta identificar los 5 (20 simulaciones):",res,"media",np.mean(res),"max",max(res))
