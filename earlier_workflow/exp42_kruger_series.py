"""exp42: the OFFICIAL series algorithm (Kruger et al., as quoted by the organizers):
 dendrogram of the HITS from ECFP4 Tanimoto distance, AVERAGE linkage; breadth-first through the dendrogram from the root; at each cluster compute the MCS of its members;
 compare the MCS with the full ASMS library (here: validation+test splits = 428,960 cpds); if found in < 100 molecules, that cluster is a SERIES described by the MCS; else descend.
Validation: series computed once on the 32 validation hits, then the number of DISTINCT series among the hits of each past submission is compared with Synapse's official cluster count.
Run with the RDKit venv. Outputs data/processed/gp/exp42_*.csv"""
import time, numpy as np, pandas as pd
from collections import deque
from scipy.cluster.hierarchy import linkage, to_tree
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, rdFMCS
RDLogger.DisableLog('rdApp.*')
t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:5.0f}s]",*a,flush=True)
V=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); T=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sv=[x for x in V.columns if 'smi' in x.lower()][0]
lib=list(V[sv])+list(T[[x for x in T.columns if 'smi' in x.lower()][0]]); smi={**dict(zip(V.CatalogID,V[sv])),**dict(zip(T.CatalogID,T[[x for x in T.columns if 'smi' in x.lower()][0]]))}
libmols=[m for m in (Chem.MolFromSmiles(s) for s in lib) if m is not None]; log("biblioteca parseada:",len(libmols))
gen=G.GetMorganGenerator(radius=2,fpSize=2048)
def count_matches(q,cap=100):
    n=0
    for m in libmols:
        if m.HasSubstructMatch(q):
            n+=1
            if n>=cap: return n
    return n
def series_of(ids,min_atoms=6,cap=100,mcs_timeout=8):
    mols=[Chem.MolFromSmiles(smi[c]) for c in ids]; fps=[gen.GetFingerprint(m) for m in mols]; n=len(ids)
    if n==1: return [(list(range(1)),"singleton")]
    D=np.array([[1-DataStructs.TanimotoSimilarity(a,b) for b in fps] for a in fps]); iu=np.triu_indices(n,1)
    root=to_tree(linkage(D[iu],method="average")); out=[]; q=deque([root])
    while q:
        node=q.popleft(); members=node.pre_order()
        if len(members)==1: out.append((members,"singleton")); continue
        r=rdFMCS.FindMCS([mols[i] for i in members],timeout=mcs_timeout,ringMatchesRingOnly=True,completeRingsOnly=True)
        ok=False
        if r.numAtoms>=min_atoms:
            qm=Chem.MolFromSmarts(r.smartsString); c=count_matches(qm,cap); ok=c<cap
        if ok: out.append((members,r.smartsString))
        else: q.append(node.get_left()); q.append(node.get_right())
    return out
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=sorted(o[o.confirmed_label==1].CatalogID)
S=series_of(VH); log("series oficiales (algoritmo Kruger) entre los",len(VH),"hits de validation:",len(S))
lab={}
for k,(mem,sm) in enumerate(S):
    for i in mem: lab[VH[i]]=k
rows=[dict(serie=k,hits=len(mem),mcs=sm,ids=";".join(VH[i] for i in mem)) for k,(mem,sm) in enumerate(S)]
pd.DataFrame(rows).sort_values("hits",ascending=False).to_csv("data/processed/gp/exp42_validation_series.csv",index=False)
print(pd.DataFrame(rows).sort_values("hits",ascending=False)[["serie","hits","mcs"]].to_string(index=False,max_colwidth=70))
OFF={93:(8,1),94:(18,8),95:(17,6),96:(14,4),97:(13,3),98:(13,3),99:(19,8),100:(11,8),112:(5,5),131:(4,3),132:(2,1),138:(2,1),153:(9,6),154:(18,15),155:(32,15)}
print("\nenvio | hits | oficial | Kruger (series distintas entre sus hits)"); err=0
for e,(h,c) in OFF.items():
    L=[l.strip() for l in open(f"submissions/Team_candeleroj_{e}.txt") if l.strip()]; H=[x for x in L if x in lab]; k=len({lab[x] for x in H}); err+=abs(k-c)
    print(f"{e:5d} | {len(H):3d} | {c:3d} | {k:3d} {'OK' if k==c else ''}")
print("error absoluto total:",err)
