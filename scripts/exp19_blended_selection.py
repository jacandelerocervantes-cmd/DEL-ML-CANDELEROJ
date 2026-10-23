"""exp19: blend of 'top by model score' (exp16) and 'forced diversity' (exp18). Two disjoint lists of 50 for the test pool.
Candidates: top-1000 unlabelled by exp16 score (quality floor). Family quota = average of (share of that family among the 39 real hits) and (equal share over families).
Within a family: best-scoring candidate whose Butina cluster (sim>=0.5) and generic Murcko scaffold are new for the list. Unfilled quota is redistributed to the next best overall.
Output: data/processed/gp/exp19_selection_{A,B}.csv and exp19_pool_scores_{A,B}.csv (CatalogID, Sel_50, Score)."""
import numpy as np, pandas as pd, collections
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
S=pd.read_csv("data/processed/gp/exp16_pool_scores.csv"); P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
al=pd.read_csv("data/raw/active_learning_data.csv"); hf=pd.read_csv("data/processed/gp/exp17_hit_families.csv")
D=P[['CatalogID',sc]].merge(S[['CatalogID','Score']],on='CatalogID'); D=D[~D.CatalogID.isin(al.CatalogID)].sort_values('Score',ascending=False).head(1000).reset_index(drop=True)
ms=[Chem.MolFromSmiles(s) for s in D[sc]]
M={'otra amida heteroaril-NH-aril':'[a]C(=O)[NH][a]','azolil-anilida meta':'O=C[NH]c1cccc(-[c;r5])c1','indazol-7-carbonilo/amida':'O=C(N)c1cccc2cn[nH]c12','2-amino benzotiazol/benzoxazol':'c1nc2ccccc2[s,o]1','amida de quinolin-3/6-ilo':'O=C[NH]c1cnc2ccccc2c1','urea':'[NH]C(=O)[NH]'}
ORDER=['indazol-7-carbonilo/amida','azolil-anilida meta','otra amida heteroaril-NH-aril','2-amino benzotiazol/benzoxazol','amida de quinolin-3/6-ilo','urea']  # same priority as exp17 labelling (first match wins)
def famof(m):
    l=[k for k in M if m.HasSubstructMatch(Chem.MolFromSmarts(M[k]))]; return l[0] if l else 'otro'
D['fam']=[famof(m) for m in ms]
fams=sorted(set(hf.familia)|set(D.fam)); share=hf.familia.value_counts(normalize=True).reindex(fams).fillna(0); w=0.5*share+0.5/len(fams)
quota=(w/w.sum()*50); q=np.floor(quota).astype(int); 
for f in (quota-q).sort_values(ascending=False).index[:50-q.sum()]: q[f]+=1
print("cuota por familia:",q.to_dict()); print("candidatos en top-1000:",D.fam.value_counts().to_dict())
gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=[gen.GetFingerprint(m) for m in ms]; n=len(fp); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.5,isDistData=True)): cl[list(c)]=k
D['cl']=cl; D['gs']=[Chem.MolToSmiles(MS.MakeScaffoldGeneric(MS.GetScaffoldForMol(m))) for m in ms]
def pick(excl):
    ch=[];ucl=set();ugs=set()
    ok=lambda i:i not in excl and i not in ch and D.cl[i] not in ucl and D.gs[i] not in ugs
    for f in q.index:
        k=0
        for i in D.index[D.fam==f]:
            if k>=q[f]: break
            if ok(i): ch.append(i);ucl.add(D.cl[i]);ugs.add(D.gs[i]);k+=1
    for i in D.index:                       # relleno por puntaje si alguna cuota no se llenó
        if len(ch)>=50: break
        if ok(i): ch.append(i);ucl.add(D.cl[i]);ugs.add(D.gs[i])
    return ch
A=pick(set()); B=pick(set(A))
for nm,ch in (('A',A),('B',B)):
    sel=D.loc[ch]; print(f"\nLista {nm}: {len(sel)} | familias {sel.fam.value_counts().to_dict()} | clusters {sel.cl.nunique()} | andamiajes {sel.gs.nunique()} | posiciones en el ranking {sel.index.min()}-{sel.index.max()} (mediana {int(np.median(sel.index))})")
    sel[['CatalogID',sc,'Score','fam','cl']].to_csv(f"data/processed/gp/exp19_selection_{nm}.csv",index=False)
    F=S.copy(); F['Sel_50']=F.CatalogID.isin(sel.CatalogID).astype(int); F.to_csv(f"data/processed/gp/exp19_pool_scores_{nm}.csv",index=False)
print("solape A-B:",len(set(D.loc[A].CatalogID)&set(D.loc[B].CatalogID)))
