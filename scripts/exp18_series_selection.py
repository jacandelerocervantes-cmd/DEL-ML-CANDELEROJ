"""exp18: two disjoint 50-compound selections for the test pool that maximise the number of DISTINCT SERIES (ranking criterion #1), not the model score.
Candidates: top-3000 unlabelled compounds by exp16 score. Each is assigned to a motif family (same SMARTS as exp17) and a Butina cluster (sim>=0.5) + generic Murcko scaffold.
Greedy round-robin over families (cap 14 per family per list): at each turn take the best-scoring candidate whose cluster AND generic scaffold are new for that list.
List A = first pass; list B = second pass on what is left (disjoint). Output: data/processed/gp/exp18_selection_{A,B}.csv (+ full-pool score files in submission format)."""
import numpy as np, pandas as pd, collections
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
S=pd.read_csv("data/processed/gp/exp16_pool_scores.csv"); P=pd.read_parquet("data/processed/gp/exp10_pool_props.parquet"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
al=pd.read_csv("data/raw/active_learning_data.csv")
D=P[['CatalogID',sc]].merge(S[['CatalogID','Score']],on='CatalogID'); D=D[~D.CatalogID.isin(al.CatalogID)].sort_values('Score',ascending=False).head(3000).reset_index(drop=True)
ms=[Chem.MolFromSmiles(s) for s in D[sc]]
M={'indazol-7-carbonilo':'O=C(N)c1cccc2cn[nH]c12','azolil-anilida meta':'O=C[NH]c1cccc(-[c;r5])c1','amida heteroaril-NH-aril':'[a]C(=O)[NH][a]','2-amino benzotiazol/oxazol':'c1nc2ccccc2[s,o]1','amida quinolin-3/6-ilo':'O=C[NH]c1cnc2ccccc2c1','urea':'[NH]C(=O)[NH]'}
fam=[]
for m in ms:
    l=[k for k,s in M.items() if m.HasSubstructMatch(Chem.MolFromSmarts(s))]; fam.append(l[0] if l else 'otro')
D['fam']=fam; gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=[gen.GetFingerprint(m) for m in ms]; n=len(fp); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.5,isDistData=True)): cl[list(c)]=k
D['cl']=cl; D['gs']=[Chem.MolToSmiles(MS.MakeScaffoldGeneric(MS.GetScaffoldForMol(m))) for m in ms]
print("candidatos por familia:",D.fam.value_counts().to_dict(),"| clusters:",D.cl.nunique())
def pick(D,exclude,cap=14):
    chosen=[];used_cl=set();used_gs=set();cnt=collections.Counter(); fams=list(D.fam.value_counts().index)
    while len(chosen)<50:
        prog=False
        for f in fams:
            if len(chosen)>=50: break
            if cnt[f]>=cap: continue
            for i,r in D[(D.fam==f)].iterrows():
                if i in exclude or i in chosen or r.cl in used_cl or r.gs in used_gs: continue
                chosen.append(i);used_cl.add(r.cl);used_gs.add(r.gs);cnt[f]+=1;prog=True;break
        if not prog: cap+=4
    return chosen
A=pick(D,set()); B=pick(D,set(A))
for nm,ch in (('A',A),('B',B)):
    sel=D.loc[ch]; print(f"\nLista {nm}: {len(sel)} compuestos | familias {sel.fam.value_counts().to_dict()} | clusters {sel.cl.nunique()} | andamientos genericos {sel.gs.nunique()} | rango de puntaje en el ranking {sel.index.min()}-{sel.index.max()}")
    sel[['CatalogID',sc,'Score','fam','cl']].to_csv(f"data/processed/gp/exp18_selection_{nm}.csv",index=False)
    F=S.copy(); F['Sel_50']=F.CatalogID.isin(sel.CatalogID).astype(int); F.to_csv(f"data/processed/gp/exp18_pool_scores_{nm}.csv",index=False)
print("solape A-B:",len(set(D.loc[A].CatalogID)&set(D.loc[B].CatalogID)))
