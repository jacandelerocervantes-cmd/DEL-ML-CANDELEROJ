"""exp14: motif-transfer probe on the VALIDATION split. Motif = 3-azolyl-anilide (SMARTS O=C[NH]c1cccc(-[c;r5])c1); in the 6,580 AL labels it has 7 hits/21.
Candidates: unsent validation compounds with the motif AND inside the hit property window; diverse sample (1 per Butina cluster, sim>=0.6),
chosen at RANDOM (seed 114), not by model score, so the motif -- not the model -- is what is tested. Output: submissions/Team_candeleroj_114.txt"""
import glob, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, Descriptors, Crippen, rdMolDescriptors as D
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
cols=['MW','logP','HBD','HBA','TPSA','aroR','rotB']
pr=lambda m:(Descriptors.MolWt(m),Crippen.MolLogP(m),D.CalcNumHBD(m),D.CalcNumHBA(m),D.CalcTPSA(m),D.CalcNumAromaticRings(m),D.CalcNumRotatableBonds(m))
al=pd.read_csv("data/raw/active_learning_data.csv"); h=al[al.Label==1]
hp=pd.DataFrame([pr(Chem.MolFromSmiles(s)) for s in h.RDKit_SMILES],columns=cols)
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[c for c in v.columns if 'smi' in c.lower()][0]
tested=set()
for f in glob.glob("submissions/Team_candeleroj_*.txt"):
    k=f.split('_')[-1][:-4]
    if k.isdigit() and 1<=int(k)<=113: tested|={l.strip() for l in open(f) if l.strip()}
v=v[~v.CatalogID.isin(tested)]; q=Chem.MolFromSmarts('O=C[NH]c1cccc(-[c;r5])c1')
ms=[Chem.MolFromSmiles(s) for s in v[sc]]; keep=[m.HasSubstructMatch(q) for m in ms]; v=v[keep].reset_index(drop=True); ms=[m for m,k in zip(ms,keep) if k]
P=pd.DataFrame([pr(m) for m in ms],columns=cols); ok=np.all([P[c].between(hp[c].min(),hp[c].max()) for c in cols],axis=0)
v=v[ok].reset_index(drop=True); ms=[m for m,k in zip(ms,ok) if k]; print("candidatos con motivo y propiedades:",len(v))
gen=G.GetMorganGenerator(radius=2,fpSize=2048); bv=[gen.GetFingerprint(m) for m in ms]; n=len(bv); dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
cl=Butina.ClusterData(dist,n,0.4,isDistData=True); print("clusters:",len(cl))
rng=np.random.default_rng(114); order=rng.permutation(len(cl))[:50]; pick=[int(rng.choice(list(cl[i]))) for i in order]
sel=v.iloc[pick]; assert len(sel)==50 and sel.CatalogID.is_unique
open("submissions/Team_candeleroj_114.txt","w").write("\n".join(sel.CatalogID)+"\n"); sel[['CatalogID',sc]].to_csv("data/processed/gp/exp14_motif_probe_detail.csv",index=False)
print(sel[['CatalogID',sc]].head(8).to_string(index=False))
