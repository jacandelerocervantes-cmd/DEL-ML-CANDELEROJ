"""exp10: which test-pool compounds can be excluded a priori? Property ranges of the 39 real ASMS hits vs the 6,541 labelled
negatives and vs the whole test pool. Light RDKit properties only. Output: data/processed/gp/exp10_pool_props.parquet"""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as D
RDLogger.DisableLog('rdApp.*')
anil=Chem.MolFromSmarts('C(=O)[NH]c'); 
def props(s):
    m=Chem.MolFromSmiles(s)
    return (Descriptors.MolWt(m),Crippen.MolLogP(m),D.CalcNumHBD(m),D.CalcNumHBA(m),D.CalcTPSA(m),D.CalcNumAromaticRings(m),D.CalcNumRotatableBonds(m),m.HasSubstructMatch(anil))
cols=['MW','logP','HBD','HBA','TPSA','aroR','rotB','anilide']
al=pd.read_csv("data/raw/active_learning_data.csv"); al[cols]=[props(s) for s in al.RDKit_SMILES]
t=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in t.columns if 'smi' in c.lower()][0]
t[cols]=[props(s) for s in t[sc]]; t.to_parquet("data/processed/gp/exp10_pool_props.parquet")
h=al[al.Label==1]
print("pool",len(t)); print(pd.DataFrame({'hits_min':h[cols[:-1]].min(),'hits_p5':h[cols[:-1]].quantile(.05),'hits_p95':h[cols[:-1]].quantile(.95),'hits_max':h[cols[:-1]].max(),'pool_med':t[cols[:-1]].median(),'neg_med':al[al.Label==0][cols[:-1]].median()}).round(1))
for nm,cond in {'aroR>=2':lambda d:d.aroR>=2,'aroR>=2 & HBD>=1':lambda d:(d.aroR>=2)&(d.HBD>=1),'aroR>=2 & HBD 1-3 & MW 250-450':lambda d:(d.aroR>=2)&d.HBD.between(1,3)&d.MW.between(250,450),'anilida':lambda d:d.anilide}.items():
    print(f"{nm:35s} hits que pasan {int(cond(h).sum())}/39 | pool que pasa {cond(t).mean():.1%} ({int(cond(t).sum())})")
