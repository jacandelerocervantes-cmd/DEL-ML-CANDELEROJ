"""exp25: novelty + solubility proxy for the exp19 lists A and B (test-split candidates). Run with a Python that has a working RDKit.
Known ligands used for novelty [V]: PGK2 cmp47 and PGK1 cmp45 SMILES recovered from data/reference/CACHE7_PDB.zip (DetermineBondOrders on the LIG chain A);
PGK2 cmp21 NOT recovered (no hydrogens, ambiguous bond orders). Also compared: the 39 real ASMS hits (AL) and the 19 validation hits.
Novelty = max Tanimoto (ECFP4) to each reference set + substructure flags for the known PGK2 chemotype (4-aminoquinazoline, 4-aminothienopyrimidine).
Solubility = ESOL (Delaney 2004) estimate of logS from RDKit descriptors; PROXY ONLY, not the organizers' criterion (unknown). Also PAINS flags."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, Descriptors, Crippen, rdMolDescriptors as D
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog('rdApp.*')
gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=lambda m:gen.GetFingerprint(m)
known={"cmp47 (PGK2)":"CNC(=O)C[C@H](Nc1nc(-c2cncc(OC)c2)nc2cc(OC)c(OC)cc12)c1ccc(C#N)cc1","cmp45 (PGK1)":"CNC(=O)C[C@@H](Nc1nc(-c2ccc(OC)c(S(=O)(=O)N3CCOCC3)c2)nc2cc(OC)c(OC)cc12)c1ccc(C#N)cc1"}
KM={k:fp(Chem.MolFromSmiles(s)) for k,s in known.items()}
al=pd.read_csv("data/raw/active_learning_data.csv"); AL=[fp(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES[al.Label==1]]
c=pd.read_csv("data/processed/confirmed_submission_outcomes_v2.csv"); v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
VH=[fp(Chem.MolFromSmiles(vs[i])) for i in c[c.confirmed_label==1].CatalogID if i in vs]
Q=Chem.MolFromSmarts('c1nc2ccccc2c([NX3])n1'); T=Chem.MolFromSmarts('c1nc2sccc2c([NX3])n1'); Q2=Chem.MolFromSmarts('c1nc(-[a])nc2ccccc12')
p=FilterCatalogParams(); p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS); pains=FilterCatalog(p)
def esol(m):
    logp=Crippen.MolLogP(m); mw=Descriptors.MolWt(m); rb=D.CalcNumRotatableBonds(m); ha=m.GetNumHeavyAtoms()
    ap=sum(a.GetIsAromatic() for a in m.GetAtoms())/ha; return 0.16-0.63*logp-0.0062*mw+0.066*rb-0.74*ap
rows=[]
for nm in "AB":
    d=pd.read_csv(f"data/processed/gp/exp19_selection_{nm}.csv"); sc2=[x for x in d.columns if 'smi' in x.lower()][0]
    for _,r in d.iterrows():
        m=Chem.MolFromSmiles(r[sc2]); f=fp(m)
        rows.append(dict(lista=nm,CatalogID=r.CatalogID,fam=r.get('fam',''),smiles=r[sc2],
            sim_cmp47=DataStructs.TanimotoSimilarity(f,KM["cmp47 (PGK2)"]),sim_cmp45=DataStructs.TanimotoSimilarity(f,KM["cmp45 (PGK1)"]),
            sim_hits_AL=max(DataStructs.BulkTanimotoSimilarity(f,AL)),sim_hits_val=max(DataStructs.BulkTanimotoSimilarity(f,VH)),
            aminoquinazolina=m.HasSubstructMatch(Q),aminotienopirimidina=m.HasSubstructMatch(T),arilquinazolina=m.HasSubstructMatch(Q2),
            logS_ESOL=esol(m),logP=Crippen.MolLogP(m),MW=Descriptors.MolWt(m),TPSA=D.CalcTPSA(m),PAINS=pains.HasMatch(m)))
R=pd.DataFrame(rows); R["sim_ligando_conocido"]=R[["sim_cmp47","sim_cmp45"]].max(axis=1)
R.to_csv("data/processed/gp/exp25_novelty_solubility.csv",index=False)
for nm,g in R.groupby("lista"):
    print(f"\nLista {nm} (n={len(g)})")
    print(f"  similitud max a ligandos conocidos de PGK2/PGK1 (cmp47, cmp45): mediana {g.sim_ligando_conocido.median():.2f}, max {g.sim_ligando_conocido.max():.2f}; >=0.4: {(g.sim_ligando_conocido>=0.4).sum()}; >=0.5: {(g.sim_ligando_conocido>=0.5).sum()}")
    print(f"  nucleo conocido: aminoquinazolina {int(g.aminoquinazolina.sum())}, aminotienopirimidina {int(g.aminotienopirimidina.sum())}, 2-arilquinazolina {int(g.arilquinazolina.sum())}")
    print(f"  similitud max a los 39 hits AL: mediana {g.sim_hits_AL.median():.2f}, max {g.sim_hits_AL.max():.2f}; a los 19 hits validation: mediana {g.sim_hits_val.median():.2f}")
    print(f"  ESOL logS: mediana {g.logS_ESOL.median():.2f}, min {g.logS_ESOL.min():.2f}; logS < -5: {(g.logS_ESOL<-5).sum()}; < -6: {(g.logS_ESOL<-6).sum()}; PAINS: {int(g.PAINS.sum())}")
print("\nlos 5 mas parecidos a ligandos conocidos:"); print(R.sort_values("sim_ligando_conocido",ascending=False)[["lista","CatalogID","sim_ligando_conocido","aminoquinazolina","smiles"]].head(5).to_string(index=False))
print("\nlos 8 menos solubles (ESOL):"); print(R.sort_values("logS_ESOL")[["lista","CatalogID","logS_ESOL","logP","MW"]].head(8).round(2).to_string(index=False))
