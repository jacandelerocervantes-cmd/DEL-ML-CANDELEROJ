"""exp35 (idea tomada del informe CASMI: mascara de analogos / identidad molecular canonica):
(1) duplicados estructurales por InChIKey14 entre el pool de test y los 6,580 etiquetados;  (2) analogos no etiquetados de los 39 hits (Tanimoto ECFP4);
(3) cuantos de esos analogos ya estan en las listas exp19 A y B. Run with the RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
al=pd.read_csv("data/raw/active_learning_data.csv"); P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
ik=lambda s:(MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14]
al["ik"]=[ik(s) for s in al.RDKit_SMILES]; P["ik"]=[ik(s) for s in P[sc]]
print("InChIKey14 unicos: AL",al.ik.nunique(),"de",len(al),"| pool",P.ik.nunique(),"de",len(P))
U=P[~P.CatalogID.isin(al.CatalogID)]; dup=U[U.ik.isin(set(al.ik))]
m=dict(zip(al.ik,al.Label)); print("compuestos SIN etiqueta del pool con la MISMA estructura (InChIKey14) que uno etiquetado:",len(dup),"| de ellos igual a un HIT:",int(sum(m[k]==1 for k in dup.ik)),"| igual a un negativo:",int(sum(m[k]==0 for k in dup.ik)))
print("duplicados internos del pool (misma estructura, distinto ID):",len(P)-P.ik.nunique())
gen=G.GetMorganGenerator(radius=2,fpSize=2048); HF=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES[al.Label==1]]
U=U.copy(); U["simhit"]=[max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),HF)) for s in U[sc]]
for t in (0.5,0.6,0.7): print(f"compuestos sin etiqueta con Tanimoto >= {t} a algun hit conocido: {(U.simhit>=t).sum()}")
L=set(pd.read_csv("data/processed/gp/exp19_selection_A.csv").CatalogID)|set(pd.read_csv("data/processed/gp/exp19_selection_B.csv").CatalogID)
print("de los analogos >=0.6, ya estan en las listas A/B:",int(U[U.simhit>=0.6].CatalogID.isin(L).sum()),"de",int((U.simhit>=0.6).sum()))
U.sort_values("simhit",ascending=False).head(15)[["CatalogID","simhit",sc]].to_csv("data/processed/gp/exp35_top_analogs.csv",index=False)
print(U.sort_values("simhit",ascending=False).head(6)[["CatalogID","simhit",sc]].round(2).to_string(index=False))
