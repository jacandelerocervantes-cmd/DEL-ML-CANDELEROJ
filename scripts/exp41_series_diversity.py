"""exp41: the official 'series' count among the 50 picks. Threshold 0.32 (Tanimoto ECFP4) was calibrated on Synapse's official cluster counts (exp36).
(1) how many series (Butina 0.32) do the current lists have inside their 50 picks? (2) rebuild A3/B3 so no two picks share a series (pairwise Tanimoto < 0.32). Run with the RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
gen=G.GetMorganGenerator(radius=2,fpSize=2048); T=0.32
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
al=pd.read_csv("data/raw/active_learning_data.csv"); lab=set(al.CatalogID)
fpc={}; fp=lambda c:fpc.setdefault(c,gen.GetFingerprint(Chem.MolFromSmiles(smi[c])))
def nser(ids):
    f=[fp(c) for c in ids]; n=len(f); d=[]
    for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(f[i],f[:i])]
    return len(Butina.ClusterData(d,n,1-T,isDistData=True))
cur={"A":"exp19_selection_A","B":"exp19_selection_B","A2":"exp38_selection_A2","B2":"exp38_selection_B2"}
for k,fn in cur.items():
    ids=list(pd.read_csv(f"data/processed/gp/{fn}.csv").CatalogID); print(f"lista {k}: {len(ids)} compuestos -> {nser(ids)} series distintas (umbral {T}); maximo posible 50")
# candidatos: top por puntaje (sin etiqueta) + analogos
sc16=pd.read_csv("data/processed/gp/exp16_pool_scores.csv"); sc16=sc16[~sc16.CatalogID.isin(lab)].sort_values("Score",ascending=False)
topc=list(sc16.CatalogID.head(6000)); ana={nm:list(pd.read_csv(f"data/processed/gp/exp38_selection_{nm}.csv").query("origen=='analogo_serie'").CatalogID) for nm in ("A2","B2")}
used=set()
for nm,out in (("A2","A3"),("B2","B3")):
    pick=[]
    def ok(c): return all(DataStructs.TanimotoSimilarity(fp(c),fp(p))<T for p in pick)
    for c in ana[nm]:
        if c not in used and ok(c): pick.append(c)
    n_ana=len(pick)
    for c in topc:
        if len(pick)>=50: break
        if c in used or c in pick or not ok(c): continue
        pick.append(c)
    used|=set(pick); assert len(pick)==50
    pd.DataFrame({"CatalogID":pick,"origen":["analogo_serie"]*n_ana+["modelo"]*(50-n_ana)}).to_csv(f"data/processed/gp/exp41_selection_{out}.csv",index=False)
    print(f"{out}: 50 compuestos ({n_ana} analogos + {50-n_ana} del modelo) -> {nser(pick)} series distintas | peor puntaje exp16 usado: posicion {max(topc.index(c) for c in pick if c in topc)+1 if any(c in topc for c in pick) else '-'} de 6000")
