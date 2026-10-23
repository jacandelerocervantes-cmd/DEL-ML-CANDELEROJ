"""exp33: use Synapse's cluster column. A pool answer with MORE HITS THAN CLUSTERS means two of its hits belong to the same series (envio 142: 2 hits, 1 cluster).
Among the hit sets still consistent (exp31 history) list the pairs inside pool 142 and score them by shared structure (MCS atoms, Tanimoto). Run with the RDKit venv."""
import itertools, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, rdFMCS
RDLogger.DisableLog('rdApp.*')
exec(open("scripts/exp31_adaptive_groups.py",encoding="utf8").read().split('if "--selftest"')[0])
RESULTS.update({132:2,133:1,134:0,135:1,136:2,137:1,138:2,139:1,140:2,141:1,142:2,143:2,144:2,145:3,146:2,147:1,148:1,149:2,150:3,152:2})
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
gen=G.GetMorganGenerator(radius=2,fpSize=2048); mol=lambda c:Chem.MolFromSmiles(vs[c])
g="sueltos"; ids=IDS[g]; c=hyps(g,history_from_files(g)); print("sueltos: combinaciones consistentes",len(c))
Pl=pd.read_csv(PLAN); pool142=set(Pl[Pl.envio==142].miembros.iloc[0].split(";")); inpool=np.array([x in pool142 for x in ids])
pairs={}
for h in c:
    ins=[i for i in h if inpool[i]]
    if len(ins)==2: pairs[tuple(sorted(ins))]=pairs.get(tuple(sorted(ins)),0)+1
print("parejas posibles dentro del pool 142:",len(pairs),"(cada una compatible con",sorted(set(pairs.values()))[:3],"... combinaciones)")
rows=[]
for (a,b),n in pairs.items():
    t=DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol(ids[a])),gen.GetFingerprint(mol(ids[b])))
    r=rdFMCS.FindMCS([mol(ids[a]),mol(ids[b])],timeout=2,ringMatchesRingOnly=True,completeRingsOnly=True); rows.append((ids[a],ids[b],round(t,2),r.numAtoms,n))
T=pd.DataFrame(rows,columns=["A","B","tanimoto","atomos_MCS","combinaciones"]).sort_values(["atomos_MCS","tanimoto"],ascending=False)
T.to_csv("data/processed/gp/exp33_sueltos_pairs.csv",index=False); print(T.head(12).to_string(index=False))
print("\nMCS de la mejor vs la siguiente:",T.atomos_MCS.iloc[0],T.atomos_MCS.iloc[1],T.atomos_MCS.iloc[2])
for x in set(T.head(4).A)|set(T.head(4).B): print(" ",x,vs[x])
