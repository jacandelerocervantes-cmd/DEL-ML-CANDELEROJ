"""exp36: how many chemical series (clusters) do the known hits form?
Step 1 calibrate a similarity threshold against Synapse's OFFICIAL cluster counts for submissions whose hit set we know exactly (hits in file = intersection with the 32 validation hits).
Step 2 apply the best threshold to: validation hits (32), test hits (39), all 71. Butina on ECFP4 Tanimoto. Run with the RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); VH=list(o[o.confirmed_label==1].CatalogID)
al=pd.read_csv("data/raw/active_learning_data.csv"); TH=dict(zip(al.CatalogID[al.Label==1],al.RDKit_SMILES[al.Label==1]))
gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=lambda s:gen.GetFingerprint(Chem.MolFromSmiles(s))
def nclus(smis,cut):
    f=[fp(s) for s in smis]; n=len(f)
    if n==1: return 1
    d=[]
    for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(f[i],f[:i])]
    return len(Butina.ClusterData(d,n,1-cut,isDistData=True))   # misma serie si Tanimoto >= cut
# envios con (hits, clusters) oficiales conocidos: 93-100 (log), 112, 131, 132, 138, 153
OFF={93:(8,1),94:(18,8),95:(17,6),96:(14,4),97:(13,3),98:(13,3),99:(19,8),100:(11,8),112:(5,5),131:(4,3),138:(2,1),153:(9,6),154:(18,15),155:(32,15)}
rows=[]
for e,(h,c) in OFF.items():
    L=[l.strip() for l in open(f"submissions/Team_candeleroj_{e}.txt") if l.strip()]; H=[x for x in L if x in set(VH)]
    rows.append((e,len(H),h,c,H))
print("envio | hits en archivo (=reportados?) | clusters oficiales | clusters por umbral Tanimoto (0.2 .. 0.6)")
cuts=[0.25,0.28,0.30,0.32,0.34,0.35,0.38,0.40,0.45]; err={c:0 for c in cuts}
for e,nh,h,c,H in rows:
    r=[nclus([vs[x] for x in H],t) for t in cuts]
    for t,k in zip(cuts,r): err[t]+=abs(k-c)
    print(f"{e:5d} | {nh:3d} ({'ok' if nh==h else 'NO'} {h}) | {c:2d} | "+" ".join(f"{k:2d}" for k in r))
print("error absoluto total por umbral:",{t:err[t] for t in cuts}); best=min(err,key=err.get); print("mejor umbral:",best)
V=[vs[x] for x in VH]; T=list(TH.values())
print(f"\nCLUSTERS con umbral {best} (Tanimoto>=): validation {len(V)} hits -> {nclus(V,best)} | test {len(T)} hits -> {nclus(T,best)} | todos {len(V)+len(T)} -> {nclus(V+T,best)}")
for t in cuts: print(f"  umbral {t}: validation {nclus(V,t)}, test {nclus(T,t)}, juntos {nclus(V+T,t)}")
