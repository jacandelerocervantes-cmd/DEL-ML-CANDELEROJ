"""exp59: INFERENCE (not identification) of which of AL1's 50 compounds are the 29 hits and how they spread over the 10 series reported by Synapse.
Prior hit probability per compound from validation (series-members 0.67 vs model picks 0.83 at the top, ratio ~0.8), scaled so that the expected number of hits is 29;
series among the 50 = Butina clusters at Tanimoto>=0.32 (calibrated against Synapse's official counts). Rejection sampling of hit indicator vectors keeping those with exactly 29 hits AND 10 series.
Outputs per-compound posterior probability. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
S=pd.read_csv("data/processed/gp/exp48_selection_A6.csv"); ids=list(S.CatalogID); n=50; fp=[g2.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in ids]; d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(d,n,1-0.32,isDistData=True)): cl[list(c)]=k
C=cl.max()+1; sizes=np.bincount(cl); print("las 50 forman",C,"grupos de similitud (0.32); tamanos:",sorted(sizes.tolist(),reverse=True)); print("hits observados: 29 en 10 series")
is_series=(S.origen=="serie_ampliada").values; p0=np.where(is_series,0.8,1.0); p=p0*29/p0.sum(); print("probabilidad previa: series ampliadas",round(p[is_series][0],3),"| modelo",round(p[~is_series][0],3))
rng=np.random.default_rng(59); acc=[]; tot=0; T=4_000_000; B=250_000
for _ in range(T//B):
    X=rng.random((B,n))<p; ok=X.sum(1)==29; X=X[ok]
    if len(X)==0: continue
    # series distintas entre los hits de cada muestra
    M=np.zeros((len(X),C),bool)
    for k in range(C): M[:,k]=X[:,cl==k].any(1)
    nc=M.sum(1); acc.append(X[nc==10]); tot+=ok.sum()
A=np.vstack(acc) if acc else np.zeros((0,n),bool); print(f"muestras con 29 hits: {tot:,} | de ellas con exactamente 10 series: {len(A):,}")
if len(A):
    post=A.mean(0); S["P_hit"]=post.round(3); S["grupo"]=cl; S["smiles"]=[smi[c] for c in ids]; S.to_csv("data/processed/gp/exp59_AL1_posterior.csv",index=False)
    print("prob. posterior: media",round(post.mean(),3),"| minima",round(post.min(),3),"| maxima",round(post.max(),3))
    print("series ampliadas:",round(post[is_series].mean(),3),"| modelo:",round(post[~is_series].mean(),3))
    print("compuestos con P>=0.8:",int((post>=0.8).sum()),"| con P<=0.4:",int((post<=0.4).sum()))
    g=pd.DataFrame({"grupo":cl,"p":post}).groupby("grupo").agg(n=("p","size"),hits_esperados=("p","sum")).round(2).sort_values("n",ascending=False); print("hits esperados por grupo de similitud:"); print(g.head(12).to_string())
