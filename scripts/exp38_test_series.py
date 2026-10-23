"""exp38: series of the 39 real TEST hits (Butina, Tanimoto>=0.32 = threshold calibrated on Synapse's official counts) and how good a bet their analogs are.
Per series: members, labelled neighbours (Tanimoto>=0.5 to a member): hits vs negatives -> empirical hit rate; unlabelled pool analogs (>=0.5). Then revised lists A2/B2:
dedup by InChIKey14, drop the 7 pool compounds identical to known negatives, reserve slots for the best analog of each series (1 in A2, next in B2), rest from exp19 A/B by score.
Run with the RDKit venv. Outputs data/processed/gp/exp38_*.csv"""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=lambda s:gen.GetFingerprint(Chem.MolFromSmiles(s)); ik=lambda s:(MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14]
al=pd.read_csv("data/raw/active_learning_data.csv"); H=al[al.Label==1].reset_index(drop=True); HF=[fp(s) for s in H.RDKit_SMILES]; n=len(HF); d=[]
for i in range(1,n): d+=[1-x for x in DataStructs.BulkTanimotoSimilarity(HF[i],HF[:i])]
cl=Butina.ClusterData(d,n,1-0.32,isDistData=True); ser=np.zeros(n,int)
for k,c in enumerate(sorted(cl,key=len,reverse=True)): ser[list(c)]=k
H["serie"]=ser; print("series entre los 39 hits del test (umbral 0.32):",ser.max()+1,"| tamanos:",sorted(np.bincount(ser).tolist(),reverse=True))
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; lab=dict(zip(al.CatalogID,al.Label))
def simmat(smis):
    f=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]; return np.array([DataStructs.BulkTanimotoSimilarity(x,HF) for x in f],dtype=np.float32)
Ln=al[al.Label==0]; SL=simmat(Ln.RDKit_SMILES); U=P[~P.CatalogID.isin(lab)].reset_index(drop=True); SU=simmat(U[sc]); print("matrices de similitud listas",flush=True)
rows=[]
for k in range(ser.max()+1):
    mem=np.where(ser==k)[0]; nl=int((SL[:,mem].max(1)>=0.5).sum()); an=SU[:,mem].max(1); na=int((an>=0.5).sum())
    rows.append(dict(serie=k,hits=len(mem),negativos_vecinos=nl,tasa_hit_vecindad=round(len(mem)/(len(mem)+nl),3),analogos_sin_etiqueta_0p5=na,representante=H.RDKit_SMILES[mem[0]],ejemplo_id=H.CatalogID[mem[0]]))
T=pd.DataFrame(rows); T.to_csv("data/processed/gp/exp38_test_series.csv",index=False)
pd.set_option('display.width',240); pd.set_option('display.max_colwidth',58); print(T[["serie","hits","negativos_vecinos","tasa_hit_vecindad","analogos_sin_etiqueta_0p5","representante"]].to_string(index=False))
# ---- listas A2/B2
sc16=dict(pd.read_csv("data/processed/gp/exp16_pool_scores.csv")[["CatalogID","Score"]].values)
U["serie"]=SU.argmax(1)
U["simmax"]=SU.max(1); U["score"]=U.CatalogID.map(sc16)
A=pd.read_csv("data/processed/gp/exp19_selection_A.csv"); B=pd.read_csv("data/processed/gp/exp19_selection_B.csv"); base=pd.concat([A.assign(l="A"),B.assign(l="B")])
cand=pd.concat([U[U.simmax>=0.5].sort_values(["serie","simmax"],ascending=[True,False]).groupby("serie").head(3),base.rename(columns={base.columns[1]:sc})[["CatalogID",sc]]]).drop_duplicates("CatalogID")
cand["ik"]=[ik(s) for s in cand[sc]]; negik={ik(s) for s in Ln.RDKit_SMILES}; cand["dup_neg_conocido"]=cand.ik.isin(negik)
print("\nanalogos candidatos:",len(cand[cand.CatalogID.isin(U[U.simmax>=0.5].CatalogID)]),"| duplicados de negativos conocidos entre candidatos:",int(cand.dup_neg_conocido.sum()))
used=set(); picks={"A2":[], "B2":[]}
for rnd,name in ((0,"A2"),(1,"B2")):
    for k in range(ser.max()+1):
        g=U[(U.serie==k)&(U.simmax>=0.5)&(~U.CatalogID.isin(used))].merge(cand[["CatalogID","ik","dup_neg_conocido"]],on="CatalogID",how="left")
        g=g[(g.dup_neg_conocido!=True)&(~g.ik.isin({x[1] for x in picks[name]}))].sort_values("score",ascending=False)
        if len(g): r=g.iloc[0]; picks[name].append((r.CatalogID,r.ik,k,round(float(r.simmax),2))); used.add(r.CatalogID)
for name,lst in (("A2",A),("B2",B)):
    sel=[x[0] for x in picks[name]]; seen={x[1] for x in picks[name]}
    rest=lst.copy(); rest["ik"]=[ik(s) for s in rest[[c for c in rest.columns if 'smi' in c.lower()][0]]]
    for _,r in rest.iterrows():
        if len(sel)>=50: break
        if r.CatalogID in used or r.ik in seen or r.ik in negik: continue
        sel.append(r.CatalogID); seen.add(r.ik); used.add(r.CatalogID)
    pd.DataFrame({"CatalogID":sel,"origen":["analogo_serie"]*len(picks[name])+["exp19"]*(len(sel)-len(picks[name]))}).to_csv(f"data/processed/gp/exp38_selection_{name}.csv",index=False)
    print(f"{name}: {len(sel)} compuestos | {len(picks[name])} analogos (1 por serie, series {sorted(x[2] for x in picks[name])}) | {len(sel)-len(picks[name])} de exp19")
