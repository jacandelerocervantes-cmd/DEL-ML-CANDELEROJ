"""exp44: lists A4/B4 built from the organizers' series logic (exp43).
Slots: (1) unlabelled members of the extended series of known hits (compounds containing the series MCS), 2-3 per series, best exp16 score first; B4 takes the next ones;
(2) best unlabelled analog (Tanimoto>=0.5) of the singleton hits, 1 in A4 / next in B4; (3) rest from the exp19 model lists. InChIKey14 dedup; labelled compounds excluded from Sel_50.
Outputs data/processed/gp/exp44_selection_{A4,B4}.csv and exp44_scores_{A4,B4}.csv (CatalogID,Sel_50,Score; labelled cpds scored by label). Run with the RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
al=pd.read_csv("data/raw/active_learning_data.csv"); lab=dict(zip(al.CatalogID,al.Label)); P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
sco=dict(pd.read_csv("data/processed/gp/exp16_pool_scores.csv")[["CatalogID","Score"]].values)
M=pd.read_csv("data/processed/gp/exp43_unlabelled_series_members.csv"); M["score"]=M.CatalogID.map(sco); R=pd.read_csv("data/processed/gp/exp43_extended_series.csv")
ik=lambda c:(MolToInchiKey(Chem.MolFromSmiles(smi[c])) or "")[:14]
quota={int(r.serie):(3 if r.sin_etiqueta>=25 else 2) for r in R.itertuples() if r.mcs!="singleton"}
sel={"A4":[],"B4":[]}; origin={"A4":[],"B4":[]}; seen={"A4":set(),"B4":set()}; used=set()
for nm in ("A4","B4"):
    for k,q in quota.items():
        g=M[(M.serie==k)&(~M.CatalogID.isin(used))].sort_values("score",ascending=False); c=0
        for _,r in g.iterrows():
            if c>=q: break
            i=ik(r.CatalogID)
            if i in seen[nm]: continue
            sel[nm].append(r.CatalogID); origin[nm].append(f"serie_ampliada_{k}"); seen[nm].add(i); used.add(r.CatalogID); c+=1
    print(nm,"de series ampliadas:",len(sel[nm]))
# analogos de hits sueltos (Tanimoto >=0.5), reutiliza los candidatos de exp38
gen=G.GetMorganGenerator(radius=2,fpSize=2048); H=al[al.Label==1]; HF=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in H.RDKit_SMILES]
ext=set(M.CatalogID); cand=pd.concat([pd.read_csv(f"data/processed/gp/exp38_selection_{n}.csv").query("origen=='analogo_serie'") for n in ("A2","B2")]).CatalogID.tolist()
cand=[c for c in dict.fromkeys(cand) if c not in lab]
# analogos directos de los hits SUELTOS (serie sin MCS en exp43): top-3 sin etiqueta por Tanimoto>=0.45 (ECFP4)
single=[h for r in R.itertuples() if r.mcs=="singleton" for h in str(r.hits_ids).split(";")]
SF=[gen.GetFingerprint(Chem.MolFromSmiles(dict(zip(al.CatalogID,al.RDKit_SMILES))[h])) for h in single]
Ucand=[c for c in P.CatalogID if c not in lab]; best={h:[] for h in single}
for c in Ucand:
    f=gen.GetFingerprint(Chem.MolFromSmiles(smi[c])); sims=DataStructs.BulkTanimotoSimilarity(f,SF); j=int(np.argmax(sims))
    if sims[j]>=0.45: best[single[j]].append((sims[j],c))
extra=[]
for rank in range(3):
    for h in single:
        L=sorted(best[h],reverse=True)
        if len(L)>rank: extra.append(L[rank][1])
cand=list(dict.fromkeys(extra+cand)); print("analogos de hits sueltos disponibles:",len(extra),"| series de hits sueltos:",len(single))
for nm in ("A4","B4"):
    for c in cand:
        if len(sel[nm])>=36: break
        if c in used or c in ext: continue
        i=ik(c)
        if i in seen[nm]: continue
        sel[nm].append(c); origin[nm].append("analogo_hit_suelto"); seen[nm].add(i); used.add(c)
base={"A4":pd.read_csv("data/processed/gp/exp19_selection_A.csv").CatalogID.tolist(),"B4":pd.read_csv("data/processed/gp/exp19_selection_B.csv").CatalogID.tolist()}
for nm in ("A4","B4"):
    for c in base[nm]:
        if len(sel[nm])>=50: break
        if c in used or c in lab: continue
        i=ik(c)
        if i in seen[nm]: continue
        sel[nm].append(c); origin[nm].append("modelo_exp19"); seen[nm].add(i); used.add(c)
    assert len(sel[nm])==50 and len(set(sel[nm]))==50 and not (set(sel[nm])&set(lab)), nm
    pd.DataFrame({"CatalogID":sel[nm],"origen":origin[nm]}).to_csv(f"data/processed/gp/exp44_selection_{nm}.csv",index=False)
    print(nm,"->",pd.Series(origin[nm]).str.replace(r"_\d+$","",regex=True).value_counts().to_dict())
print("solape A4/B4:",len(set(sel['A4'])&set(sel['B4'])))
s=pd.read_csv("data/processed/gp/exp16_pool_scores.csv")[["CatalogID","Score"]]; s["lab"]=s.CatalogID.map(lab); unl=s.lab.isna(); lo,hi=s.loc[unl,"Score"].min(),s.loc[unl,"Score"].max()
s["Score2"]=np.nan; s.loc[unl,"Score2"]=0.001+0.998*(s.loc[unl,"Score"]-lo)/(hi-lo); s.loc[s.lab==1,"Score2"]=1.0; s.loc[s.lab==0,"Score2"]=0.0
for nm in ("A4","B4"):
    o=pd.DataFrame({"CatalogID":s.CatalogID,"Sel_50":s.CatalogID.isin(set(sel[nm])).astype(int),"Score":s.Score2.round(6)}); assert len(o)==184632 and o.Sel_50.sum()==50 and o.Score.notna().all()
    # los elegidos, arriba entre los no etiquetados
    o.loc[o.Sel_50==1,"Score"]=o.loc[o.Sel_50==1,"Score"].clip(lower=0.9990); o.to_csv(f"data/processed/gp/exp44_scores_{nm}.csv",index=False)
