"""exp43: EXTENDED series of the 39 known TEST hits, following the organizers' recipe: series (Kruger algorithm, exp42) -> MCS -> every library compound containing the MCS.
For each non-singleton series: members in the test pool, labelled neighbours (hits / negatives) and UNLABELLED compounds containing the MCS (the finite set where hidden hits of that series can be).
Run with the RDKit venv. Outputs data/processed/gp/exp43_*.csv"""
import numpy as np, pandas as pd
src=open("scripts/exp42_kruger_series.py",encoding="utf8").read().split("o=pd.read_csv(\"data/processed/confirmed_submission_outcomes_v4.csv\")")[0]
exec(src)
al=pd.read_csv("data/raw/active_learning_data.csv"); TH=list(al[al.Label==1].CatalogID); smi.update(dict(zip(al.CatalogID,al.RDKit_SMILES))); lab=dict(zip(al.CatalogID,al.Label))
S=series_of(TH,cap=100); log("series (Kruger) entre los",len(TH),"hits del test:",len(S),"| con MCS (no singleton):",sum(sm!="singleton" for _,sm in S))
tm=[(cid,Chem.MolFromSmiles(s)) for cid,s in zip(T.CatalogID,T[[x for x in T.columns if 'smi' in x.lower()][0]])]; tm=[(c,m) for c,m in tm if m is not None]
rows=[]; unl_any=set(); detail=[]
for k,(mem,sm) in enumerate(S):
    ids=[TH[i] for i in mem]
    if sm=="singleton": rows.append(dict(serie=k,hits=len(mem),mcs="singleton",en_pool=np.nan,etiquetados_hits=len(mem),etiquetados_neg=np.nan,sin_etiqueta=np.nan,hits_ids=";".join(ids))); continue
    q=Chem.MolFromSmarts(sm); hit_ids=[c for c,m in tm if m.HasSubstructMatch(q)]
    nh=sum(lab.get(c)==1 for c in hit_ids); nn=sum(lab.get(c)==0 for c in hit_ids); un=[c for c in hit_ids if c not in lab]; unl_any|=set(un)
    rows.append(dict(serie=k,hits=len(mem),mcs=sm,en_pool=len(hit_ids),etiquetados_hits=nh,etiquetados_neg=nn,sin_etiqueta=len(un),tasa_en_etiquetados=round(nh/max(nh+nn,1),2),hits_ids=";".join(ids)))
    detail+= [dict(serie=k,CatalogID=c,smiles=smi[c]) for c in un]
R=pd.DataFrame(rows).sort_values("hits",ascending=False); R.to_csv("data/processed/gp/exp43_extended_series.csv",index=False); pd.DataFrame(detail).to_csv("data/processed/gp/exp43_unlabelled_series_members.csv",index=False)
pd.set_option('display.width',230); print(R[["serie","hits","en_pool","etiquetados_hits","etiquetados_neg","sin_etiqueta","tasa_en_etiquetados"]].to_string(index=False))
print("\nTOTAL de compuestos SIN etiqueta que contienen la MCS de alguna serie conocida (union):",len(unl_any))
