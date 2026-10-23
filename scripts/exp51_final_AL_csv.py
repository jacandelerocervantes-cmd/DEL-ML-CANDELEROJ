"""exp51: final Active Learning test CSVs, in the format of the organizers' example (active_learning_submission_example.csv): 178,052 rows (the UNLABELLED test compounds), columns CatalogID, Score, Sel_50.
Score = min-max of z(model) + 0.1*z(DEL logistic); the model z is recovered from exp48's rescaled score (a linear map of the raw decision function, so z is unchanged).
Sel_50 = the exp48 lists A6 (AL1) and B6 (AL2); the 50 picked compounds are lifted to the top of the Score range so the file is coherent. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
cnt=lambda ms:np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
tmpl=pd.read_csv(r"data/raw/active_learning_submission_example.csv"); ids=list(tmpl.CatalogID)
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]; smi=dict(zip(P.CatalogID,P[sc]))
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
XD=cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES)+list(neg.SMILES)]); dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(XD,np.r_[np.ones(len(pos)),np.zeros(len(neg))])
ms=[Chem.MolFromSmiles(smi[c]) for c in ids]; sd=np.concatenate([dm.decision_function(cnt(ms[i:i+20000])) for i in range(0,len(ms),20000)]); print("DEL listo sobre",len(ids),"compuestos",flush=True)
base=pd.read_csv("data/processed/gp/exp48_scores_A6.csv").set_index("CatalogID").Score.reindex(ids).values   # modelo reescalado (lineal)
z=lambda a:(a-a.mean())/a.std(); blend=z(base)+0.1*z(sd); S=(blend-blend.min())/(blend.max()-blend.min())
for nm,tag in (("A6","AL1"),("B6","AL2")):
    sel=set(pd.read_csv(f"data/processed/gp/exp48_selection_{nm}.csv").CatalogID); assert len(sel)==50 and sel<=set(ids)
    m=np.array([c in sel for c in ids]); s=S.copy(); s[m]=np.maximum(s[m],0.999)+ (s[m]-s[m].min())*0   # elegidos arriba
    o=pd.DataFrame({"CatalogID":ids,"Score":np.round(s,6),"Sel_50":m.astype(int)}).sort_values("Score",ascending=False)
    p=f"submissions/active_learning_candidates/Team_candeleroj_{tag}_20261005.csv"; o.to_csv(p,index=False)
    chk=pd.read_csv(p); assert list(chk.columns)==["CatalogID","Score","Sel_50"] and len(chk)==178052 and chk.Sel_50.sum()==50 and set(chk.CatalogID)==set(ids) and chk.Score.between(0,1).all() and chk.Score.notna().all()
    print(tag,"->",p,"| filas",len(chk),"| Sel_50",int(chk.Sel_50.sum()),"| mismos IDs que la plantilla: True | Score unicos",chk.Score.nunique(),"| rango de Score",round(chk.Score.min(),4),round(chk.Score.max(),4))
