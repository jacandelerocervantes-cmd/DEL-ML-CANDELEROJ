"""exp70: RETROSPECTIVE analysis of the 100 submitted compounds (50 of AL1 + 50 of AL2) against the DEL-ML model. It does NOT regenerate or modify any submission.
All thresholds are fixed in advance (top 5/10/20/30 % of the DEL ranking); nothing is tuned so that the 100 fall inside them.
Parts: (1) position of the 100 in the DEL ranking vs a random null; (2) families (Butina 0.32) and DEL position per family; (3) physico-chemical profile (MW, cLogP, TPSA, HBD, aromatic rings, Fsp3, ESOL logS)
of the 100 vs pool vs the 39 labelled hits, and Spearman correlation DEL-AL-properties over the pool; (4) ablation of a pre-specified DEL-first design (DEL top 20 % pool, score z(DEL)+z(AL), diversity 0.6)
against the same design without DEL. Run from the repository root after exp43/exp48/exp60 (needs data/processed/gp/*). RDKit venv. Output: results/exp70_del_retrospective.json"""
import json, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rd, rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr, hypergeom, mannwhitneyu
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048); g3=G.GetMorganGenerator(radius=3,fpSize=2048)
cnt=lambda ms,g:np.log1p(np.array([g.GetCountFingerprintAsNumPy(m) for m in ms],dtype=np.float32))
F=lambda smis,g:np.vstack([cnt([Chem.MolFromSmiles(s) for s in smis[i:i+20000]],g) for i in range(0,len(smis),20000)])
z=lambda a:(a-a.mean())/a.std(); out={}
al=pd.read_csv("data/raw/active_learning_data.csv"); y=al.Label.values
P=pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc=[c for c in P.columns if 'smi' in c.lower()][0]
U=P[~P.CatalogID.isin(set(al.CatalogID))].reset_index(drop=True).rename(columns={sc:"smi"})
# DEL model (same as exp51) and AL model (same cross-fit as exp48, seed 48; raw scores, selected compounds NOT lifted)
t=pd.read_pickle("data/processed/del_sel.pkl"); pos=t[(t.count_PGK2>=5)&(t.count_NTC==0)]; neg=t[t.count_PGK2<=1].sample(50000,random_state=0)
dm=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(F(list(pos.SMILES)+list(neg.SMILES),g2),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
U["del"]=np.concatenate([dm.decision_function(F(list(U.smi[i:i+20000]),g2)) for i in range(0,len(U),20000)])
XU=F(list(U.smi),g3); XL=F(list(al.RDKit_SMILES),g3); rng=np.random.default_rng(48); half=rng.permutation(len(XU)); H1,H2=half[:len(XU)//2],half[len(XU)//2:]; s=np.zeros(len(XU),dtype=np.float32)
for tr,te in ((H1,H2),(H2,H1)):
    ex=rng.choice(tr,50000,replace=False); m=LogisticRegression(C=0.05,class_weight="balanced",max_iter=3000).fit(np.vstack([XL,XU[ex]]),np.r_[y,np.zeros(len(ex))]); s[te]=m.decision_function(XU[te])
U["al"]=s; U[["CatalogID","del","al"]].to_csv("data/processed/gp/exp70_pool_scores.csv",index=False); U["zdel"]=z(U["del"]); U["zal"]=z(U.al); U["rdel"]=U["del"].rank(ascending=False)/len(U); U["ral"]=U.al.rank(ascending=False)/len(U)
A1=list(pd.read_csv("data/processed/gp/exp48_selection_A6.csv").CatalogID); A2=list(pd.read_csv("data/processed/gp/exp60_selection_AL2_newgroups.csv").CatalogID)
U["grp"]=np.where(U.CatalogID.isin(A1),"AL1",np.where(U.CatalogID.isin(A2),"AL2","pool")); S=U[U.grp!="pool"].copy()
# (1) position in the DEL ranking vs random null
N=len(U); res={}
for nm,ids in (("AL1",A1),("AL2",A2),("AL1+AL2",A1+A2)):
    r=U[U.CatalogID.isin(ids)].rdel.values; row={}
    for q in (0.05,0.10,0.20,0.30):
        k=int((r<=q).sum()); row[f"top{int(q*100)}pct"]=dict(n=k,of=len(r),p_vs_random=float(hypergeom.sf(k-1,N,int(q*N),len(r))))
    row["median_percentile"]=float(np.median(r)*100); res[nm]=row
out["1_del_position"]=res
# (2) families: Butina 0.32 over the 100
fp=[g2.GetFingerprint(Chem.MolFromSmiles(x)) for x in S.smi]; n=len(fp); d=[]
for i in range(1,n): d+=[1-v for v in DataStructs.BulkTanimotoSimilarity(fp[i],fp[:i])]
S["fam"]=0
for k,c in enumerate(Butina.ClusterData(d,n,1-0.32,isDistData=True)): S.iloc[list(c),S.columns.get_loc("fam")]=k
M=pd.read_csv("data/processed/gp/exp43_unlabelled_series_members.csv")[["CatalogID","serie"]].drop_duplicates("CatalogID"); S=S.merge(M,on="CatalogID",how="left")
fam=S.groupby("fam").agg(n=("CatalogID","size"),AL1=("grp",lambda g:(g=="AL1").sum()),AL2=("grp",lambda g:(g=="AL2").sum()),med_del_pct=("rdel",lambda r:100*r.median()),series=("serie",lambda s_:sorted(set(s_.dropna().astype(int)))),example=("smi","first")).sort_values("n",ascending=False)
out["2_families"]=dict(n_families=int(len(fam)),families_with_both_AL1_AL2=int(((fam.AL1>0)&(fam.AL2>0)).sum()),table=json.loads(fam.head(40).to_json(orient="records")))
# (3) profile
def prof(smi):
    m=Chem.MolFromSmiles(smi); mw=Descriptors.MolWt(m); lp=Crippen.MolLogP(m); rb=rd.CalcNumRotatableBonds(m); ha=m.GetNumHeavyAtoms(); ar=sum(a.GetIsAromatic() for a in m.GetAtoms())/ha
    return dict(MW=mw,cLogP=lp,TPSA=rd.CalcTPSA(m),HBD=rd.CalcNumHBD(m),HBA=rd.CalcNumHBA(m),RotB=rb,AroRings=rd.CalcNumAromaticRings(m),Fsp3=rd.CalcFractionCSP3(m),ESOL_logS=0.16-0.63*lp-0.0062*mw+0.066*rb-0.74*ar)
Pf=pd.DataFrame([prof(x) for x in U.smi]); Pf["CatalogID"]=U.CatalogID.values; Hf=pd.DataFrame([prof(x) for x in al.RDKit_SMILES[y==1]])
Sf=Pf[Pf.CatalogID.isin(S.CatalogID)]; cols=[c for c in Pf.columns if c!="CatalogID"]
out["3_profile_medians"]={c:dict(pool=float(Pf[c].median()),the100=float(Sf[c].median()),labelled_hits=float(Hf[c].median()),mannwhitney_p_100_vs_pool=float(mannwhitneyu(Sf[c],Pf[c]).pvalue)) for c in cols}
out["3_spearman_over_pool"]={"DEL_vs_AL":float(spearmanr(U["del"],U.al)[0]),**{f"DEL_vs_{c}":float(spearmanr(U["del"],Pf[c])[0]) for c in cols},**{f"AL_vs_{c}":float(spearmanr(U.al,Pf[c])[0]) for c in cols}}
# (4) ablation of the pre-specified DEL-first design
def pick(df,score,n=50,th=0.6):
    o=[];f_=[]
    for r in df.assign(_s=score).sort_values("_s",ascending=False).head(4000).itertuples():
        f=g2.GetFingerprint(Chem.MolFromSmiles(r.smi))
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=th: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=n: break
    return o
pool=U[U.rdel<=0.20]; with_del=pick(pool,pool.zdel+pool.zal); no_del=pick(U,U.zal); old=set(A1+A2)
out["4_ablation_DEL_first_top20pct"]=dict(selected=len(with_del),shared_with_without_DEL=len(set(with_del)&set(no_del)),changed_by_removing_DEL=50-len(set(with_del)&set(no_del)),with_DEL_in_the_100=len(set(with_del)&old),without_DEL_in_the_100=len(set(no_del)&old))
json.dump(out,open("results/exp70_del_retrospective.json","w"),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if k!="2_families"},indent=1,default=float)); print("familias:",out["2_families"]["n_families"],"con AL1 y AL2:",out["2_families"]["families_with_both_AL1_AL2"])
