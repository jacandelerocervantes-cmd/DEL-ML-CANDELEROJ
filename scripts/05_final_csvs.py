"""Step 5: final AL1/AL2 CSVs (CatalogID, Score, Sel_50; 178,052 rows) from the DEL-first design of step 2 (, DIVERSITY=0.6).
AL1 = top-50 diversified of z(DEL)+W_AL*z(AL) inside the top-20% DEL pool. AL2 = same rule, but pool members that share a chemical family with any AL1 pick are excluded;
family = same Murcko scaffold, same generic (carbon-skeleton) scaffold, or Morgan-r2 Tanimoto >= FAMILY_TANI to any AL1 pick. Score = min-max of the blend; picked compounds lifted to >=0.999 so file is coherent.
New files are written with the tag DELML so the earlier files are not overwritten. Output also results/ablation_final.json (picks changed when DEL is removed). RDKit venv."""
import json, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
RDLogger.DisableLog('rdApp.*'); g2=G.GetMorganGenerator(radius=2,fpSize=2048)
PCT,W,DIV,CAND,FAMILY_TANI=50,0.6,0.6,6000,0.45  # W = weight of z(DEL) vs z(AL)=1; DEL v2 (historic_hits<=1)
U=pd.read_csv("results/pool_scores_v2.csv"); U["del"]=U.del2; z=lambda a:(a-a.mean())/a.std(); U["zdel"]=z(U["del"]); U["zal"]=z(U.al); U["blend"]=W*U.zdel+U.zal
mol={}; fp={}
M=lambda c,s: mol.setdefault(c,Chem.MolFromSmiles(s)); F=lambda c,s: fp.setdefault(c,g2.GetFingerprint(M(c,s)))
def pick(df,score,exclude=(),n=50):
    o=[];f_=[]
    for r in df[~df.CatalogID.isin(exclude)].assign(_s=score).sort_values("_s",ascending=False).head(CAND).itertuples():
        f=F(r.CatalogID,r.SMILES)
        if f_ and max(DataStructs.BulkTanimotoSimilarity(f,f_))>=DIV: continue
        o.append(r.CatalogID); f_.append(f)
        if len(o)>=n: break
    return o
def scaf(c,s,generic=False):
    try:
        sc=MS.GetScaffoldForMol(M(c,s))
        if generic: sc=MS.MakeScaffoldGeneric(sc)
        return Chem.MolToSmiles(sc)
    except Exception: return ""
pool=U[U["del"].rank(ascending=False)<=PCT/100*len(U)].copy()
a1=pick(pool,pool.blend); A1=pool[pool.CatalogID.isin(a1)]
fs={scaf(r.CatalogID,r.SMILES) for r in A1.itertuples()}-{""}; fg={scaf(r.CatalogID,r.SMILES,True) for r in A1.itertuples()}-{""}; f1=[F(r.CatalogID,r.SMILES) for r in A1.itertuples()]
def same_family(r): return scaf(r.CatalogID,r.SMILES) in fs or scaf(r.CatalogID,r.SMILES,True) in fg or max(DataStructs.BulkTanimotoSimilarity(F(r.CatalogID,r.SMILES),f1))>=FAMILY_TANI
# cheap pre-filter on the candidates actually reachable by the picker (top CAND*3 of the blend), then exact family test
cand=pool.sort_values("blend",ascending=False).head(CAND*3); fam={r.CatalogID for r in cand.itertuples() if r.CatalogID in set(a1) or same_family(r)}
a2=pick(pool,pool.blend,exclude=fam)
assert len(a1)==50 and len(a2)==50 and not set(a1)&set(a2)
chk=[r for r in pool[pool.CatalogID.isin(a2)].itertuples() if same_family(r)]; assert not chk, "AL2 still shares a family with AL1"
S=(U.blend-U.blend.min())/(U.blend.max()-U.blend.min())
for tag,sel in (("AL1",a1),("AL2",a2)):
    m=U.CatalogID.isin(set(sel)).values; s=np.where(m,np.maximum(S,0.999),S*0.998)
    o=pd.DataFrame({"CatalogID":U.CatalogID,"Score":np.round(s,6),"Sel_50":m.astype(int)}).sort_values("Score",ascending=False); o.to_csv(f"active_learning_candidates/Team_candeleroj_{tag}_DELML_20261022.csv",index=False)
r1=pick(U,U.zal); r2=pick(U,U.zal,exclude={x.CatalogID for x in U.itertuples() if x.CatalogID in fam or x.CatalogID in set(a1)})
d="active_learning_candidates/"; T1=set(pd.read_csv(d+"Team_candeleroj_AL1_20261005.csv").query("Sel_50==1").CatalogID); T2=set(pd.read_csv(d+"Team_candeleroj_AL2_20261005.csv").query("Sel_50==1").CatalogID)
out=dict(config=dict(PCT=PCT,W_DEL=W,DIVERSITY=DIV,FAMILY_TANI=FAMILY_TANI),AL1_changed_without_DEL=50-len(set(a1)&set(r1)),AL2_changed_without_DEL=50-len(set(a2)&set(r2)),
 overlap_with_earlier=dict(newAL1_vs_oldAL1=len(set(a1)&T1),newAL2_vs_oldAL2=len(set(a2)&T2)),
 DEL_percentile_mean=dict(AL1=float(U["del"].rank(pct=True)[U.CatalogID.isin(a1)].mean()),AL2=float(U["del"].rank(pct=True)[U.CatalogID.isin(a2)].mean())))
json.dump(out,open("results/ablation_final.json","w"),indent=1); print(json.dumps(out,indent=1))
