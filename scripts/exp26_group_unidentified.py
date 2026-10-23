"""exp26: group the still-uncertain unidentified validation compounds (194 after exp6 with outcomes_v2) into chemical families and bound the number of hits
each family can hold, using the same exact-count constraints as exp6 (scipy milp). Families = Butina clusters (Tanimoto>=0.4) over ECFP4, labelled by motif and by
the nearest KNOWN validation hit. Run with a Python that has RDKit (the isolated 3.12 venv). Output: data/processed/gp/exp26_unidentified_families.csv"""
import ast, os, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
from scipy.optimize import milp, LinearConstraint, Bounds
RDLogger.DisableLog('rdApp.*')
src=open("scripts/build_confirmed_outcomes.py",encoding="utf8").read(); i=src.index("ROUND_OUTCOMES = {"); j=src.index("\n}",i)
R=ast.literal_eval(src[i+len("ROUND_OUTCOMES = "):j+2])
R.update({"Team_candeleroj_101.txt":3,"Team_candeleroj_102.txt":2,"Team_candeleroj_103.txt":0,"Team_candeleroj_112.txt":5,"Team_candeleroj_116.txt":0, "Team_candeleroj_117.txt":0, "Team_candeleroj_118.txt":0, "Team_candeleroj_119.txt":0, "Team_candeleroj_120.txt":0, "Team_candeleroj_121.txt":1, "Team_candeleroj_122.txt":0, "Team_candeleroj_123.txt":1, "Team_candeleroj_124.txt":1, "Team_candeleroj_125.txt":0, "Team_candeleroj_126.txt":0, "Team_candeleroj_127.txt":1, "Team_candeleroj_128.txt":1, "Team_candeleroj_129.txt":1, "Team_candeleroj_130.txt":0, "Team_candeleroj_131.txt":4})
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v2.csv"); lab=dict(zip(o.CatalogID,o.confirmed_label))
rows=[]; unk=set()
for f,n in R.items():
    p="submissions/"+f
    if not os.path.exists(p): continue
    ids=[l.strip() for l in open(p) if l.strip()]; known=sum(lab.get(c)==1 for c in ids); u=[c for c in ids if c not in lab or pd.isna(lab[c])]
    unk|=set(u)
    if u: rows.append((n-known,u))
unk=sorted(unk); ix={c:k for k,c in enumerate(unk)}
A=np.zeros((len(rows),len(unk))); lo=np.array([r[0] for r in rows],float)
for k,(_,u) in enumerate(rows): A[k,[ix[c] for c in u]]=1
cons=LinearConstraint(A,lo,lo); solve=lambda c:(lambda r:None if r.status!=0 else r.x)(milp(c,constraints=cons,bounds=Bounds(0,1),integrality=np.ones(len(unk))))
def bounds(idx):
    c=np.zeros(len(unk)); c[idx]=1; return (int(round(float(c@solve(c)))),int(round(float(c@solve(-c)))))
cs=pd.read_csv("data/processed/gp/exp6_unidentified_constraints.csv")
unc=cs[cs.estado=="incierto"].CatalogID.tolist(); assert set(unc)<=set(unk), "constraints csv desactualizado"
print("variables",len(unk),"| inciertos",len(unc))
v=pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc=[x for x in v.columns if 'smi' in x.lower()][0]; vs=dict(zip(v.CatalogID,v[sc]))
gen=G.GetMorganGenerator(radius=2,fpSize=2048); fp=lambda s:gen.GetFingerprint(Chem.MolFromSmiles(s))
F=[fp(vs[c]) for c in unc]; n=len(F); dist=[]
for a in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(F[a],F[:a])]
cl=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.6,isDistData=True)): cl[list(c)]=k
hits=[c for c in o[o.confirmed_label==1].CatalogID if c in vs]; HF=[fp(vs[c]) for c in hits]
M={"tienopirimidina/-ona":'[#6]1[#7][#6]2[#16][#6][#6][#6]2[#6][#7]1',"quinazolina/-ona":'c1ncnc2ccccc12',"quinoxalina":'c1cnc2ccccc2n1',"piridina/pirimidina 3-il (biaril)":'c1cccnc1-c1ncccn1'}
def motif(s):
    m=Chem.MolFromSmiles(s)
    for k,p in M.items():
        if m.HasSubstructMatch(Chem.MolFromSmarts(p)): return k
    return "otro"
D=pd.DataFrame({"CatalogID":unc,"cluster":cl,"smiles":[vs[c] for c in unc]}); D["motivo"]=D.smiles.map(motif)
sims=[DataStructs.BulkTanimotoSimilarity(f,HF) for f in F]; D["sim_hit_conocido"]=[max(s) for s in sims]; D["hit_mas_cercano"]=[hits[int(np.argmax(s))] for s in sims]
out=[]
for k,g in D.groupby("cluster"):
    idx=[ix[c] for c in g.CatalogID]; mn,mx=bounds(idx)
    out.append(dict(cluster=k,tam=len(g),min_hits=mn,max_hits=mx,motivo=g.motivo.mode()[0],sim_max_a_hit_conocido=round(g.sim_hit_conocido.max(),2),ejemplo=g.smiles.iloc[0]))
G_=pd.DataFrame(out).sort_values("tam",ascending=False); D.to_csv("data/processed/gp/exp26_unidentified_families.csv",index=False); G_.to_csv("data/processed/gp/exp26_family_summary.csv",index=False)
tot=bounds([ix[c] for c in unc]); print("hits totales entre los",len(unc),"inciertos: min",tot[0],"max",tot[1])
print(f"familias (clusters sim>=0.4): {len(G_)} | de 1 solo compuesto: {(G_.tam==1).sum()} | de 2 a 4: {G_.tam.between(2,4).sum()} | de 5 o mas: {(G_.tam>=5).sum()}")
pd.set_option('display.width',220); pd.set_option('display.max_colwidth',60)
print(G_[G_.tam>=3].to_string(index=False))
print("\nmotivos entre los 194:",D.motivo.value_counts().to_dict())
