"""exp68: what do the known hits look like? Properties and families of (A) the 32 validation hits identified one by one, (B) the 39 test hits labelled by the organizers, compared with
(C) the labelled test negatives and (D) the whole test pool. RDKit descriptors + substructure flags; families = Butina groups at ECFP4 Tanimoto 0.32 inside each hit set.
Outputs: data/processed/gp/exp68_known_hits_properties.csv (per compound, git-ignored) and a printed summary. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rd, rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
al = pd.read_csv("data/raw/active_learning_data.csv")
T = pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); tsc = [c for c in T.columns if 'smi' in c.lower()][0]
V = pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); vsc = [c for c in V.columns if 'smi' in c.lower()][0]
lab = pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); vh = lab[lab.confirmed_label == 1].CatalogID.tolist()
vsmi = dict(zip(V.CatalogID, V[vsc]))
sets = {"A_validation_hits(32)": [vsmi[c] for c in vh],
        "B_test_hits(39)": list(al[al.Label == 1].RDKit_SMILES),
        "C_test_labelled_negatives": list(al[al.Label == 0].RDKit_SMILES),
        "D_test_pool(sample 5000)": list(T[tsc].sample(5000, random_state=68))}
pats = {"amide": "C(=O)[NX3]", "anilide": "c[NX3H1]C(=O)[#6]", "sulfonamide": "S(=O)(=O)[NX3]", "pyrimidine": "c1ncccn1", "quinazoline": "c1ncc2ccccc2n1",
        "thieno[2,3-d]pyrimidine": "c1ncc2ccsc2n1", "aminopyridine/pyrimidine NH": "[nX2]:c-[NX3;H1,H2]", "basic amine": "[NX3;H0,H1,H2;!$(NC=O);!$(NS=O);!$(N-a)]([#6])", "carboxylic acid": "C(=O)[OH]", "halogen": "[F,Cl,Br,I]"}
pats = {k: Chem.MolFromSmarts(v) for k, v in pats.items()}
g = G.GetMorganGenerator(radius=2, fpSize=2048)
def props(smi):
    m = Chem.MolFromSmiles(smi)
    d = {"MW": Descriptors.MolWt(m), "cLogP": Crippen.MolLogP(m), "TPSA": rd.CalcTPSA(m), "HBD": rd.CalcNumHBD(m), "HBA": rd.CalcNumHBA(m), "rot_bonds": rd.CalcNumRotatableBonds(m),
         "rings": rd.CalcNumRings(m), "arom_rings": rd.CalcNumAromaticRings(m), "heavy_atoms": m.GetNumHeavyAtoms(), "Fsp3": rd.CalcFractionCSP3(m), "n_amide_bonds": len(m.GetSubstructMatches(pats["amide"]))}
    for k, p in pats.items(): d["has_" + k] = int(m.HasSubstructMatch(p))
    return d
rows = []
for nm, smis in sets.items():
    for s in smis: r = props(s); r["set"] = nm; r["smiles"] = s; rows.append(r)
P = pd.DataFrame(rows); P.to_csv("data/processed/gp/exp68_known_hits_properties.csv", index=False)
num = ["MW", "cLogP", "TPSA", "HBD", "HBA", "rot_bonds", "rings", "arom_rings", "heavy_atoms", "Fsp3", "n_amide_bonds"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("=== medianas [Q1-Q3] por conjunto")
for nm in sets:
    q = P[P.set == nm][num]; print(nm, "n=%d" % len(q)); print("  " + " | ".join("%s %.2f [%.2f-%.2f]" % (c, q[c].median(), q[c].quantile(.25), q[c].quantile(.75)) for c in num))
fl = [c for c in P.columns if c.startswith("has_")]
print("\n=== % de compuestos con cada subestructura")
print((P.groupby("set")[fl].mean() * 100).round(0).T.to_string())
print("\n=== familias dentro de cada conjunto de hits (Butina, Tanimoto ECFP4 0.32)")
for nm in list(sets)[:2]:
    smis = sets[nm]; fp = [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]; n = len(fp); d = []
    for i in range(1, n): d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(fp[i], fp[:i])]
    cl = sorted(Butina.ClusterData(d, n, 1 - 0.32, isDistData=True), key=len, reverse=True)
    print(nm, "->", len(cl), "familias; tamanos:", [len(c) for c in cl])
    for c in cl[:6]:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smis[c[0]]); print("   familia de %d: andamio representativo %s" % (len(c), sc))
