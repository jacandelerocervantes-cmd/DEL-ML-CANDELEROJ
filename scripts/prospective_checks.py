"""Prospective-phase checks for a list of candidate molecules (the organizers rank the prospective phase by number of hits, chemical novelty and chemical diversity [Evaluation page]).
Input: a CSV with a SMILES column (and optionally an ID column). Usage:
    python scripts/prospective_checks.py candidates.csv [out_prefix]
Per molecule: novelty = max ECFP4 Tanimoto to (R1) the CACHE #7 co-crystal ligands of PGK2/PGK1, (R2) the public DEL hit candidates of the CACHE #7 page, (R3) the 39 ASMS hits released for Active Learning;
solubility proxy = ESOL logS (Delaney 2004; PROXY ONLY, the organizers' solubility criterion is not known) and PAINS flag.
Per set: number of ECFP4 Tanimoto 0.32 groups, distinct Murcko scaffolds, mean pairwise similarity, share of molecules below novelty thresholds.
Reference files (obtained from the challenge pages, not included in the repository): data/reference/DEL_hit_candidates_CACHE7.csv, data/raw/active_learning_data.csv. RDKit environment."""
import sys, os, json
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, Descriptors, Crippen, rdMolDescriptors as D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
KNOWN = {"CACHE7 cmp47 (PGK2)": "CNC(=O)C[C@H](Nc1nc(-c2cncc(OC)c2)nc2cc(OC)c(OC)cc12)c1ccc(C#N)cc1",
         "CACHE7 cmp45 (PGK1)": "CNC(=O)C[C@@H](Nc1nc(-c2ccc(OC)c(S(=O)(=O)N3CCOCC3)c2)nc2cc(OC)c(OC)cc12)c1ccc(C#N)cc1"}
gen = G.GetMorganGenerator(radius=2, fpSize=2048)
fp = lambda m: gen.GetFingerprint(m)
def esol(m):
    ha = m.GetNumHeavyAtoms(); ap = sum(a.GetIsAromatic() for a in m.GetAtoms()) / ha
    return 0.16 - 0.63 * Crippen.MolLogP(m) - 0.0062 * Descriptors.MolWt(m) + 0.066 * D.CalcNumRotatableBonds(m) - 0.74 * ap
def refs():
    out = {"R1 CACHE7 co-crystal ligands": [fp(Chem.MolFromSmiles(s)) for s in KNOWN.values()]}
    p = "data/reference/DEL_hit_candidates_CACHE7.csv"
    if os.path.exists(p):
        ms = [Chem.MolFromSmiles(s) for s in pd.read_csv(p).SMILES]; out["R2 CACHE7 public DEL hit candidates"] = [fp(m) for m in ms if m is not None]
    p = "data/raw/active_learning_data.csv"
    if os.path.exists(p):
        al = pd.read_csv(p); out["R3 ASMS hits released for Active Learning (39)"] = [fp(Chem.MolFromSmiles(s)) for s in al[al.Label == 1].RDKit_SMILES]
    return out
def run(df, scol, idcol=None):
    R = refs(); P = FilterCatalogParams(); P.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS); pains = FilterCatalog(P)
    rows, fps = [], []
    for i, r in df.iterrows():
        m = Chem.MolFromSmiles(r[scol])
        if m is None: rows.append({"id": r[idcol] if idcol else i, "smiles": r[scol], "valid": False}); fps.append(None); continue
        f = fp(m); fps.append(f)
        d = {"id": r[idcol] if idcol else i, "smiles": r[scol], "valid": True, "logS_ESOL": esol(m), "PAINS": pains.HasMatch(m), "MW": Descriptors.MolWt(m), "cLogP": Crippen.MolLogP(m),
             "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=m)}
        for k, v in R.items(): d["maxsim " + k] = max(DataStructs.BulkTanimotoSimilarity(f, v))
        rows.append(d)
    T = pd.DataFrame(rows); ok = [x for x in fps if x is not None]; n = len(ok); dist = []
    for i in range(1, n): dist += [1 - x for x in DataStructs.BulkTanimotoSimilarity(ok[i], ok[:i])]
    groups = len(Butina.ClusterData(dist, n, 1 - 0.32, isDistData=True)) if n > 1 else n
    sims = [x for i in range(1, n) for x in DataStructs.BulkTanimotoSimilarity(ok[i], ok[:i])]
    nov = [c for c in T.columns if c.startswith("maxsim ")]; v = T[T.valid]; S = {"molecules": int(n), "ECFP4_0.32_groups": int(groups), "distinct_Murcko_scaffolds": int(v.scaffold.nunique()),
         "mean_pairwise_ECFP4_similarity": round(float(np.mean(sims)), 3) if sims else None, "ESOL_logS_median": round(float(v.logS_ESOL.median()), 2),
         "ESOL_logS_below_-5": int((v.logS_ESOL < -5).sum()), "ESOL_logS_below_-6": int((v.logS_ESOL < -6).sum()), "PAINS": int(v.PAINS.sum())}
    for c in nov:
        S[c + " | median"] = round(float(v[c].median()), 2); S[c + " | max"] = round(float(v[c].max()), 2)
        for t in (0.3, 0.4, 0.5): S[c + f" | share < {t}"] = round(float((v[c] < t).mean()), 2)
    return T, S
if __name__ == "__main__":
    path = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + "_prospective"
    df = pd.read_csv(path); scol = [c for c in df.columns if 'smi' in c.lower()][0]; idcol = next((c for c in df.columns if c.lower() in ("catalogid", "id")), None)
    T, S = run(df, scol, idcol); T.to_csv(pre + "_per_molecule.csv", index=False); json.dump(S, open(pre + "_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
