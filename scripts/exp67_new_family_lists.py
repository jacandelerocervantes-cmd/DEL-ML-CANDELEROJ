"""exp67: two validation lists to count the new hit clusters among the 60 candidate compounds that do not resemble any identified hit (Tanimoto < 0.32 to the 32 hits).
Inputs: exp65/exp66 outputs (candidates still possibly hit), confirmed outcomes v4 (hits and inactives), exp65 bounds (compounds forced inactive).
Each list = ~30 candidates (families kept together) + padding of compounds known to be inactive (first those forced inactive by the counts, then the confirmed ones), seed 67.
Lists must not repeat any earlier submission. Output: submissions/Team_candeleroj_180.txt and _181.txt (CRLF, no trailing newline, as the organizers' validation template). RDKit venv."""
import os, glob
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*')
rng = np.random.default_rng(67)
lab = pd.read_csv("data/processed/confirmed_submission_outcomes_v4.csv"); L = dict(zip(lab.CatalogID, lab.confirmed_label))
V = pd.read_csv("data/raw/Val-Test-set/PGK2_Validation_split.csv"); sc = [c for c in V.columns if 'smi' in c.lower()][0]; smi = dict(zip(V.CatalogID, V[sc])); vids = set(V.CatalogID)
g = G.GetMorganGenerator(radius=2, fpSize=2048)
H = [c for c, x in L.items() if x == 1]; fh = [g.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in H]
cf = pd.read_csv("data/processed/gp/exp65_candidate_families.csv")
cf["maxsim"] = [max(DataStructs.BulkTanimotoSimilarity(g.GetFingerprint(Chem.MolFromSmiles(smi[c])), fh)) for c in cf.CatalogID]
fam = cf.groupby("familia").agg(maxsim=("maxsim", "max")).reset_index(); new_fams = set(fam[fam.maxsim < 0.32].familia)
new = cf[cf.familia.isin(new_fams)].copy(); print("candidatos nuevos:", len(new), "| familias:", len(new_fams))
# repartir manteniendo familias juntas, equilibrado
groups = sorted([g_.CatalogID.tolist() for _, g_ in new.groupby("familia")], key=len, reverse=True); A, B = [], []
for gr in groups: (A if len(A) <= len(B) else B).extend(gr)
b = pd.read_csv("data/processed/gp/exp65_validation_unresolved_bounds.csv"); forced_inactive = sorted(b[b["max"] == 0].CatalogID)
confirmed_inactive = sorted(c for c, x in L.items() if x == 0 and c not in set(forced_inactive))
prev = [frozenset(l.strip() for l in open(f) if l.strip()) for f in glob.glob("submissions/Team_candeleroj_[0-9]*.txt") if f.replace("\\", "/").split("/")[-1][len("Team_candeleroj_"):-4].isdigit()]
pad_pool = forced_inactive + confirmed_inactive; used_pad = set()
def build(cands, name):
    need = 50 - len(cands); pool = [c for c in forced_inactive if c not in used_pad]
    pad = list(rng.choice(pool, min(need, len(pool)), replace=False)) if pool else []
    if len(pad) < need:
        rest = [c for c in confirmed_inactive if c not in used_pad]; pad += list(rng.choice(rest, need - len(pad), replace=False))
    used_pad.update(pad); ids = list(cands) + [str(x) for x in pad]; assert len(ids) == 50 and len(set(ids)) == 50 and all(i in vids for i in ids)
    assert frozenset(ids) not in prev, "lista repetida"
    fi = set(forced_inactive); assert all(L.get(c) == 0 or c in fi or c in set(cands) for c in ids)
    open(f"submissions/{name}", "w", newline="").write("\r\n".join(ids)); return ids
la = build(A, "Team_candeleroj_180.txt"); lb = build(B, "Team_candeleroj_181.txt")
print("lista 180:", len(A), "candidatos +", 50 - len(A), "inactivos conocidos | lista 181:", len(B), "candidatos +", 50 - len(B), "inactivos conocidos | solape entre listas:", len(set(la) & set(lb)))
print("relleno de la lista 180 forzado-inactivo (nuevo):", sum(c in set(forced_inactive) for c in la if c not in set(A)), "| lista 181:", sum(c in set(forced_inactive) for c in lb if c not in set(B)))
