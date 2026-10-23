"""
Candidates from the FULL library (validation + test = 428,960) inside series 1 and 2, ranked with the restrictions learned
in step 9 (negatives / neighbours of series 1 and 2):
  - the 2-substituent must be an exact pyridin-3-yl (pyridin-2-yl, pyridin-4-yl, methyl, H, other aryl = varies);
  - series 1 must be thieno[2,3-d]pyrimidine (thieno[3,2-d] = varies);
  - C4 substituent class is reported and used to sub-rank (hits carry N-alkyl / cyclic amines; the negatives inside the
    exact MCS carried N-aryl, S-linked and O/C-linked groups).
Priority: 1 = keeps 2-(pyridin-3-yl) [+ [2,3-d] for S1] and a hit-like C4 (N-alkyl or cyclic amine);
          2 = keeps the core restrictions but C4 like the negatives seen (N-aryl / S / O / C / halogen / H);
          3 = varies a restricted position (lower priority, expected inactive after step 9).
Only compounds NOT among the 13 hits, 2,269 confirmed negatives or 111 group-testing suspects are candidates.
For each candidate: submissions 1-100 it already appeared in (+ round outcome if known) and a Gamma-Poisson SECONDARY signal:
max Tanimoto (ECFP4 2048 bits, r=2) to the top-2000 DEL compounds ranked by NB-null enrichment (transfer option A of
step 6; that signal showed NO predictive power for the confirmed hits, AUC 0.42).
Run: python -m scripts.gp_step8_series_candidates   -> data/processed/gp/step8_candidates.csv, step8_summary.json
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdSubstructLibrary

RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
OUT = "data/processed/gp"

# 6-atom pyrimidine core; indices: 0 C2, 1 N3, 2 C4, 3 C4a, 4-6 thiophene, 7 C7a, 8 N1 (T23: S on C7a; T32: S on C4a)
T23 = Chem.MolFromSmarts("[#6]1:[#7]:[#6]:[#6]2:[#6]:[#6]:[#16]:[#6]:2:[#7]:1")
T32 = Chem.MolFromSmarts("[#6]1:[#7]:[#6]:[#6]2:[#16]:[#6]:[#6]:[#6]:2:[#7]:1")
# series-2 relaxed core: 0,1 chain C, 2 exocyclic N, 3 C4, 4 N3, 5 C2, 6 N1, 7 C8a/C6, 8 C4a/C5
R2 = Chem.MolFromSmarts("[#6&!R]-!@[#6&!R]-!@[#7&!R]-!@[#6]1:[#7]:[#6]:[#7]:[#6]:[#6]:1")
PYRIDINE_LIKE = {1: "piridin-2-ilo", 2: "piridin-3-ilo", 3: "piridin-4-ilo"}


def c2_class(mol, match, c2_idx, core):
    core = set(core)
    nbs = [n for n in mol.GetAtomWithIdx(c2_idx).GetNeighbors() if n.GetIdx() not in core]
    if not nbs:
        return "H"
    n = nbs[0]
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if n.GetIdx() in ring and len(ring) == 6 and not (set(ring) & core) and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            nn = [i for i, a in enumerate(ring) if mol.GetAtomWithIdx(a).GetSymbol() == "N"]
            if len(nn) == 1:
                i = ring.index(n.GetIdx())
                d = min(abs(i - nn[0]), 6 - abs(i - nn[0]))
                return PYRIDINE_LIKE.get(d, "otro")
            return "fenilo/arilo" if not nn else "diazinilo u otro heteroarilo 6"
    if n.GetIsAromatic():
        return "heteroarilo 5-miembros/otro aromatico"
    if n.GetSymbol() == "C" and not n.IsInRing():
        return "alquilo (p. ej. metilo)"
    if n.GetSymbol() in ("N", "O", "S"):
        return "N/O/S-enlazado"
    return "otro"


def c4_class(mol, c4_idx, core):
    core = set(core)
    nbs = [n for n in mol.GetAtomWithIdx(c4_idx).GetNeighbors() if n.GetIdx() not in core]
    if not nbs:
        return "H"
    n = nbs[0]
    s = n.GetSymbol()
    if s == "N":
        if any(x.GetIsAromatic() and x.GetIdx() != c4_idx for x in n.GetNeighbors()):
            return "N-arilo"
        return "N-ciclico" if n.IsInRing() else "N-alquilo"
    return {"S": "S-enlazado", "O": "O-enlazado", "C": "C-enlazado"}.get(s, "halogeno/otro")


def main():
    hits = pd.read_csv("docs/redteam_hits_confirmados.csv")
    negs = pd.read_csv("docs/redteam_negativos_confirmados.csv")
    H, N = set(hits.CatalogID), set(negs.CatalogID)
    S80 = {n: {l.strip() for l in open(f"submissions/Team_candeleroj_{n}.txt") if l.strip()} for n in (80, 81, 82)}
    SUS = set().union(*S80.values()) - H
    KNOWN = H | N | SUS

    pools = []
    for sp in ("Validation", "Test"):
        d = pd.read_csv(f"data/raw/Val-Test-set/PGK2_{sp}_split.csv")
        d["split"] = sp.lower()
        pools.append(d)
    pool = pd.concat(pools, ignore_index=True)
    mols, keep = [], []
    for i, s in enumerate(pool.SMILES):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            keep.append(i)
    pool = pool.iloc[keep].reset_index(drop=True)
    print(f"[*] library {len(mols):,}", flush=True)
    sl = rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.MolHolder(), rdSubstructLibrary.PatternHolder())
    for m in mols:
        sl.AddMol(m)

    step6 = json.load(open(f"{OUT}/step6_series.json"))["B"]["series"]
    mcs1, mcs2 = (Chem.MolFromSmarts(s["smarts"][0]) for s in step6[:2])
    ix = lambda q: set(sl.GetMatches(q, maxResults=1000000))
    m_t23, m_t32, m_r2, m_mcs1, m_mcs2 = ix(T23), ix(T32), ix(R2), ix(mcs1), ix(mcs2)
    print(f"[*] matches: thieno[2,3-d] core {len(m_t23)}, thieno[3,2-d] core {len(m_t32)}, S2-relaxed core {len(m_r2)}, "
          f"exact MCS S1 {len(m_mcs1)}, exact MCS S2 {len(m_mcs2)}", flush=True)

    rows = []
    for i in sorted(m_t23 | m_t32 | m_r2):
        mol, cid = mols[i], pool.CatalogID[i]
        series, core_name, c2, c4 = [], None, None, None
        if i in m_t23 or i in m_t32:
            patt, core_name = (T23, "tieno[2,3-d]pirimidina") if i in m_t23 else (T32, "tieno[3,2-d]pirimidina")
            match = mol.GetSubstructMatch(patt)
            c2, c4 = c2_class(mol, match, match[0], match), c4_class(mol, match[2], match)
            series.append("S1")
        if i in m_r2:
            match = mol.GetSubstructMatch(R2)
            core = match[3:]
            c2_s2 = c2_class(mol, match, match[5], core)
            if c2 is None:
                c2, c4, core_name = c2_s2, "N-alquilo", "pirimidina 4-(N-alquil)amino (cadena)"
            series.append("S2")
        keep_core = core_name is not None
        ok_c2 = c2 == "piridin-3-ilo"
        ok_core = not (i in m_t32 and i not in m_t23)  # thieno[3,2-d] varies the restricted core position
        hitlike = c4 in ("N-alquilo", "N-ciclico")
        if ok_c2 and ok_core and hitlike:
            pr = 1
        elif ok_c2 and ok_core:
            pr = 2
        else:
            pr = 3
        rows.append({"CatalogID": cid, "SMILES": pool.SMILES[i], "split": pool.split[i], "series": "+".join(series),
                     "core": core_name, "c2_substituent": c2, "c4_class": c4, "priority": pr,
                     "in_exact_mcs_S1": i in m_mcs1, "in_exact_mcs_S2": i in m_mcs2,
                     "known_hit": cid in H, "known_negative": cid in N, "known_suspect": cid in SUS})
    df = pd.DataFrame(rows)
    known = df[df.known_hit | df.known_negative | df.known_suspect]
    cand = df[~(df.known_hit | df.known_negative | df.known_suspect)].copy()
    print(f"[*] compounds in the relaxed series set: {len(df)}; already known: {len(known)} "
          f"(hits {int(df.known_hit.sum())}, negatives {int(df.known_negative.sum())}, suspects {int(df.known_suspect.sum())}); "
          f"new candidates: {len(cand)}", flush=True)

    # previous submissions
    where = {}
    for f in glob.glob("submissions/Team_candeleroj_*.txt"):
        r = re.search(r"_(\d+)\.txt$", f)
        if not r:
            continue
        for l in open(f):
            l = l.strip()
            if l:
                where.setdefault(l, []).append(int(r.group(1)))
    outc = pd.read_csv("data/processed/confirmed_submission_outcomes.csv").set_index("CatalogID")
    cand["prior_rounds"] = [";".join(map(str, sorted(where.get(c, [])))) for c in cand.CatalogID]
    cand["prior_round_outcome"] = [outc.round_outcome.get(c, "") if c in where else "" for c in cand.CatalogID]
    cand["previously_submitted"] = cand.prior_rounds != ""

    # Gamma-Poisson secondary signal
    ref = pd.read_csv(f"{OUT}/gp_top2000_ref.csv")
    rfp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in ref.SMILES]
    cand["gp_transfer_maxsim"] = [max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), rfp))
                                  for s in cand.SMILES]
    cand["gp_pct_within_candidates"] = cand.gp_transfer_maxsim.rank(pct=True).round(3)
    c4order = {"N-alquilo": 0, "N-ciclico": 1}
    cand["_c4"] = cand.c4_class.map(lambda x: c4order.get(x, 2))
    cand["_c2"] = cand.c2_substituent.map(lambda x: 0 if x == "piridin-3-ilo" else 1)
    cand = cand.sort_values(["priority", "_c4", "gp_transfer_maxsim"], ascending=[True, True, False]).drop(columns=["_c4", "_c2"])
    cand.to_csv(f"{OUT}/step8_candidates.csv", index=False)
    known.to_csv(f"{OUT}/step8_known_in_set.csv", index=False)
    summ = {"library": len(mols), "n_relaxed_set": len(df), "n_known": len(known), "n_candidates": len(cand),
            "matches": {"T23": len(m_t23), "T32": len(m_t32), "R2": len(m_r2), "mcs1": len(m_mcs1), "mcs2": len(m_mcs2)},
            "priority_counts": cand.priority.value_counts().sort_index().to_dict(),
            "by_series_priority": cand.groupby(["series", "priority"]).size().to_dict().__repr__(),
            "c2_counts": cand.c2_substituent.value_counts().to_dict(),
            "previously_submitted": int(cand.previously_submitted.sum()),
            "known_in_set_by_c2": known.c2_substituent.value_counts().to_dict(),
            "split_counts": cand.split.value_counts().to_dict()}
    json.dump(summ, open(f"{OUT}/step8_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(cand[["CatalogID", "split", "series", "c2_substituent", "c4_class", "priority", "previously_submitted",
                "gp_transfer_maxsim", "SMILES"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
