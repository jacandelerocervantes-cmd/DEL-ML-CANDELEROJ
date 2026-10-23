"""
Substrate-site (3-phosphoglycerate) exploration -- SEPARATE line of work from the ATP-site series.

Receptor: data/reference/CACHE7_PDB/CACHE7_PDB/PGK2_cmp47.pdb (PGK2 chain A with 3PG bound in the substrate site and the
compound LIG in the ATP site). Box centre = centroid of the 3PG heavy atoms; protein only (ATOM, chain A).

  fg      structural filter without docking: anionic / phosphate-carboxylate mimetic groups (carboxylic acid, phosphonic /
          phosphoric acid, sulfonic acid, NH-tetrazole; extra: acyl sulfonamide) in the test and validation pools, in the 2,269
          confirmed negatives, the 13 hits and the 111 group-testing suspects
  prep    build the rigid PDBQT receptor (needs obabel)
  redock  control: dock 3PG itself and compare the pose with the crystal (P atom and heavy-atom centroid distance)
  retro   dock the 13 confirmed hits + 130 size-matched confirmed negatives (same set as the ATP-site control) and report
          AUC / bootstrap CI / Mann-Whitney / ranks
Same Vina set-up as scripts/dock_pgk2.py (exhaustiveness 4, seed 42, 24 A box).
Run on the VM: python -m scripts.gp_step9_3pg all
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu, rankdata

from scripts import dock_pgk2

RDLogger.DisableLog("rdApp.*")
OUT = os.environ.get("GP_DIR", "data/processed/gp")
D3 = "data/docking/site3pg"
SRC_PDB = "data/docking/PGK2_cmp47.pdb"
CRYSTAL_P = None

FG = {
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",
    "phosphonic_or_phosphoric_acid": "[PX4](=O)[OX2H1,OX1-]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1,OX1-]",
    "tetrazole_NH": "[#6]1:[#7]:[#7]:[#7]:[#7H]:1",
    "tetrazole_NH_2H": "[#6]1:[#7]:[#7]:[#7H]:[#7]:1",
    "extra_acyl_sulfonamide": "[CX3](=O)[NX3H1][SX4](=O)(=O)",
    "extra_phosphonate_ester_neutral": "[PX4](=O)(O[#6])O[#6]",
}
FG_ANIONIC = ["carboxylic_acid", "phosphonic_or_phosphoric_acid", "sulfonic_acid", "tetrazole_NH", "tetrazole_NH_2H"]


def load_sets():
    hits = pd.read_csv("docs/redteam_hits_confirmados.csv")[["CatalogID", "SMILES"]]
    negs = pd.read_csv("docs/redteam_negativos_confirmados.csv")[["CatalogID", "SMILES"]]
    H = set(hits.CatalogID)
    sus_ids = set()
    for n in (80, 81, 82):
        sus_ids |= {l.strip() for l in open(f"submissions/Team_candeleroj_{n}.txt") if l.strip()}
    sus_ids -= H
    pools = {}
    for sp in ("Validation", "Test"):
        pools[sp.lower()] = pd.read_csv(f"data/raw/Val-Test-set/PGK2_{sp}_split.csv")
    allp = pd.concat(pools.values(), ignore_index=True).set_index("CatalogID").SMILES
    sus = pd.DataFrame({"CatalogID": sorted(sus_ids), "SMILES": [allp.get(i, "") for i in sorted(sus_ids)]})
    return hits, negs, sus, pools


def flag(df):
    pats = {k: Chem.MolFromSmarts(v) for k, v in FG.items()}
    rows = []
    for cid, smi in zip(df.CatalogID, df.SMILES):
        m = Chem.MolFromSmiles(smi)
        r = {"CatalogID": cid, "SMILES": smi}
        for k, p in pats.items():
            r[k] = bool(m is not None and m.HasSubstructMatch(p))
        rows.append(r)
    f = pd.DataFrame(rows)
    f["tetrazole_NH"] = f.tetrazole_NH | f.tetrazole_NH_2H
    f = f.drop(columns=["tetrazole_NH_2H"])
    f["anionic_any"] = f[[c for c in FG_ANIONIC if c in f.columns]].any(axis=1)
    return f


def fg():
    hits, negs, sus, pools = load_sets()
    where = {}
    for fpath in glob.glob("submissions/Team_candeleroj_*.txt"):
        r = re.search(r"_(\d+)\.txt$", fpath)
        if r:
            for l in open(fpath):
                l = l.strip()
                if l:
                    where.setdefault(l, []).append(int(r.group(1)))
    outc = pd.read_csv("data/processed/confirmed_submission_outcomes.csv").set_index("CatalogID")
    res, summ = {}, {}
    for name, df in (("test", pools["test"]), ("validation", pools["validation"]), ("negatives_2269", negs),
                     ("hits_13", hits), ("suspects_111", sus)):
        f = flag(df)
        res[name] = f
        cols = [c for c in f.columns if c not in ("CatalogID", "SMILES")]
        summ[name] = {"n": len(f), **{c: int(f[c].sum()) for c in cols},
                      "anionic_any_pct": round(100 * float(f.anionic_any.mean()), 3)}
        print(name, json.dumps(summ[name]), flush=True)
    for name in ("test", "validation"):
        f = res[name]
        a = f[f.anionic_any].copy()
        a["prior_rounds"] = [";".join(map(str, sorted(where.get(c, [])))) for c in a.CatalogID]
        a["prior_round_outcome"] = [outc.round_outcome.get(c, "") if c in where else "" for c in a.CatalogID]
        a.to_csv(f"{OUT}/step12_anionic_{name}.csv", index=False)
    fn = res["negatives_2269"]
    fn[fn.anionic_any].to_csv(f"{OUT}/step12_anionic_confirmed_negatives.csv", index=False)
    val_ids = set(res["validation"][res["validation"].anionic_any].CatalogID)
    summ["validation_anionic_already_tested"] = {
        "confirmed_negatives": len(val_ids & set(negs.CatalogID)), "hits": len(val_ids & set(hits.CatalogID)),
        "suspects": len(val_ids & set(sus.CatalogID)), "ever_submitted": len(val_ids & set(where))}
    json.dump(summ, open(f"{OUT}/step12_fg_summary.json", "w"), indent=1)
    print(json.dumps(summ["validation_anionic_already_tested"]))


def prep():
    os.makedirs(D3, exist_ok=True)
    prot, lig = [], []
    for l in open(SRC_PDB):
        if l.startswith("ATOM") and l[21] == "A" and l[16] in " A":
            prot.append(l)
        elif l.startswith("HETATM") and l[17:20].strip() == "3PG" and l[21] == "A":
            el = (l[76:78].strip() or l[12:16].strip()[0]).upper()
            if el != "H":
                lig.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    center = np.mean(lig, axis=0).tolist()
    clean = f"{D3}/pgk2_3pg_protein.pdb"
    open(clean, "w").write("".join(prot) + "END\n")
    pdbqt = f"{D3}/pgk2_3pg.pdbqt"
    subprocess.run(["obabel", clean, "-xr", "-h", "-O", pdbqt], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    meta = {"pgk2_3pg": {"pdbqt": pdbqt, "center": center, "n_lig_atoms": len(lig), "n_prot_atoms": len(prot)}}
    json.dump(meta, open(f"{D3}/receptors.json", "w"))
    print(f"[*] 3-PG site receptor: {len(prot)} protein atoms; 3PG heavy atoms {len(lig)}; centre {np.round(center, 2).tolist()}", flush=True)
    return meta


def redock():
    from vina import Vina
    meta = json.load(open(f"{D3}/receptors.json"))["pgk2_3pg"]
    crys = []
    for l in open(SRC_PDB):
        if l.startswith("HETATM") and l[17:20].strip() == "3PG":
            el = (l[76:78].strip() or l[12:16].strip()[0]).upper()
            if el != "H":
                crys.append((el, np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    cP = [c for e, c in crys if e == "P"][0]
    cC = np.mean([c for _, c in crys], axis=0)
    smiles = {"3PG_dianion_trianion": "OC(COP(=O)([O-])[O-])C(=O)[O-]", "3PG_neutral": "OC(COP(=O)(O)O)C(=O)O"}
    out = {}
    for name, smi in smiles.items():
        try:
            pq = dock_pgk2.ligand_pdbqt(smi)
            v = Vina(sf_name="vina", cpu=4, seed=42, verbosity=0)
            v.set_receptor(rigid_pdbqt_filename=meta["pdbqt"])
            v.compute_vina_maps(center=meta["center"], box_size=[dock_pgk2.BOX] * 3)
            v.set_ligand_from_string(pq)
            v.dock(exhaustiveness=16, n_poses=9)
            en = v.energies(n_poses=9)
            poses = v.poses(n_poses=9).split("ENDMDL")
            best = []
            for i, blk in enumerate(p for p in poses if "ATOM" in p):
                xyz, el = [], []
                for l in blk.splitlines():
                    if l.startswith(("ATOM", "HETATM")):
                        e = l[76:78].strip() or l[12:16].strip()[0]
                        if e.upper() not in ("H", "HD"):
                            xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                            el.append(e.upper())
                xyz = np.array(xyz)
                pP = xyz[[j for j, e in enumerate(el) if e == "P"][0]]
                best.append({"pose": i + 1, "energy": float(en[i][0]), "P_dist_A": float(np.linalg.norm(pP - cP)),
                             "heavy_centroid_dist_A": float(np.linalg.norm(xyz.mean(axis=0) - cC))})
            out[name] = best
            print(name, [(b["pose"], round(b["energy"], 2), round(b["P_dist_A"], 2), round(b["heavy_centroid_dist_A"], 2)) for b in best], flush=True)
        except Exception as ex:
            out[name] = {"error": repr(ex)}
            print(name, "FAILED", repr(ex), flush=True)
    json.dump(out, open(f"{OUT}/step12_redock_3pg.json", "w"), indent=1)


def auc(y, s):
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def retro(jobs=4):
    meta = json.load(open(f"{D3}/receptors.json"))
    lig = pd.read_csv("data/docking/retro/ligands.csv")
    items = list(zip(lig.CatalogID, lig.SMILES))
    batches = [items[i::jobs] for i in range(jobs)]
    res = Parallel(n_jobs=jobs, backend="loky", verbose=5)(delayed(dock_pgk2._dock_batch)(b, meta, dock_pgk2.EXHAUSTIVENESS) for b in batches if b)
    df = pd.DataFrame([r for part in res for r in part])
    df.to_csv(f"{OUT}/step12_retro_3pg_scores.csv", index=False)
    d = lig.merge(df.rename(columns={"id": "CatalogID"}), on="CatalogID").dropna(subset=["pgk2_3pg"])
    y = d.y.to_numpy()
    s = -d.pgk2_3pg.to_numpy()
    rng = np.random.RandomState(0)
    bs = []
    for _ in range(2000):
        i = rng.randint(0, len(d), len(d))
        if y[i].min() != y[i].max():
            bs.append(auc(y[i], s[i]))
    p = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue
    rank = (-s).argsort().argsort() + 1
    rep = {"n": len(d), "n_hits": int(y.sum()), "auc": float(auc(y, s)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "mwu_p": float(p), "hit_ranks_of_n": sorted(int(r) for r in rank[y == 1]),
           "mean_energy_hits": float(d[d.y == 1].pgk2_3pg.mean()), "mean_energy_negs": float(d[d.y == 0].pgk2_3pg.mean())}
    json.dump(rep, open(f"{OUT}/step12_retro_3pg_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fg", "prep", "redock", "retro", "all"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.mode in ("fg", "all"):
        fg()
    if a.mode in ("prep", "all"):
        prep()
    if a.mode in ("redock", "all"):
        redock()
    if a.mode in ("retro", "all"):
        retro(a.jobs)
