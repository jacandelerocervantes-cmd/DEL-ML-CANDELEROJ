"""
Structure-based scoring for PGK2 (Vina) into the nucleotide (ATP) site defined by
the CACHE #7 co-crystal ligands: PGK2_cmp21 (chain A, LIG 501) and PGK1_cmp45
(chain A, LIG 601; PGK1/PGK2 ATP sites share the same contact residues).

  prep    build rigid PDBQT receptors + box (centroid of the co-crystal ligand)
  gate    control: do docking scores separate ATP-competed DEL hits from
          SIZE-MATCHED DEL non-hits? (Vina rewards size, so matching heavy-atom
          count is what makes AUC > 0.5 mean anything). Docking is only used
          downstream if this passes.
  screen  dock the top-N scored pool candidates into both receptors.

Usage: python -m scripts.dock_pgk2 prep | gate [--n 120] | screen [--n 2000]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

DOCK_DIR = "data/docking"
RECEPTORS = {
    "pgk2": {"pdb": f"{DOCK_DIR}/PGK2_cmp21.pdb", "chain": "A", "lig": "LIG", "lig_chain": "A"},
    "pgk1": {"pdb": f"{DOCK_DIR}/PGK1_cmp45.pdb", "chain": "A", "lig": "LIG", "lig_chain": "A"},
}
BOX = 24.0
EXHAUSTIVENESS = 4


def prep():
    meta = {}
    for name, r in RECEPTORS.items():
        prot, ligxyz = [], []
        for l in open(r["pdb"]):
            if l.startswith("ATOM") and l[21] == r["chain"] and l[16] in " A":
                prot.append(l)
            elif l.startswith("HETATM") and l[17:20].strip() == r["lig"] and l[21] == r["lig_chain"]:
                el = (l[76:78].strip() or l[12:16].strip()[0]).upper()
                if el != "H":
                    ligxyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        center = np.mean(ligxyz, axis=0).tolist()
        clean = f"{DOCK_DIR}/{name}_protein.pdb"
        open(clean, "w").write("".join(prot) + "END\n")
        pdbqt = f"{DOCK_DIR}/{name}.pdbqt"
        subprocess.run(["obabel", clean, "-xr", "-h", "-O", pdbqt], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta[name] = {"pdbqt": pdbqt, "center": center, "n_lig_atoms": len(ligxyz), "n_prot_atoms": len(prot)}
        print(f"[*] {name}: {len(prot)} protein atoms, ligand {len(ligxyz)} heavy atoms, "
              f"box centre {np.round(center, 1).tolist()}, size {BOX} A -> {pdbqt}", flush=True)
    json.dump(meta, open(f"{DOCK_DIR}/receptors.json", "w"))


def ligand_pdbqt(smiles, seed=0):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    if AllChem.EmbedMolecule(m, ps) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception:
        pass
    setups = MoleculePreparation().prepare(m)
    pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    return pdbqt if ok else None


def _dock_batch(items, meta, exhaustiveness):
    from vina import Vina
    vs = {}
    for name, r in meta.items():
        v = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)
        v.set_receptor(rigid_pdbqt_filename=r["pdbqt"])
        v.compute_vina_maps(center=r["center"], box_size=[BOX] * 3)
        vs[name] = v
    out = []
    for id_, smi in items:
        row = {"id": id_}
        try:
            pq = ligand_pdbqt(smi)
        except Exception:
            pq = None
        for name, v in vs.items():
            row[name] = np.nan
            if pq is None:
                continue
            try:
                v.set_ligand_from_string(pq)
                v.dock(exhaustiveness=exhaustiveness, n_poses=1)
                row[name] = float(v.energies(n_poses=1)[0][0])
            except Exception:
                pass
        out.append(row)
    return out


def run_docking(items, jobs):
    meta = json.load(open(f"{DOCK_DIR}/receptors.json"))
    batches = [items[i::jobs] for i in range(jobs)]
    t0 = time.time()
    res = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(_dock_batch)(b, meta, EXHAUSTIVENESS) for b in batches if b)
    df = pd.DataFrame([r for part in res for r in part])
    print(f"[*] docked {len(df)} ligands in {(time.time() - t0) / 60:.1f} min; "
          f"failures: {int(df[list(meta)].isna().all(axis=1).sum())}", flush=True)
    return df


def gate(n, jobs, labels_path, out_csv):
    from sklearn.metrics import roc_auc_score
    lab = pd.read_parquet(labels_path,
                          columns=["SMILES", "Label", "is_orthosteric", "count_PGK2"])
    rng = np.random.RandomState(0)
    pos = lab[(lab.is_orthosteric == 1) & (lab.count_PGK2 >= 10)].sample(700, random_state=0)
    neg = lab[lab.Label == 0].sample(40000, random_state=0)
    hac = lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
    pos["hac"] = [hac(s) for s in pos.SMILES]
    neg["hac"] = [hac(s) for s in neg.SMILES]
    pos = pos[(pos.hac >= 25) & (pos.hac <= 38)].head(n)
    negs = []
    for h in pos.hac:
        cand = neg[(neg.hac - h).abs() <= 1]
        pick = cand.sample(1, random_state=rng.randint(1e9))
        negs.append(pick.index[0])
        neg = neg.drop(pick.index)
    neg = lab.loc[negs].assign(hac=pos.hac.to_numpy())
    print(f"[*] gate set: {len(pos)} ATP-competed hits vs {len(neg)} size-matched non-hits | "
          f"mean HAC {pos.hac.mean():.1f} vs {neg.hac.mean():.1f}", flush=True)

    items = [(f"P{i}", s) for i, s in zip(pos.index, pos.SMILES)] + [(f"N{i}", s) for i, s in zip(neg.index, neg.SMILES)]
    df = run_docking(items, jobs)
    df["y"] = df["id"].str.startswith("P").astype(int)
    df.to_csv(out_csv, index=False)
    d = df.dropna()
    print(f"[*] gate: {len(d)} ligands docked in all receptors", flush=True)
    for name in ("pgk2", "pgk1"):
        print(f"    AUC({name}) = {roc_auc_score(d.y, -d[name]):.3f}   (score: hits {d[d.y == 1][name].mean():.2f} "
              f"vs non-hits {d[d.y == 0][name].mean():.2f} kcal/mol)", flush=True)
    both = -(d.pgk2 + d.pgk1)
    print(f"    AUC(ensemble mean) = {roc_auc_score(d.y, both):.3f}", flush=True)


def screen(n, jobs):
    cand = pd.read_csv("data/processed/v2_candidates.csv").head(n)
    items = list(zip(cand.CatalogID, cand.SMILES))
    df = run_docking(items, jobs)
    df.to_csv(f"{DOCK_DIR}/screen_scores.csv", index=False)
    print(f"[*] wrote {DOCK_DIR}/screen_scores.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "gate", "screen"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--labels", default="data/processed/PGK2_selection_labeled.parquet",
                     help="Labeled parquet to draw the gate's positive/negative sample from")
    ap.add_argument("--out", default=None, help="Output CSV for gate scores (default: data/docking/gate_scores.csv)")
    a = ap.parse_args()
    out_csv = a.out or f"{DOCK_DIR}/gate_scores.csv"
    {"prep": prep, "gate": lambda: gate(a.n, a.jobs, a.labels, out_csv), "screen": lambda: screen(a.n, a.jobs)}[a.mode]()
