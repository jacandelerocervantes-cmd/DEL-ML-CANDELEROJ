"""
Retroactive docking control: do the 13 confirmed ASMS hits dock better than
size-matched confirmed ASMS negatives (same Vina setup as scripts/dock_pgk2.py)?

Input : data/docking/retro/ligands.csv (CatalogID, SMILES, hac, y) -- 13 hits +
        10 confirmed negatives per hit matched on heavy-atom count (|dHAC|<=1).
Output: data/docking/retro/retro_scores.csv (id, pgk2, pgk1), then AUC / rank report.

Usage: python -m scripts.dock_retro dock [--jobs 4] | report
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.dock_pgk2 import run_docking

LIG = "data/docking/retro/ligands.csv"
OUT = "data/docking/retro/retro_scores.csv"


def dock(jobs):
    lig = pd.read_csv(LIG)
    df = run_docking(list(zip(lig.CatalogID, lig.SMILES)), jobs)
    df.to_csv(OUT, index=False)
    print(f"[*] wrote {OUT}", flush=True)


def report():
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    lig = pd.read_csv(LIG)
    s = pd.read_csv(OUT).rename(columns={"id": "CatalogID"})
    d = lig.merge(s, on="CatalogID").dropna(subset=["pgk2", "pgk1"])
    print(f"[*] {len(d)} docked ({int(d.y.sum())} hits, {int((d.y == 0).sum())} negatives)")
    rng = np.random.RandomState(0)
    for name, sc in (("pgk2", -d.pgk2), ("pgk1", -d.pgk1), ("ens(-(pgk2+pgk1))", -(d.pgk2 + d.pgk1)),
                     ("sel(pgk1-pgk2)", d.pgk1 - d.pgk2)):
        auc = roc_auc_score(d.y, sc)
        bs = []
        for _ in range(2000):
            i = rng.randint(0, len(d), len(d))
            if d.y.iloc[i].nunique() == 2:
                bs.append(roc_auc_score(d.y.iloc[i], sc.iloc[i]))
        p = mannwhitneyu(sc[d.y == 1], sc[d.y == 0], alternative="greater").pvalue
        print(f"    AUC({name}) = {auc:.3f}  IC95 boot [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  MWU p={p:.3g}")
    d["rank_pgk2"] = (-d.pgk2).rank(ascending=False)
    print(d[d.y == 1][["CatalogID", "hac", "pgk2", "pgk1", "rank_pgk2"]].sort_values("pgk2").to_string(index=False))
    print(f"    mean score hits {d[d.y == 1].pgk2.mean():.2f} vs negatives {d[d.y == 0].pgk2.mean():.2f} kcal/mol (pgk2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dock", "report"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    dock(a.jobs) if a.mode == "dock" else report()
