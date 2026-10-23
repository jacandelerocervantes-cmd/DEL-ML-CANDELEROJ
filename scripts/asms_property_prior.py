"""
Property prior from ASMS-confirmed hits: log-likelihood-ratio per property bin
(hit vs non-hit) using the organizer-provided Challenge #1 (WDR91) ASMS pool
(138 confirmed hits among 339,258). It says what confirmed ASMS hits look like
physically, independent of DEL chemistry. Weak by construction (other target,
few hits), so it is shrunk by 0.5 and used with a small weight downstream.

Output: data/processed/asms_property_prior.json
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "FractionCSP3"]
N_BINS = 5
SHRINK = 0.5


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Lipinski.NumHDonors(m),
            Lipinski.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcFractionCSP3(m)]


def main():
    import pandas as pd
    t = pd.read_csv("data/processed/wdr91_asms_sample.csv")
    hits = t[t.LABEL == 1]["SMILES"].tolist()
    non = t[t.LABEL == 0]["SMILES"].tolist()
    H = np.array([p for p in map(props, hits) if p is not None])
    N = np.array([p for p in map(props, non) if p is not None])
    print(f"hits={len(H)} nonhits(sample)={len(N)}")

    prior = {"names": NAMES, "edges": [], "log_lr": []}
    for j, name in enumerate(NAMES):
        edges = np.unique(np.quantile(N[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1]))
        hb = np.bincount(np.searchsorted(edges, H[:, j]), minlength=len(edges) + 1)
        nb = np.bincount(np.searchsorted(edges, N[:, j]), minlength=len(edges) + 1)
        k = len(edges) + 1
        lr = np.log((hb + 1) / (hb.sum() + k)) - np.log((nb + 1) / (nb.sum() + k))
        lr = SHRINK * lr
        prior["edges"].append(edges.tolist())
        prior["log_lr"].append(lr.tolist())
        print(f"{name:18s} log-LR per bin: {np.round(lr, 2).tolist()}")
    os.makedirs("data/processed", exist_ok=True)
    json.dump(prior, open("data/processed/asms_property_prior.json", "w"))
    print("saved data/processed/asms_property_prior.json")


if __name__ == "__main__":
    main()
