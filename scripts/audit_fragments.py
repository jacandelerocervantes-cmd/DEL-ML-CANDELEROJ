"""
Biochemical audit of the top-weighted ECFP4 bits. For each bit, its frequency in:
  base (non-hit DEL), all hits, ORTHOSTERIC hits (inhibitor-competed => ATP site),
  non-orthosteric hits, bead/matrix binders (count_NTC>0 & count_PGK2>=3),
  promiscuous hitters (historic_hits>=5), reported as fold-enrichment vs base.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
BITS = [1883, 1798, 298, 1646, 503, 1178, 66, 1768, 1711, 1079, 713, 1320, 1311, 1549]


def main():
    lab = pq.read_table("data/processed/PGK2_selection_labeled.parquet",
                        columns=["SMILES", "Label", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC",
                                 "historic_hits"]).to_pandas()
    X = sp.load_npz("data/processed/pgk2_ecfp4_sparse.npz").tocsc()
    y = lab.Label.to_numpy() == 1
    ortho = y & (lab.count_PGK2_with_inhibitor.to_numpy() < 0.1 * lab.count_PGK2.to_numpy())
    nonortho = y & ~ortho
    bead = (lab.count_NTC.to_numpy() > 0) & (lab.count_PGK2.to_numpy() >= 3)
    prom = (lab.count_PGK2.to_numpy() >= 3) & (lab.historic_hits.to_numpy() >= 5)
    base = ~y & ~bead & ~prom
    groups = {"hits": y, "ortho": ortho, "non-ortho": nonortho, "bead(NTC>0)": bead, "promiscuous": prom}
    print("group sizes:", {k: int(v.sum()) for k, v in groups.items()}, "| base:", int(base.sum()), flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sample = lab[y].SMILES.sample(3000, random_state=0).tolist()
    frag = {}
    for smi in sample:
        m = Chem.MolFromSmiles(smi)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetCountFingerprint(m, additionalOutput=ao)
        bi = ao.GetBitInfoMap()
        for b in BITS:
            if b in bi and b not in frag:
                a, r = bi[b][0]
                env = Chem.FindAtomEnvironmentOfRadiusN(m, r, a)
                sub = Chem.PathToSubmol(m, env)
                frag[b] = Chem.MolToSmiles(sub) if sub.GetNumAtoms() else m.GetAtomWithIdx(a).GetSymbol()

    print(f"{'bit':>5} {'fold hits':>9} {'ortho':>7} {'non-ortho':>9} {'bead':>7} {'promisc':>8}  fragment", flush=True)
    for b in BITS:
        col = X[:, b]
        present = np.zeros(len(lab), dtype=bool)
        present[col.indices] = True
        fb = present[base].mean()
        row = [present[g].mean() / max(fb, 1e-9) for g in groups.values()]
        print(f"{b:>5} {row[0]:>9.1f} {row[1]:>7.1f} {row[2]:>9.1f} {row[3]:>7.1f} {row[4]:>8.1f}  {frag.get(b)}", flush=True)


if __name__ == "__main__":
    main()
