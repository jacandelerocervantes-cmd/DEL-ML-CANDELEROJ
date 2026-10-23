"""Context for the LOO pharmacophore (2-(pyridin-3-yl)pyrimidine core): which negative matches, and how many of the
validation-pool matches already have a confirmed label (hits / confirmed negatives / never tested).
Run: python -m scripts.gp_step5b_pattern_context"""
import json

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sma = json.load(open("data/processed/gp/step5_loo.json"))["pattern_full13"]
q = Chem.MolFromSmarts(sma)
lig = pd.read_csv("data/docking/retro/ligands.csv")
neg = lig[lig.y == 0]
print("negatives (of the 130) matching:", [(r.CatalogID, r.SMILES) for r in neg.itertuples() if Chem.MolFromSmiles(r.SMILES).HasSubstructMatch(q)])
hits = set(pd.read_csv("docs/redteam_hits_confirmados.csv").CatalogID)
negall = pd.read_csv("docs/redteam_negativos_confirmados.csv")
negall_m = negall[[Chem.MolFromSmiles(s).HasSubstructMatch(q) for s in negall.SMILES]]
print(f"all {len(negall)} confirmed negatives: {len(negall_m)} match ->", negall_m.CatalogID.tolist())
for split in ("validation", "test"):
    d = pd.read_csv(f"data/raw/Val-Test-set/PGK2_{split.capitalize()}_split.csv")
    m = d[[Chem.MolFromSmiles(s).HasSubstructMatch(q) for s in d.SMILES]]
    m.to_csv(f"data/processed/gp/step5_pattern_matches_{split}.csv", index=False)
    ids = set(m.CatalogID)
    print(f"{split}: {len(m)} matches ({100*len(m)/len(d):.3f}% of pool); known hits among them: {len(ids & hits)}; "
          f"confirmed negatives: {len(ids & set(negall.CatalogID))}; unlabelled: {len(ids - hits - set(negall.CatalogID))}")
