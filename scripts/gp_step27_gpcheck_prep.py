"""
Prepare the reference sets for the full Gamma-Poisson transfer check (no RDKit needed): top-K DEL compounds by NB-null enrichment (phi = moments estimate) and by the Gamma posterior
lower quantile, plus RANDOM DEL reference sets of the same sizes as controls. Uses the corrected step-2 scores (empty-SMILES rows excluded): data/processed/gp_clean/step2_scores.parquet.
Output: data/processed/gp/gpcheck_refs.csv  (columns: set, SMILES)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(".")
from modules.denoising.stouffer import enrichment_pvalue

PHI_A = 0.7348549395157504
u = pd.read_parquet("data/processed/gp_clean/step2_scores.parquet", columns=["SMILES", "count_PGK2", "E_noise", "post_q05", "in_supplement_only"])
u = u[~u.in_supplement_only & u.SMILES.notna() & (u.SMILES.str.strip() != "")].reset_index(drop=True)
rng = np.random.RandomState(0)
u["tie"] = rng.rand(len(u))
c = u[u.count_PGK2 >= 2].copy()
c["z"] = norm.isf(enrichment_pvalue(c.count_PGK2.to_numpy(float), c.E_noise.to_numpy(float), dispersion=PHI_A))
rows = []
for K in (500, 2000, 10000):
    top = c.sort_values(["z", "count_PGK2", "post_q05", "tie"], ascending=False).head(K)
    rows += [("top_z_K%d" % K, s) for s in top.SMILES]
    topq = c.sort_values(["post_q05", "count_PGK2", "z", "tie"], ascending=False).head(K)
    rows += [("top_q05_K%d" % K, s) for s in topq.SMILES]
allc = u[u.count_PGK2 >= 1]
for K, n in ((500, 10), (2000, 10), (10000, 3)):
    for d in range(n):
        rows += [("random_K%d_%d" % (K, d), s) for s in allc.sample(K, random_state=100 + d).SMILES]
df = pd.DataFrame(rows, columns=["set", "SMILES"])
df.to_csv("data/processed/gp/gpcheck_refs.csv", index=False)
print(df.groupby("set").size().to_string())
print("count_PGK2 min in top_z_K10000:", int(c.sort_values(['z', 'count_PGK2', 'post_q05', 'tie'], ascending=False).head(10000).count_PGK2.min()))
