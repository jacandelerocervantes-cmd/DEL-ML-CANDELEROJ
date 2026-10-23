"""Prepares the small reference set used as the Gamma-Poisson 'secondary signal' for ASMS candidates (no RDKit needed):
the top-2000 DEL compounds by NB-null enrichment z (phi = moments estimate, same construction/tie-break as
scripts/gp_step4_transfer_A.py, K=2000). Output: data/processed/gp/gp_top2000_ref.csv (SMILES, count_PGK2, z, post_q05)."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from modules.denoising.stouffer import enrichment_pvalue

PHI_A = 0.7348549395157504
u = pd.read_parquet("data/processed/gp/step2_scores.parquet",
                    columns=["SMILES", "count_PGK2", "E_noise", "post_q05", "in_supplement_only"])
u = u[~u.in_supplement_only & u.SMILES.notna() & (u.SMILES.str.strip() != "")].reset_index(drop=True)
u["tie"] = np.random.RandomState(0).rand(len(u))
c = u[u.count_PGK2 >= 2].copy()
c["z"] = norm.isf(enrichment_pvalue(c.count_PGK2.to_numpy(float), c.E_noise.to_numpy(float), dispersion=PHI_A))
top = c.sort_values(["z", "count_PGK2", "post_q05", "tie"], ascending=False).head(2000)
top[["SMILES", "count_PGK2", "z", "post_q05"]].to_csv("data/processed/gp/gp_top2000_ref.csv", index=False)
print(len(top), "rows; min count", int(top.count_PGK2.min()), "z range", round(top.z.min(), 2), round(top.z.max(), 2))
