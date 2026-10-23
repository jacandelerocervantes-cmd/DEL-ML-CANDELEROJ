"""
Gamma-Poisson labeling, step 2: per-compound enrichment of PGK2 over the NTC-derived
background, Gamma posterior (mean + lower quantile), Negative-Binomial null z, Stouffer
aggregation per disynthon, and the inhibitor-competition feature (feature only, not a filter).

Model (per compound i):
  background rate in NTC units  mu_i = (c_NTC,i + a0) / (1 + b0)   Gamma(a0,b0) prior, moment-fit on the NTC counts
  expected PGK2 noise           E_i  = r * mu_i,  r = N_PGK2 / N_NTC  (PROXY: ratio of summed counts)
  enrichment theta_i            c_PGK2,i ~ Poisson(theta_i * E_i),  theta_i ~ Gamma(k, k)   (prior mean 1 = no enrichment)
  posterior                     theta_i | c ~ Gamma(c + k, E_i + k):  mean (c+k)/(E+k), lower 5% quantile
  NB null z                     z_i = Phi^-1(1 - P(X >= c | NB(mean=E_i, dispersion=phi)))
  disynthon                     z_AB, z_AC, z_BC = Stouffer sum(z)/sqrt(n) over compounds sharing the BB pair
  competition (feature)         s = log((c_PGK2 + a)/(c_inh + a)) and a depth-normalised variant

KNOWN BIASES (documented, not fixed): (1) N_PGK2 is under-estimated because the raw file is filtered to
count_PGK2 >= 1; (2) the PGK2 counts are zero-truncated, so phi is estimated on a truncated sample;
(3) disynthon aggregation only sees compounds present in the file (selection filter + NTC-only supplement).

Input : data/processed/gp/step1_union.parquet     Output: data/processed/gp/step2_scores.parquet, step2_report.json
Run   : python -m scripts.gp_step2_enrichment
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from modules.denoising.stouffer import enrichment_pvalue, estimate_nb_dispersion

GP_DIR = os.environ.get("GP_DIR", "data/processed/gp")
IN = f"{GP_DIR}/step1_union.parquet"
OUT = f"{GP_DIR}/step2_scores.parquet"
KAPPA = 1.0
A_COMP = 1.0
Q = 0.05


def stouffer_by(df, keys, z="z_cmp"):
    g = df.groupby(keys, sort=False)[z]
    s, n = g.transform("sum"), g.transform("size")
    return (s / np.sqrt(n)).astype("float32"), n.astype("int32")


def main():
    u = pd.read_parquet(IN)
    parts = u.compound.str.split("-", expand=True)
    u["lib"], u["b1"], u["b2"], u["b3"] = parts[0], parts[1], parts[2], parts[3]
    c, ntc, inh = u.count_PGK2.to_numpy(float), u.count_NTC.to_numpy(float), u.count_PGK2_with_inhibitor.to_numpy(float)

    r = c.sum() / ntc.sum()
    r_inh = c.sum() / max(inh.sum(), 1.0)
    m = ntc.mean()
    phi_ntc = estimate_nb_dispersion(ntc, trim_frac=0.0005)
    phi_ntc_var = ((ntc - m) ** 2).sum() / (len(ntc) - 1)
    a0 = 1.0 / max(phi_ntc, 1e-6)
    b0 = a0 / m
    mu = (ntc + a0) / (1.0 + b0)
    E = r * mu
    phi = estimate_nb_dispersion(c[(ntc == 0) & (c > 0)], trim_frac=0.01)

    post_mean = (c + KAPPA) / (E + KAPPA)
    post_q05 = gamma.ppf(Q, c + KAPPA, scale=1.0 / (E + KAPPA))
    p = enrichment_pvalue(c, E, dispersion=phi)
    z = norm.isf(p)
    u["z_cmp"] = z.astype("float32")
    u["post_mean"] = post_mean.astype("float32")
    u["post_q05"] = post_q05.astype("float32")
    u["E_noise"] = E.astype("float32")
    for name, keys in (("AB", ["lib", "b1", "b2"]), ("AC", ["lib", "b1", "b3"]), ("BC", ["lib", "b2", "b3"])):
        u[f"z_{name}"], u[f"n_{name}"] = stouffer_by(u, keys)
    u["z_disyn"] = ((u.z_AB + u.z_AC + u.z_BC) / np.sqrt(3)).astype("float32")
    u["s_comp"] = np.log((c + A_COMP) / (inh + A_COMP)).astype("float32")
    u["s_comp_depthnorm"] = np.log(((c + A_COMP) / c.sum()) / ((inh + A_COMP) / max(inh.sum(), 1.0))).astype("float32")

    keep = ["SMILES", "compound", "lib", "b1", "b2", "b3", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC",
            "historic_hits", "post_mean", "post_q05", "z_cmp", "E_noise", "z_AB", "z_AC", "z_BC", "z_disyn",
            "n_AB", "n_AC", "n_BC", "s_comp", "s_comp_depthnorm", "in_supplement_only"]
    u[keep].to_parquet(OUT, index=False)
    q = lambda a: {k: float(np.quantile(a, k)) for k in (0.5, 0.9, 0.99, 0.999)}
    rep = {
        "r_N_PGK2_over_N_NTC_proxy": r, "r_PGK2_over_inh_proxy": r_inh,
        "ntc_mean": m, "ntc_var": float(phi_ntc_var), "phi_ntc_trimmed": phi_ntc, "prior_a0": a0, "prior_b0": b0,
        "phi_pgk2_nb_null": phi, "kappa": KAPPA,
        "E_noise_quantiles": q(E), "z_cmp_quantiles": q(z), "post_q05_quantiles": q(post_q05),
        "n_post_q05_gt_1": int((post_q05 > 1).sum()), "n_z_cmp_gt_3": int((z > 3).sum()),
        "disynthon_sizes_median": {k: float(u[f"n_{k}"].median()) for k in ("AB", "AC", "BC")},
        "frac_disynthon_size_1": {k: float((u[f"n_{k}"] == 1).mean()) for k in ("AB", "AC", "BC")},
    }
    json.dump(rep, open(f"{GP_DIR}/step2_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
