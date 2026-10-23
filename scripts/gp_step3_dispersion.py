"""
Gamma-Poisson step 3a: three estimates of the NB dispersion phi (Var = mu + phi*mu^2), with
convergence diagnostics for the zero-truncated MLE.
  (a) method of moments, no trimming, on count_PGK2 of compounds with NTC == 0
  (b) NTC dispersion as common noise (trim 0.05%)
  (c) zero-truncated NB MLE on count_PGK2 (>=1) of compounds with NTC == 0
Run: python -m scripts.gp_step3_dispersion   -> data/processed/gp/step3_dispersion.json
"""
import json
import os
import sys

GP_DIR = os.environ.get("GP_DIR", "data/processed/gp")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from modules.denoising.stouffer import estimate_nb_dispersion


def nb_logpmf(k, mu, phi):
    r = 1.0 / phi
    return gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu))


def zt_nll(theta, k, w):
    mu, phi = np.exp(theta)
    r = 1.0 / phi
    log_p0 = r * np.log(r / (r + mu))
    p0 = np.exp(log_p0)
    if not np.isfinite(p0) or p0 >= 1.0 - 1e-12:
        return 1e30
    return -(w * (nb_logpmf(k, mu, phi) - np.log1p(-p0))).sum()


def main():
    u = pd.read_parquet(f"{GP_DIR}/step2_scores.parquet", columns=["count_PGK2", "count_NTC", "in_supplement_only"])
    sel = u[(~u.in_supplement_only) & (u.count_NTC == 0) & (u.count_PGK2 > 0)].count_PGK2
    ntc_all = u.count_NTC
    vc = sel.value_counts().sort_index()
    k, w = vc.index.to_numpy(float), vc.to_numpy(float)
    mean, var = sel.mean(), sel.var(ddof=1)
    out = {"n_compounds_ntc0": int(len(sel)), "mean": float(mean), "var": float(var),
           "share_count_1": float((sel == 1).mean()), "max": int(sel.max()),
           "count_distribution_head": {int(a): int(b) for a, b in vc.head(8).items()}}
    out["phi_a_moments_untrimmed"] = float(max((var - mean) / mean ** 2, 1e-8))
    out["phi_b_ntc_trim0.05pct"] = float(estimate_nb_dispersion(ntc_all.to_numpy(float), trim_frac=0.0005))

    best = None
    starts = [(np.log(mean), np.log(p)) for p in (0.01, 0.1, 1.0, 10.0)]
    fits = []
    for s in starts:
        res = minimize(zt_nll, s, args=(k, w), method="L-BFGS-B", bounds=[(-8, 8), (-12, 8)])
        fits.append({"start_phi": float(np.exp(s[1])), "mu": float(np.exp(res.x[0])), "phi": float(np.exp(res.x[1])),
                     "nll": float(res.fun), "success": bool(res.success), "at_bound": bool(abs(res.x[1] - (-12)) < 1e-6 or abs(res.x[1] - 8) < 1e-6)})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    out["mle_fits"] = fits
    mu_h, phi_h = np.exp(best[1])
    out["phi_c_mle_zero_truncated"], out["mu_c_mle"] = float(phi_h), float(mu_h)
    # profile likelihood over phi (mu re-optimised) to see how flat the surface is
    grid = np.logspace(-3, 1.5, 10)
    prof = []
    for p in grid:
        r = minimize(lambda t: zt_nll([t[0], np.log(p)], k, w), [np.log(mu_h)], method="L-BFGS-B", bounds=[(-8, 8)])
        prof.append({"phi": float(p), "nll_minus_min": float(r.fun - best[0])})
    out["profile_nll_minus_min"] = prof
    # rough 95% interval: phi where nll rises by 1.92
    ok = [x["phi"] for x in prof if x["nll_minus_min"] <= 1.92]
    out["phi_c_approx_95ci_from_profile_grid"] = [min(ok), max(ok)] if ok else None
    json.dump(out, open(f"{GP_DIR}/step3_dispersion.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
