"""
Gamma-Poisson step 4 (option A): parameter-free DEL -> ASMS transfer by similarity.
Score of an ASMS compound = max Tanimoto (ECFP4, 2048 bits, r=2) to the top-K DEL compounds ranked by
enrichment. Fire test: 13 confirmed ASMS hits vs 130 confirmed negatives (same set as the docking control).

Reference-set rankings (one per dispersion variant of the NB null, E from step 2):
  a: phi=0.735 (moments)   b: phi=0.599 (NTC)   c: phi=359.9 (zero-truncated MLE)   q05: Gamma posterior lower quantile
Controls: negatives vs the same top-K (built into the AUC), and K RANDOM DEL compounds (count_PGK2>=1), 5 draws.
Ties in the ranking key are broken by (count_PGK2, post_q05, seed-0 random).

Run: python -m scripts.gp_step4_transfer_A  -> data/processed/gp/step4_transferA.json
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import mannwhitneyu, norm, spearmanr
from sklearn.metrics import roc_auc_score

from modules.denoising.stouffer import enrichment_pvalue

RDLogger.DisableLog("rdApp.*")
PHI = {"a_moments": 0.7348549395157504, "b_ntc": 0.598991408596019, "c_mle_zt": 359.88390465921833}
KS = (500, 2000, 10000)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fps(smiles):
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def sim_scores(lig_fps, ref_fps):
    m = np.array([DataStructs.BulkTanimotoSimilarity(f, ref_fps) for f in lig_fps])
    top3 = -np.sort(-m, axis=1)[:, :3].mean(axis=1)
    return m.max(axis=1), top3


def metrics(y, s, rng):
    auc = roc_auc_score(y, s)
    bs = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], s[i]))
    p = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue
    rank = (-s).argsort().argsort() + 1  # 1 = highest similarity
    return {"auc": float(auc), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "mwu_p": float(p), "hit_ranks": sorted(int(r) for r in rank[y == 1]),
            "median_hit_rank": float(np.median(rank[y == 1]))}


def main():
    lig = pd.read_csv("data/docking/retro/ligands.csv")
    y = lig.y.to_numpy()
    lig_fps = fps(lig.SMILES)
    u = pd.read_parquet("data/processed/gp/step2_scores.parquet",
                        columns=["SMILES", "count_PGK2", "E_noise", "post_q05", "in_supplement_only"])
    u = u[~u.in_supplement_only].reset_index(drop=True)
    rng_tie = np.random.RandomState(0)
    u["tie"] = rng_tie.rand(len(u))
    cand = u[u.count_PGK2 >= 2].copy()  # top-K (K<=10000) is entirely within c>=2
    print(f"[*] DEL compounds {len(u):,}; candidates for the reference sets (c>=2): {len(cand):,}")
    keys = {}
    for name, phi in PHI.items():
        p = enrichment_pvalue(cand.count_PGK2.to_numpy(float), cand.E_noise.to_numpy(float), dispersion=phi)
        cand[f"z_{name}"] = norm.isf(p)
        keys[name] = f"z_{name}"
    keys["q05"] = "post_q05"
    rng = np.random.RandomState(1)
    out = {"phi": PHI, "n_lig": len(lig), "n_hits": int(y.sum()), "results": {}, "top_sets_overlap": {}}
    sel_sets, sims = {}, {}
    for kname, col in keys.items():
        order = cand.sort_values([col, "count_PGK2", "post_q05", "tie"], ascending=False)
        for K in KS:
            top = order.head(K)
            sel_sets[(kname, K)] = set(top.SMILES)
            mx, t3 = sim_scores(lig_fps, fps(top.SMILES))
            sims[(kname, K)] = mx
            for sc_name, sc in (("max", mx), ("top3mean", t3)):
                r = metrics(y, sc, rng)
                r.update({"mean_hits": float(sc[y == 1].mean()), "mean_neg": float(sc[y == 0].mean()),
                          "min_c_in_set": int(top.count_PGK2.min()), "median_c_in_set": float(top.count_PGK2.median())})
                out["results"][f"{kname}|K={K}|{sc_name}"] = r
    # random-K controls
    allu = u[u.count_PGK2 >= 1]
    for K in KS:
        aucs, mh, mn = [], [], []
        for d in range(5):
            ref = allu.sample(K, random_state=100 + d)
            mx, _ = sim_scores(lig_fps, fps(ref.SMILES))
            aucs.append(roc_auc_score(y, mx)); mh.append(mx[y == 1].mean()); mn.append(mx[y == 0].mean())
        out["results"][f"random|K={K}|max"] = {"auc_mean": float(np.mean(aucs)), "auc_min": float(min(aucs)),
                                               "auc_max": float(max(aucs)), "mean_hits": float(np.mean(mh)), "mean_neg": float(np.mean(mn))}
    # stability across dispersion variants (K=2000): overlap of top sets and per-hit score changes
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sel_sets[(names[i], 2000)], sel_sets[(names[j], 2000)]
            out["top_sets_overlap"][f"{names[i]}~{names[j]}"] = len(a & b) / 2000
    h = np.where(y == 1)[0]
    M = np.array([sims[(n, 2000)][h] for n in names])
    out["hit_maxsim_K2000"] = {n: M[i].round(3).tolist() for i, n in enumerate(names)}
    out["hit_maxsim_mean_abs_diff_vs_a"] = {n: float(np.abs(M[i] - M[0]).mean()) for i, n in enumerate(names)}
    out["hit_maxsim_spearman_vs_a"] = {n: float(spearmanr(M[i], M[0]).correlation) if np.std(M[i]) > 0 else None for i, n in enumerate(names)}
    json.dump(out, open("data/processed/gp/step4_transferA.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else v) for k, v in out.items() if k != "results"}, indent=1))
    print("\nvariant|K|score  AUC [CI95]  MWU p  mean_hits/mean_neg  median_hit_rank(of 143)")
    for k, r in out["results"].items():
        if "auc" in r:
            print(f"{k:28s} {r['auc']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] p={r['mwu_p']:.3f}  {r['mean_hits']:.3f}/{r['mean_neg']:.3f}  {r['median_hit_rank']:.0f}  minc={r['min_c_in_set']}")
        else:
            print(f"{k:28s} AUC mean {r['auc_mean']:.3f} (min {r['auc_min']:.3f}, max {r['auc_max']:.3f})  {r['mean_hits']:.3f}/{r['mean_neg']:.3f}")


if __name__ == "__main__":
    main()
