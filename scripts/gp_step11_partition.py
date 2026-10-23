"""
Infer how the 14 identified hits split into series (clusters) from the Synapse (N_hits, N_clusters) of the 43 listed rounds.
Model: every hit belongs to one fixed series; N_clusters of a round = number of distinct series among its hits.
For a round with identified hits I (|I| of the round's N_hits h): distinct(I) <= c <= distinct(I) + (h - |I|); equality when all hits are identified.
Enumerates ALL set partitions of the 14 that satisfy every round (restricted-growth DFS with pruning). Run: python -m scripts.gp_step11_partition
"""
import json
import sys

sys.path.append(".")
from scripts.gp_step10_hit_constraints import ROUNDS, IDENT

sets = {r: [l.strip() for l in open(f"submissions/Team_candeleroj_{r}.txt") if l.strip()] for r in ROUNDS}
idx = {h: i for i, h in enumerate(IDENT)}
cons = []
for r, (c, h) in ROUNDS.items():
    I = sorted(idx[x] for x in sets[r] if x in idx)
    if I:
        cons.append((r, I, c, h))
print("rounds containing identified hits:", len(cons), "| fully resolved (all hits identified):", sum(1 for r, I, c, h in cons if len(I) == h))
by_max = {}
for con in cons:
    by_max.setdefault(max(con[1]), []).append(con)
sols = []
lab = [-1] * len(IDENT)


def ok(k):
    for r, I, c, h in by_max.get(k, []):
        d = len({lab[i] for i in I})
        if not (d <= c <= d + (h - len(I))):
            return False
    return True


def dfs(k, ncl):
    if k == len(IDENT):
        sols.append(tuple(lab)); return
    for l in range(ncl + 1):
        lab[k] = l
        if ok(k):
            dfs(k + 1, max(ncl, l + 1))
    lab[k] = -1


dfs(0, 0)
print("consistent partitions of the 14:", len(sols))
from collections import Counter
print("number of series among the 14 across solutions:", dict(Counter(len(set(s)) for s in sols)))
n = len(IDENT)
tog = [[sum(1 for s in sols if s[i] == s[j]) / max(len(sols), 1) for j in range(n)] for i in range(n)]
json.dump({"n_solutions": len(sols), "ident": IDENT, "together_fraction": tog, "first_solutions": [list(s) for s in sols[:20]]}, open("data/processed/gp/step15_partition.json", "w"), indent=1)
if sols:
    always = [(IDENT[i], IDENT[j]) for i in range(n) for j in range(i + 1, n) if tog[i][j] == 1.0]
    never = [(IDENT[i], IDENT[j]) for i in range(n) for j in range(i + 1, n) if tog[i][j] == 0.0]
    print("pairs ALWAYS in the same series:", len(always)); print("pairs NEVER in the same series:", len(never))
    print("undetermined pairs:", n * (n - 1) // 2 - len(always) - len(never))
    if len(sols) <= 5:
        for s in sols: print(dict(zip(IDENT, s)))
