"""exp64: how much does the DEL term change AL2's selection and ranking? Runs the exp60 selection rule twice, with DEL weight 0.3 (as submitted) and 0.0 (model only), same candidates and same rule,
and reports (a) overlap of the 50 selected, (b) overlap of the top 100 / 1,000 / 10,000 of the blend, (c) spread of the DEL z-score. Nothing is written except the printed summary. RDKit venv."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog('rdApp.*'); g2 = G.GetMorganGenerator(radius=2, fpSize=2048)
cnt = lambda ms: np.log1p(np.array([g2.GetCountFingerprintAsNumPy(m) for m in ms], dtype=np.float32))
tmpl = pd.read_csv(r"data/raw/active_learning_submission_example.csv"); ids = list(tmpl.CatalogID)
P = pd.read_csv("data/raw/Val-Test-set/PGK2_Test_split.csv"); sc = [c for c in P.columns if 'smi' in c.lower()][0]; smi = dict(zip(P.CatalogID, P[sc]))
t = pd.read_pickle("data/processed/del_sel.pkl"); pos = t[(t.count_PGK2 >= 5) & (t.count_NTC == 0)]; neg = t[t.count_PGK2 <= 1].sample(50000, random_state=0)
dm = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(cnt([Chem.MolFromSmiles(s) for s in list(pos.SMILES) + list(neg.SMILES)]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
ms = [Chem.MolFromSmiles(smi[c]) for c in ids]; sd = np.concatenate([dm.decision_function(cnt(ms[i:i + 20000])) for i in range(0, len(ms), 20000)]); print("DEL listo", flush=True)
base = pd.read_csv("data/processed/gp/exp48_scores_A6.csv").set_index("CatalogID").Score.reindex(ids).values
z = lambda a: (a - a.mean()) / a.std()
al1 = set(pd.read_csv("data/processed/gp/exp48_selection_A6.csv").CatalogID); fa = [g2.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in al1]

def select(W):
    blend = z(base) + W * z(sd); df = pd.DataFrame({"CatalogID": ids, "blend": blend}); df["smiles"] = [smi[c] for c in ids]
    top = df[~df.CatalogID.isin(al1)].sort_values("blend", ascending=False).head(6000).reset_index(drop=True); fp = [g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.smiles]
    top["sim_al1"] = [max(DataStructs.BulkTanimotoSimilarity(f, fa)) for f in fp]
    n = len(fp); d = []
    for i in range(1, n): d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(fp[i], fp[:i])]
    cl = np.zeros(n, int)
    for k, c in enumerate(Butina.ClusterData(d, n, 1 - 0.32, isDistData=True)): cl[list(c)] = k
    top["g"] = cl
    new = top[top.sim_al1 < 0.32]; newpick = list(new.groupby("g").head(1).sort_values("blend", ascending=False).head(18).CatalogID)
    rest = top[~top.CatalogID.isin(newpick)]; cap = {}; hp = []; seen = set()
    for r in rest.itertuples():
        if len(hp) >= 50 - len(newpick): break
        if cap.get(r.g, 0) >= 2: continue
        k = (MolToInchiKey(Chem.MolFromSmiles(r.smiles)) or "")[:14]
        if k in seen: continue
        cap[r.g] = cap.get(r.g, 0) + 1; hp.append(r.CatalogID); seen.add(k)
    return set(newpick + hp), set(newpick), set(hp), df

s3, n3, h3, d3 = select(0.3); s0, n0, h0, d0 = select(0.0)
print("DEL z-score: min %.2f, mediana %.2f, max %.2f, p99 %.2f" % (z(sd).min(), np.median(z(sd)), z(sd).max(), np.percentile(z(sd), 99)))
print("50 de AL2 con DEL 0.3 frente a sin DEL: comparten %d de 50 | de los 18 de grupos nuevos comparten %d | de los 32 de alta probabilidad comparten %d" % (len(s3 & s0), len(n3 & n0), len(h3 & h0)))
b3 = z(base) + 0.3 * z(sd); b0 = z(base)
for k in (50, 100, 1000, 10000):
    a = set(np.array(ids)[np.argsort(-b3)[:k]]); b = set(np.array(ids)[np.argsort(-b0)[:k]])
    print("top %d del blend con 0.3 frente a sin DEL: comparten %d (%.0f%%)" % (k, len(a & b), 100 * len(a & b) / k))
print("correlacion de rangos entre modelo y DEL (Spearman): %.3f" % pd.Series(base).corr(pd.Series(sd), method="spearman"))
