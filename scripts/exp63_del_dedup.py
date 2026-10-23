"""exp63: does deduplicating the DEL data by structure (as the organizers recommend: sum counts per SMILES) change the signal of the DEL model on the 39 labelled test hits?
Same protocol as exp8/exp51 (logistic regression C=0.05 balanced, ECFP4 radius 2 count fingerprints 2048 bits, 50,000 random negatives with seed 0, positives = count_PGK2 >= 5 and count_NTC == 0),
run twice: (a) rows as released, (b) after summing count_PGK2 and count_NTC per SMILES. Evaluated on the 6,580 labelled Active Learning compounds (not used for training): ROC-AUC, PR-AUC, hits in the top 1% (66).
RDKit venv. Output: data/processed/gp/exp63_del_dedup.txt"""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*')
g2 = G.GetMorganGenerator(radius=2, fpSize=2048)
def cnt(smiles):
    out = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None: out[i] = g2.GetCountFingerprintAsNumPy(m)
    return np.log1p(out)
al = pd.read_csv("data/raw/active_learning_data.csv"); XA = cnt(list(al.RDKit_SMILES)); y = al.Label.values
t = pd.read_pickle("data/processed/del_sel.pkl")
def run(df, name):
    pos = df[(df.count_PGK2 >= 5) & (df.count_NTC == 0)]
    neg = df[df.count_PGK2 <= 1].sample(50000, random_state=0)
    X = cnt(list(pos.SMILES) + list(neg.SMILES)); yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(X, yy)
    s = m.decision_function(XA); top = np.argsort(-s)[:66]
    print(f"{name}: positivos {len(pos)} | negativos {len(neg)} | ROC-AUC {roc_auc_score(y, s):.3f} | PR-AUC {average_precision_score(y, s):.3f} | hits en el top 1% (66): {int(y[top].sum())} de {int(y.sum())}", flush=True)
run(t, "filas como se publicaron")
d = t.groupby("SMILES", as_index=False).agg(count_PGK2=("count_PGK2", "sum"), count_NTC=("count_NTC", "sum"))
print("SMILES unicos tras sumar conteos:", len(d))
run(d, "desduplicado por SMILES (suma de conteos)")
