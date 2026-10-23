"""Step 0 of the reproduction: checks the raw files and builds data/processed/del_sel.pkl (a pickle copy of PGK2_selection.parquet used by the DEL scripts).
Run from the repository root:  python scripts/prepare_data.py
Raw files must be placed as described in REPRODUCE.md. A checksum mismatch is reported but does not stop the script."""
import hashlib, os, sys
import pandas as pd
EXPECTED = {
    "data/raw/PGK2_selection.parquet": "f1832ca598e647eb66ccd1f38a9d7ea5",
    "data/raw/Val-Test-set/PGK2_Test_split.csv": "43d5f11f644893d77704d0c9aa38f5c8",
    "data/raw/Val-Test-set/PGK2_Validation_split.csv": "348f76e1c4603358a3ac572d12c35020",
    "data/raw/active_learning_data.csv": "470533959ac0427504e71042231cbf9b",
    "data/raw/active_learning_submission_example.csv": "85b92f931a1828df74fa39f2c6952907",
}
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
missing = False
for p, want in EXPECTED.items():
    if not os.path.exists(p): print("FALTA   ", p); missing = True; continue
    got = md5(p); print("OK      " if got == want else "DISTINTO ", p, got)
if missing: sys.exit("Faltan archivos crudos; ver REPRODUCE.md")
os.makedirs("data/processed/gp", exist_ok=True); os.makedirs("submissions/active_learning_candidates", exist_ok=True)
t = pd.read_parquet("data/raw/PGK2_selection.parquet")
print("filas:", len(t), "| SMILES unicos:", t["SMILES"].nunique(), "| columnas:", list(t.columns))
t.to_pickle("data/processed/del_sel.pkl"); print("escrito data/processed/del_sel.pkl")
