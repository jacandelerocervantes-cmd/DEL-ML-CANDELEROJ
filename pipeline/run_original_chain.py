"""Ejecuta la cadena ORIGINAL de scripts que genero AL1 y AL2, en una carpeta aislada (pipeline_out/original_chain/), sin tocar data/processed ni los envios.
Cadena (dependencias a nivel de archivo, ver pipeline/dag.py):
  datos crudos -> del_sel.pkl -> exp10_prefilter -> exp16_pool_scoring -> exp43_extended_series (usa exp42) -> exp48_final_recipe -> exp51_final_AL_csv (AL1) -> exp60_AL2_new_groups (AL2)
Despues compara AL1/AL2 resultantes con (a) los CSV enviados y (b) los del pipeline unico (pipeline_out/Team_candeleroj_AL*.csv).
Uso: python pipeline/run_original_chain.py [--force]      Salida: pipeline_out/original_chain/ y pipeline_out/original_chain_check.json"""
import json, os, shutil, subprocess, sys, time
import pandas as pd

SB = os.path.abspath(os.environ.get("CHAIN_DIR", "pipeline_out/original_chain")); ROOT = os.path.abspath("."); UNI = os.environ.get("PIPELINE_OUT", "pipeline_out")
RAW = ["data/raw/PGK2_selection.parquet", "data/raw/active_learning_data.csv", "data/raw/Val-Test-set/PGK2_Test_split.csv", "data/raw/Val-Test-set/PGK2_Validation_split.csv"]
STEPS = ["exp10_prefilter", "exp16_pool_scoring", "exp43_extended_series", "exp48_final_recipe", "exp51_final_AL_csv", "exp60_AL2_new_groups"]
NEED = ["exp42_kruger_series"]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def setup(force):
    if force and os.path.exists(SB): shutil.rmtree(SB)
    for d in ("data/raw/Val-Test-set", "data/processed/gp", "scripts", "submissions/active_learning_candidates", "logs"): os.makedirs(os.path.join(SB, d), exist_ok=True)
    for f in RAW:
        dst = os.path.join(SB, f)
        if not os.path.exists(dst): shutil.copy2(f, dst)
    for s in STEPS + NEED: shutil.copy2(f"scripts/{s}.py", os.path.join(SB, "scripts", f"{s}.py"))
    tmpl = os.path.join(SB, "data/raw/active_learning_submission_example.csv")
    if not os.path.exists(tmpl):   # plantilla de la organizacion no disponible: mismos IDs que el pool sin etiqueta del split Test
        T = pd.read_csv(RAW[2]); al = pd.read_csv(RAW[1]); U = T[~T.CatalogID.isin(set(al.CatalogID))]
        pd.DataFrame({"CatalogID": U.CatalogID, "Score": 0.0, "Sel_50": 0}).to_csv(tmpl, index=False)
    pk = os.path.join(SB, "data/processed/del_sel.pkl")   # mismas dos lineas finales de scripts/00_prepare_data.py
    if not os.path.exists(pk): pd.read_parquet(RAW[0]).to_pickle(pk)


def run_step(name):
    mark = os.path.join(SB, "logs", f"{name}.done")
    if os.path.exists(mark): log("ya hecho:", name); return
    log("== ejecutando", name); env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(os.path.join(SB, "logs", f"{name}.log"), "w", encoding="utf8") as lg:
        r = subprocess.run([sys.executable, os.path.join("scripts", f"{name}.py")], cwd=SB, env=env, stdout=lg, stderr=subprocess.STDOUT)
    if r.returncode != 0: sys.exit(f"{name} fallo (codigo {r.returncode}); ver {SB}/logs/{name}.log")
    open(mark, "w").write("ok"); log("listo:", name)


def compare():
    out = {}
    pairs = dict(AL1=("Team_candeleroj_AL1_20261005.csv", "active_learning_candidates/Team_candeleroj_AL1_20261005.csv", UNI + "/Team_candeleroj_AL1.csv"),
                 AL2=("Team_candeleroj_AL2_newgroups_20261005.csv", "active_learning_candidates/Team_candeleroj_AL2_newgroups_20261005.csv", UNI + "/Team_candeleroj_AL2.csv"))
    for tag, (fn, sub, uni) in pairs.items():
        a = pd.read_csv(os.path.join(SB, "submissions/active_learning_candidates", fn)); s = lambda d: set(d[d.Sel_50 == 1].CatalogID)
        r = {}
        for nm, p in (("enviado", sub), ("pipeline_unico", uni)):
            b = pd.read_csv(p); m = a.merge(b, on="CatalogID", suffixes=("_cadena", "_otro"))
            r[nm] = dict(sel_comunes=len(s(a) & s(b)), solo_cadena=sorted(s(a) - s(b)), solo_otro=sorted(s(b) - s(a)), max_dif_score=float((m.Score_cadena - m.Score_otro).abs().max()))
        out[tag] = r
    json.dump(out, open(os.path.join(UNI, "original_chain_check.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    setup("--force" in sys.argv); log("sandbox listo en", SB)
    for st in STEPS: run_step(st)
    compare()
