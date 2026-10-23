"""Linaje del pipeline unico: de los primeros envios con data DEL hasta AL1/AL2, como una sola cadena de refinamientos.

No ejecuta los scripts historicos; los CLASIFICA por lo que su codigo lee (hechos del codigo, no nombres supuestos) y une cada tanda de envios con su resultado real en Synapse.
Etapa de un script = la mas tardia que toca, por precedencia:
  S4_etiquetas_AL     lee active_learning_data.csv (etiquetas de la ronda AL, publicadas al final del Blind Test)
  S3_resultados_envios lee resultados de envios anteriores (confirmed_submission_outcomes / submissions/ / Synapse)
  S2_DEL_puntaje      lee la data DEL (PGK2_selection / del_sel) o datos DEL externos y puntua o entrena con ella
  S1_otros            el resto (utilidades, docking, modulos)
Un script puede leer DEL en cualquier etapa: la columna reads_DEL_data lo dice por separado, porque DEL atraviesa todo el linaje.
Entradas: scripts/*.py, earlier_workflow/*.py, pipeline_in/synapse_results.csv, submissions/*.txt. Salidas en pipeline_out/lineage_*.
Uso: python pipeline/lineage.py   (o  python pipeline/run_pipeline.py --stage lineage)"""
import glob, json, os, re
import numpy as np, pandas as pd

OUT = "pipeline_out"
PAT = dict(
    reads_DEL_data=r"PGK2_selection|del_sel\.pkl|PGK2_selection_labeled|PGK2_denoised|step1_union|step2_scores",
    reads_AL_labels=r"active_learning_data",
    reads_outcomes=r"confirmed_submission_outcomes|confirmed_hits|submissions/Team_candeleroj|syn75352369",
    ext_NTC_supplement=r"PGK2_NTC_supplement",
    ext_OpenDEL_buildingblocks=r"OpenDEL_libraries",
    ext_WDR91_TrainSet=r"DREAM_Challenge_1_TrainSet",
    ext_CACHE7_docking=r"CACHE7_PDB|data/docking",
    ext_ChEMBL=r"chembl_repurposing",
    ext_pretrained=r"models/pretrained|delbert_wdr91|model_mpnn_cnn_davis|pgk2_xgboost_asl",
)
FINAL = {"exp42_kruger_series", "exp43_extended_series", "exp48_final_recipe", "exp51_final_AL_csv", "exp60_AL2_new_groups"}
# Clasificacion de cada envio de Blind Validation segun el catalogo del propio proyecto (comentarios por ronda en scripts/build_confirmed_outcomes.py y scripts de decodificacion).
#   identificacion = localiza hits ya medidos (biseccion, pools binarios, aislar un compuesto/familia con relleno de ceros conocidos) o RECOMBINA bloques de hits ya identificados para maximizar clusters. No mide prediccion.
#   hipotesis      = prueba una hipotesis quimica sobre sospechosos o una pareja pareada (72, 74, 78, 87-91); mixta.
#   prediccion     = exploracion con un ranking o receta sobre compuestos no medidos (todo lo demas).
# Fuente: rondas 26-100 = comentarios de build_confirmed_outcomes.py; 101-112 = exp6_adaptive_r99.py; 116-152 = exp27_decode.py / exp31_adaptive_groups.py. Rondas 1-25: propositos no registrados al enviarse.
IDENTIFICACION = (set(range(26, 32)) | {41} | set(range(42, 48)) | {49, 50, 52, 73, 75, 76, 77, 79} | set(range(80, 87)) | set(range(92, 101)) | set(range(101, 113)) | set(range(116, 153)))
HIPOTESIS = {72, 74, 78, 87, 88, 89, 90, 91}
BINS = [(1, 10), (11, 23), (24, 54), (55, 100), (101, 130), (131, 155), (156, 199)]


def classify(row):
    if row.reads_AL_labels: return "S4_etiquetas_AL"
    if row.reads_outcomes: return "S3_resultados_envios"
    if row.reads_DEL_data or any(row[c] for c in row.index if c.startswith("ext_")): return "S2_DEL_puntaje"
    return "S1_otros"


def scripts_table():
    rows = []
    for f in sorted(glob.glob("scripts/*.py") + glob.glob("earlier_workflow/*.py")):
        s = open(f, encoding="utf8", errors="ignore").read(); name = os.path.basename(f)[:-3]
        m = re.search(r'"""(.*?)"""', s, re.S); doc = " ".join((m.group(1) if m else "").split())[:160]
        r = dict(script=name, carpeta=os.path.dirname(f), final_AL1_AL2=name in FINAL, doc=doc)
        r.update({k: bool(re.search(p, s)) for k, p in PAT.items()}); rows.append(r)
    t = pd.DataFrame(rows).drop_duplicates("script"); t["etapa"] = t.apply(classify, axis=1); return t


def submissions_table():
    d = pd.read_csv("pipeline_in/synapse_results.csv"); d["num"] = d.name.str.extract(r"_(\d+)\.txt")[0]
    t = d[d.num.notna()].copy(); t["num"] = t.num.astype(int)
    share = json.load(open(os.path.join(OUT, "del_share_sent.json"))) if os.path.exists(os.path.join(OUT, "del_share_sent.json")) else {}
    t["proposito"] = np.where(t.num.isin(IDENTIFICACION), "identificacion", np.where(t.num.isin(HIPOTESIS), "hipotesis", np.where(t.num >= 180, "sin_clasificar", "prediccion"))); t[["num", "proposito", "N_hits", "N_clusters"]].to_csv(os.path.join(OUT, "lineage_submission_purpose.csv"), index=False)
    rows = []
    for a, b in BINS:
        g = t[(t.num >= a) & (t.num <= b) & (t.proposito == "prediccion")]   # solo envios de prediccion: los de identificacion localizan hits ya conocidos
        sh = share.get(f"{a}-{b}", share.get("156-179") if (a, b) == (156, 199) else {})
        rows.append(dict(tanda=f"{a}-{b}", envios=len(g), con_hits=int((g.N_hits > 0).sum()), hits_max=int(g.N_hits.max()) if len(g) else 0, hits_media=round(float(g.N_hits.mean()), 1) if len(g) else 0.0,
                         mediana_percentil_DEL=sh.get("mediana_pct"), pct_en_top10_DEL=sh.get("top10")))
    ra = d[d.name.str.contains("AL1|AL2")][["name", "N_hits", "N_clusters", "PR_AUC", "ROC_AUC", "rank"]]
    for r in ra.itertuples():
        k = "AL1" if "AL1" in r.name else "AL2"
        rows.append(dict(tanda=k, envios=1, con_hits=int(r.N_hits > 0), hits_max=int(r.N_hits), hits_media=float(r.N_hits), mediana_percentil_DEL=share.get(k, {}).get("mediana_pct"), pct_en_top10_DEL=share.get(k, {}).get("top10"),
                         N_clusters=r.N_clusters, PR_AUC=r.PR_AUC, ROC_AUC=r.ROC_AUC, rank=r.rank))
    return pd.DataFrame(rows)


def build():
    os.makedirs(OUT, exist_ok=True); S = scripts_table(); S.to_csv(os.path.join(OUT, "lineage_scripts.csv"), index=False)
    ext = [c for c in S.columns if c.startswith("ext_")]
    resumen = S.groupby("etapa").agg(scripts=("script", "count"), leen_data_DEL=("reads_DEL_data", "sum"), leen_etiquetas_AL=("reads_AL_labels", "sum"), leen_resultados=("reads_outcomes", "sum")).reset_index()
    E = pd.DataFrame({c: [int(S[c].sum())] for c in ext}).T.rename(columns={0: "scripts_que_lo_leen"})
    sub = submissions_table(); sub.to_csv(os.path.join(OUT, "lineage_submissions.csv"), index=False)
    out = dict(
        scripts_total=len(S), scripts_que_leen_data_DEL=int(S.reads_DEL_data.sum()),
        scripts_que_leen_data_DEL_o_externo_DEL=int((S.reads_DEL_data | S[ext].any(axis=1)).sum()),
        por_etapa=resumen.to_dict("records"), datos_externos=E.scripts_que_lo_leen.to_dict(),
        scripts_finales_AL1_AL2=sorted(FINAL), nota="Clasificacion por lectura de datos en el codigo; no prueba que el script haya generado un envio concreto.")
    json.dump(out, open(os.path.join(OUT, "lineage_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(resumen.to_string(index=False)); print(E.to_string()); print(sub.to_string(index=False)); print("scripts:", len(S), "| leen data DEL:", out["scripts_que_leen_data_DEL"]); return out


if __name__ == "__main__":
    build()
