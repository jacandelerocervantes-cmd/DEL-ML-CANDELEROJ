"""Pipeline unico y ejecutable: de los datos crudos a los CSV de AL1 y AL2 (Team_candeleroj).

Terminologia (fija):
  AL1 / AL2   = rondas de envio de Active Learning (NO son modelos).
  DEL         = la DATA de seleccion DEL (data/raw/PGK2_selection.parquet), no un modelo.
  Metodo      = UN solo metodo: regresion logistica L2 (C=0.05, class_weight=balanced) sobre huellas Morgan de conteos (log1p).
                Se entrena dos veces, con dos fuentes de datos distintas, y las dos senales se combinan en un puntaje:
                  senal_etiquetas : entrenada con las 6.580 etiquetas de la ronda AL (active_learning_data.csv) + negativos de fondo del pool (Morgan r3)
                  senal_DEL       : entrenada con la data DEL (PGK2_selection: positivos count_PGK2>=5 y count_NTC==0; 50.000 negativos count_PGK2<=1) (Morgan r2)
  puntaje     = z(senal_etiquetas) + W_DEL * z(senal_DEL)

Lo que se envio (reconstruido desde exp48/exp51/exp60 y verificado contra los CSV enviados con `--stage verify`):
  AL1: W_DEL = 0.1 SOLO en la columna Score. La seleccion Sel_50 se hace con senal_etiquetas (picks por modelo + miembros de series ampliadas); la senal DEL no entra a Sel_50.
  AL2: W_DEL = 0.3 en Score Y en la seleccion; excluye los 50 compuestos de AL1; 18 de grupos nuevos (Tanimoto<0.32 a AL1) + 32 de alta probabilidad (max 2 por grupo Butina).

Uso (desde la raiz del repositorio):  python pipeline/run_pipeline.py [--stage all|inputs|signals|series|al1|al2|verify|ablation] [--force]
Salidas en pipeline_out/ (no sobrescribe nada fuera de esa carpeta)."""
import argparse, hashlib, json, os, platform, sys, time
from collections import deque
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, rdFMCS
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog("rdApp.*")
OUT = os.environ.get("PIPELINE_OUT", "pipeline_out")
RAW = dict(
    del_data="data/raw/PGK2_selection.parquet",
    labels="data/raw/active_learning_data.csv",
    test="data/raw/Val-Test-set/PGK2_Test_split.csv",
    validation="data/raw/Val-Test-set/PGK2_Validation_split.csv",
)
SUBMITTED = dict(AL1="active_learning_candidates/Team_candeleroj_AL1_20261005.csv",
                 AL2="active_learning_candidates/Team_candeleroj_AL2_newgroups_20261005.csv")
CONFIG = dict(
    lr=dict(C=0.05, class_weight="balanced", max_iter=3000),
    del_data=dict(min_count_PGK2=5, max_count_NTC=0, n_negatives=50000, max_count_neg=1, neg_seed=0, morgan_radius=2, nbits=2048),
    labels=dict(morgan_radius=3, nbits=2048, crossfit_seed=48, n_background_per_half=50000),
    series=dict(min_atoms=6, cap=100, mcs_timeout=8, min_hits_for_quota=2, quota_big=3, quota_small=2, big_threshold=25, max_series_members=17),
    AL1=dict(w_del_score=0.1, w_del_selection=0.0, n=50, diversity=0.6, maxcand=4000),
    AL2=dict(w_del=0.3, n=50, top=6000, butina_sim=0.32, n_new_groups=18, cap_per_group=2),
)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
P = lambda name: os.path.join(OUT, name)
z = lambda a: (a - a.mean()) / a.std()
g2 = G.GetMorganGenerator(radius=2, fpSize=2048)
g3 = G.GetMorganGenerator(radius=3, fpSize=2048)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def smiles_col(df):
    return [c for c in df.columns if "smi" in c.lower()][0]


def feats(smis, gen):
    """log1p de huellas Morgan de conteos, en bloques de 20.000."""
    smis = list(smis); out = []
    for i in range(0, len(smis), 20000):
        ms = [Chem.MolFromSmiles(s) for s in smis[i:i + 20000]]
        out.append(np.log1p(np.array([gen.GetCountFingerprintAsNumPy(m) for m in ms], dtype=np.float32)))
    return np.vstack(out)


def LR():
    return LogisticRegression(**CONFIG["lr"])


def load_pool():
    """Pool de AL: compuestos del split Test SIN etiqueta en la ronda AL (178.052)."""
    al = pd.read_csv(RAW["labels"]); T = pd.read_csv(RAW["test"]); sc = smiles_col(T)
    U = T[~T.CatalogID.isin(set(al.CatalogID))].reset_index(drop=True).rename(columns={sc: "SMILES"})[["CatalogID", "SMILES"]]
    return al, T, U


# ---------------------------------------------------------------- stage: inputs
def stage_inputs():
    os.makedirs(OUT, exist_ok=True)
    al, T, U = load_pool()
    info = {k: dict(path=p, md5=md5(p), bytes=os.path.getsize(p)) for k, p in RAW.items()}
    info["_derived"] = dict(labelled=len(al), hits=int(al.Label.sum()), test_split=len(T), unlabelled_pool=len(U))
    assert len(U) == 178052 and len(al) == 6580 and int(al.Label.sum()) == 39, "conteos distintos de los enviados"
    json.dump(info, open(P("inputs.json"), "w"), indent=1); log("inputs OK", info["_derived"]); return info


# ---------------------------------------------------------------- stage: signals
def stage_signals():
    al, T, U = load_pool(); y = al.Label.values
    c = CONFIG["del_data"]
    # --- senal DEL: data DEL -> regresion logistica
    d = pd.read_parquet(RAW["del_data"], columns=["SMILES", "count_PGK2", "count_NTC"]); log("DEL data filas:", len(d))
    pos = d[(d.count_PGK2 >= c["min_count_PGK2"]) & (d.count_NTC <= c["max_count_NTC"])]
    neg = d[d.count_PGK2 <= c["max_count_neg"]].sample(c["n_negatives"], random_state=c["neg_seed"])
    log("DEL positivos:", len(pos), "| negativos:", len(neg))
    dm = LR().fit(feats(list(pos.SMILES) + list(neg.SMILES), g2), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    del d
    # evaluacion de la data DEL sola: ranquea las 6.580 etiquetas de AL, que NUNCA se usaron para entrenar la senal DEL
    from sklearn.metrics import roc_auc_score, average_precision_score
    dl = dm.decision_function(feats(al.RDKit_SMILES, g2)); rng_b = np.random.default_rng(0); bs = []
    for _ in range(1000):
        i = rng_b.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y): bs.append(roc_auc_score(y[i], dl[i]))
    top5 = np.argsort(-dl)[:int(.05 * len(dl))]; top1 = np.argsort(-dl)[:66]
    ev = dict(senal="DEL (solo data DEL, sin etiquetas AL)", n_etiquetas=len(y), hits=int(y.sum()), AUC=round(float(roc_auc_score(y, dl)), 4), AUC_IC95_bootstrap=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
              PR_AUC=round(float(average_precision_score(y, dl)), 4), PR_AUC_azar=round(float(y.mean()), 4), hits_en_top5pct=int(y[top5].sum()), hits_en_top66=int(y[top1].sum()), positivos_DEL=len(pos), negativos_DEL=len(neg))
    json.dump(ev, open(P("del_only_eval.json"), "w"), indent=1, ensure_ascii=False); log("DEL solo ->", ev)
    U["del_raw"] = np.concatenate([dm.decision_function(feats(U.SMILES[i:i + 20000], g2)) for i in range(0, len(U), 20000)]); log("senal DEL lista")
    # --- senal etiquetas: etiquetas de la ronda AL, validacion cruzada en dos mitades del pool
    XU = feats(U.SMILES, g3); XL = feats(al.RDKit_SMILES, g3)
    rng = np.random.default_rng(CONFIG["labels"]["crossfit_seed"]); half = rng.permutation(len(XU)); H1, H2 = half[:len(XU) // 2], half[len(XU) // 2:]
    s = np.zeros(len(XU), dtype=np.float32)
    for tr, te in ((H1, H2), (H2, H1)):
        ex = rng.choice(tr, CONFIG["labels"]["n_background_per_half"], replace=False)
        s[te] = LR().fit(np.vstack([XL, XU[ex]]), np.r_[y, np.zeros(len(ex))]).decision_function(XU[te])
    U["labels_raw"] = s
    U.to_csv(P("pool_signals.csv"), index=False)
    log("senales listas | Spearman DEL vs etiquetas:", round(float(U.del_raw.rank().corr(U.labels_raw.rank())), 3)); return U


# ---------------------------------------------------------------- stage: series (algoritmo Kruger, solo sobre los 39 hits etiquetados)
def series_of(ids, smi, libmols):
    cfg = CONFIG["series"]
    def count_matches(q):
        n = 0
        for m in libmols:
            if m.HasSubstructMatch(q):
                n += 1
                if n >= cfg["cap"]: return n
        return n
    mols = [Chem.MolFromSmiles(smi[c]) for c in ids]; fps = [g2.GetFingerprint(m) for m in mols]; n = len(ids)
    if n == 1: return [([0], "singleton")]
    D = np.array([[1 - DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps]); iu = np.triu_indices(n, 1)
    root = to_tree(linkage(D[iu], method="average")); out = []; q = deque([root])
    while q:
        node = q.popleft(); members = node.pre_order()
        if len(members) == 1: out.append((members, "singleton")); continue
        r = rdFMCS.FindMCS([mols[i] for i in members], timeout=cfg["mcs_timeout"], ringMatchesRingOnly=True, completeRingsOnly=True)
        ok = False
        if r.numAtoms >= cfg["min_atoms"]:
            ok = count_matches(Chem.MolFromSmarts(r.smartsString)) < cfg["cap"]
        if ok: out.append((members, r.smartsString))
        else: q.append(node.get_left()); q.append(node.get_right())
    return out


def stage_series():
    al, T, U = load_pool(); V = pd.read_csv(RAW["validation"]); sv, st = smiles_col(V), smiles_col(T)
    vm = [m for m in (Chem.MolFromSmiles(s) for s in V[sv]) if m is not None]
    tm = [(c, m) for c, m in ((c, Chem.MolFromSmiles(s)) for c, s in zip(T.CatalogID, T[st])) if m is not None]
    libmols = vm + [m for _, m in tm]; log("biblioteca (val+test) parseada:", len(libmols))
    smi = dict(zip(al.CatalogID, al.RDKit_SMILES)); lab = dict(zip(al.CatalogID, al.Label)); TH = list(al[al.Label == 1].CatalogID)
    S = series_of(TH, smi, libmols); log("series entre los", len(TH), "hits:", len(S))
    rows, detail = [], []
    for k, (mem, sm) in enumerate(S):
        ids = [TH[i] for i in mem]
        if sm == "singleton":
            rows.append(dict(serie=k, hits=len(mem), mcs="singleton", en_pool=np.nan, etiquetados_hits=len(mem), etiquetados_neg=np.nan, sin_etiqueta=np.nan, hits_ids=";".join(ids))); continue
        q = Chem.MolFromSmarts(sm); hit_ids = [c for c, m in tm if m.HasSubstructMatch(q)]
        nh = sum(lab.get(c) == 1 for c in hit_ids); nn = sum(lab.get(c) == 0 for c in hit_ids); un = [c for c in hit_ids if c not in lab]
        rows.append(dict(serie=k, hits=len(mem), mcs=sm, en_pool=len(hit_ids), etiquetados_hits=nh, etiquetados_neg=nn, sin_etiqueta=len(un), hits_ids=";".join(ids)))
        detail += [dict(serie=k, CatalogID=c) for c in un]
    R = pd.DataFrame(rows).sort_values("hits", ascending=False); R.to_csv(P("series_extended.csv"), index=False)
    pd.DataFrame(detail).to_csv(P("series_unlabelled_members.csv"), index=False); log("series guardadas"); return R


# ---------------------------------------------------------------- helpers de seleccion
def diversified(df, score, th, n, exclude=(), maxcand=4000):
    top = df.assign(_s=score).sort_values("_s", ascending=False).head(maxcand).reset_index(drop=True)
    fps = [g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.SMILES]; pick, pf, seen = [], [], set()
    for i, r in top.iterrows():
        if len(pick) >= n: break
        if r.CatalogID in exclude: continue
        k = (MolToInchiKey(Chem.MolFromSmiles(r.SMILES)) or "")[:14]
        if k in seen: continue
        if pf and max(DataStructs.BulkTanimotoSimilarity(fps[i], pf)) >= th: continue
        pick.append(r.CatalogID); pf.append(fps[i]); seen.add(k)
    return pick


def series_quota(R):
    c = CONFIG["series"]; multi = R[(R.mcs != "singleton") & (R.hits >= c["min_hits_for_quota"])]
    return {int(r.serie): c["quota_big"] if r.sin_etiqueta >= c["big_threshold"] else c["quota_small"] for r in multi.itertuples()}


def write_candidate(ids, blend, sel, path):
    S = (blend - blend.min()) / (blend.max() - blend.min()); m = np.array([c in sel for c in ids]); s = S.copy(); s[m] = np.maximum(s[m], 0.999)
    o = pd.DataFrame({"CatalogID": ids, "Score": np.round(s, 6), "Sel_50": m.astype(int)}).sort_values("Score", ascending=False)
    o.to_csv(path, index=False)
    chk = pd.read_csv(path); assert list(chk.columns) == ["CatalogID", "Score", "Sel_50"] and len(chk) == 178052 and chk.Sel_50.sum() == 50 and chk.Score.between(0, 1).all() and chk.CatalogID.is_unique
    return o


# ---------------------------------------------------------------- stage: AL1
def select_al1(U, R):
    c = CONFIG["AL1"]; M = pd.read_csv(P("series_unlabelled_members.csv")).merge(U[["CatalogID", "labels_raw"]], on="CatalogID")
    used, mem = set(), []
    for k, q in series_quota(R).items():
        g = M[(M.serie == k) & (~M.CatalogID.isin(used))].sort_values("labels_raw", ascending=False).head(q); mem += list(g.CatalogID); used |= set(g.CatalogID)
    mem = list(dict.fromkeys(mem))[:CONFIG["series"]["max_series_members"]]; used |= set(mem)
    mod = diversified(U, U.labels_raw.values, c["diversity"], n=c["n"] - len(mem), exclude=used, maxcand=c["maxcand"])
    pick = mem + mod; assert len(pick) == c["n"] and len(set(pick)) == c["n"]
    return pick, mem, mod


def base_A6(U, al1):
    """Senal de etiquetas tal como la uso el envio: reescalada a [0.001, 0.999] sobre el pool y con los 50 de AL1 elevados a >= 0.999 y redondeo a 6 decimales (exp48_scores_A6).
    Es un artefacto del envio original: hace que los 50 de AL1 queden arriba en la base de AL1 y de AL2, con independencia del DEL. Se replica para reproducir el Score enviado."""
    raw = U.labels_raw.values.astype(float); lo, hi = raw.min(), raw.max(); b = 0.001 + 0.998 * (raw - lo) / (hi - lo); m = U.CatalogID.isin(set(al1)).values
    b[m] = np.maximum(b[m], 0.999); return np.round(b, 6)


def stage_al1():
    U = pd.read_csv(P("pool_signals.csv")); R = pd.read_csv(P("series_extended.csv")); c = CONFIG["AL1"]
    pick, mem, mod = select_al1(U, R)
    pd.DataFrame({"CatalogID": pick, "origen": ["serie_ampliada"] * len(mem) + ["modelo"] * len(mod)}).to_csv(P("selection_AL1.csv"), index=False)
    blend = z(base_A6(U, pick)) + c["w_del_score"] * z(U.del_raw.values)
    write_candidate(list(U.CatalogID), blend, set(pick), P("Team_candeleroj_AL1.csv")); log("AL1 escrito |", len(mem), "de series +", len(mod), "por puntaje"); return pick


# ---------------------------------------------------------------- stage: AL2
def select_al2(U, al1, w, base):
    c = CONFIG["AL2"]; blend = z(base) + w * z(U.del_raw.values)
    df = pd.DataFrame({"CatalogID": U.CatalogID, "blend": blend, "smiles": U.SMILES})
    smi = dict(zip(U.CatalogID, U.SMILES)); fa = [g2.GetFingerprint(Chem.MolFromSmiles(smi[x])) for x in al1]
    top = df[~df.CatalogID.isin(al1)].sort_values("blend", ascending=False).head(c["top"]).reset_index(drop=True)
    fp = [g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.smiles]
    top["sim_al1"] = [max(DataStructs.BulkTanimotoSimilarity(f, fa)) for f in fp]
    n = len(fp); d = []
    for i in range(1, n): d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(fp[i], fp[:i])]
    cl = np.zeros(n, int)
    for k, cc in enumerate(Butina.ClusterData(d, n, 1 - c["butina_sim"], isDistData=True)): cl[list(cc)] = k
    top["g"] = cl
    new = top[top.sim_al1 < c["butina_sim"]]; newpick = list(new.groupby("g").head(1).sort_values("blend", ascending=False).head(c["n_new_groups"]).CatalogID)
    rest = top[~top.CatalogID.isin(newpick)]; cap, hp, seen = {}, [], set()
    for r in rest.itertuples():
        if len(hp) >= c["n"] - len(newpick): break
        if cap.get(r.g, 0) >= c["cap_per_group"]: continue
        k = (MolToInchiKey(Chem.MolFromSmiles(r.smiles)) or "")[:14]
        if k in seen: continue
        cap[r.g] = cap.get(r.g, 0) + 1; hp.append(r.CatalogID); seen.add(k)
    sel = set(newpick + hp); assert len(sel) == c["n"] and not (sel & set(al1)), "AL2 debe excluir todo AL1"
    return newpick, hp, blend


def stage_al2():
    U = pd.read_csv(P("pool_signals.csv")); al1 = list(pd.read_csv(P("selection_AL1.csv")).CatalogID)
    newpick, hp, blend = select_al2(U, al1, CONFIG["AL2"]["w_del"], base_A6(U, al1))
    pd.DataFrame({"CatalogID": newpick + hp, "origen": ["grupo_nuevo"] * len(newpick) + ["alta_prob"] * len(hp)}).to_csv(P("selection_AL2.csv"), index=False)
    write_candidate(list(U.CatalogID), blend, set(newpick + hp), P("Team_candeleroj_AL2.csv")); log("AL2 escrito |", len(newpick), "grupos nuevos +", len(hp), "alta prob"); return newpick + hp


# ---------------------------------------------------------------- stage: ablation (quitar la senal DEL)
def stage_ablation():
    U = pd.read_csv(P("pool_signals.csv")); R = pd.read_csv(P("series_extended.csv")); al1 = list(pd.read_csv(P("selection_AL1.csv")).CatalogID); al2 = list(pd.read_csv(P("selection_AL2.csv")).CatalogID)
    n0, h0, _ = select_al2(U, al1, 0.0, base_A6(U, al1)); out = dict(
        AL1_seleccion_cambia_sin_DEL=0,
        AL1_nota="Sel_50 de AL1 no usa la senal DEL (solo senal_etiquetas + series); la senal DEL solo afecta la columna Score (W=0.1). Por construccion quitar DEL no cambia la seleccion.",
        AL2_seleccion_cambia_sin_DEL=50 - len(set(n0 + h0) & set(al2)))
    json.dump(out, open(P("ablation.json"), "w"), indent=1, ensure_ascii=False); log("ablacion:", out); return out


# ---------------------------------------------------------------- stage: verify contra los CSV enviados
def stage_verify():
    rep = {}
    for tag, sub in SUBMITTED.items():
        a = pd.read_csv(P(f"Team_candeleroj_{tag}.csv")); b = pd.read_csv(sub)
        sa, sb = set(a[a.Sel_50 == 1].CatalogID), set(b[b.Sel_50 == 1].CatalogID); m = a.merge(b, on="CatalogID", suffixes=("_nuevo", "_enviado"))
        rep[tag] = dict(sel_comunes=len(sa & sb), sel_solo_nuevo=sorted(sa - sb), sel_solo_enviado=sorted(sb - sa), mismas_filas=len(a) == len(b) and set(a.CatalogID) == set(b.CatalogID),
                        max_dif_score=float((m.Score_nuevo - m.Score_enviado).abs().max()), md5_nuevo=md5(P(f"Team_candeleroj_{tag}.csv")), md5_enviado=md5(sub))
        rep[tag]["IDENTICO_EN_SELECCION"] = rep[tag]["sel_comunes"] == 50
    s1 = set(pd.read_csv(P("Team_candeleroj_AL1.csv")).query("Sel_50==1").CatalogID); s2 = set(pd.read_csv(P("Team_candeleroj_AL2.csv")).query("Sel_50==1").CatalogID)
    rep["AL1_interseccion_AL2"] = len(s1 & s2)
    rep["entorno"] = dict(python=sys.version.split()[0], platform=platform.platform(), **{m: __import__(m).__version__ for m in ("rdkit", "sklearn", "numpy", "pandas", "scipy", "pyarrow")})
    rep["config"] = CONFIG; rep["inputs"] = json.load(open(P("inputs.json")))
    json.dump(rep, open(P("verification.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("config", "inputs", "entorno")}, indent=1, ensure_ascii=False)); return rep


# ---------------------------------------------------------------- stage: delml (diseno DEL-first con pesos actualizados; ablacion sustancial)
CONFIG["DELML"] = dict(pool_pct=50, w_del=0.6, diversity=0.6, cand=6000, family_tani=0.45, historic_hits_max=1,
                       files=dict(AL1="active_learning_candidates/Team_candeleroj_AL1_DELML_20261022.csv", AL2="active_learning_candidates/Team_candeleroj_AL2_DELML_20261022.csv"))


def stage_delml():
    """Mismo metodo y mismas senales, pesos actualizados: la senal DEL (data DEL sin promiscuos, historic_hits<=1) define el 50 % superior del pool y pesa 0.6 en el puntaje.
    AL2 excluye toda la familia quimica de AL1 (mismo Murcko, mismo esqueleto generico o Tanimoto >= 0.45 a algun elegido de AL1). Ablacion = misma regla sin la senal DEL."""
    from rdkit.Chem.Scaffolds import MurckoScaffold as MS
    C = CONFIG["DELML"]; c = CONFIG["del_data"]; U = pd.read_csv(P("pool_signals.csv"))
    d = pd.read_parquet(RAW["del_data"], columns=["SMILES", "count_PGK2", "count_NTC", "historic_hits"])
    pos = d[(d.count_PGK2 >= c["min_count_PGK2"]) & (d.count_NTC <= c["max_count_NTC"]) & (d.historic_hits <= C["historic_hits_max"])]
    neg = d[d.count_PGK2 <= c["max_count_neg"]].sample(c["n_negatives"], random_state=c["neg_seed"]); log("DELML positivos:", len(pos)); del d
    dm = LR().fit(feats(list(pos.SMILES) + list(neg.SMILES), g2), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    U["del2"] = np.concatenate([dm.decision_function(feats(U.SMILES[i:i + 20000], g2)) for i in range(0, len(U), 20000)])
    U["zdel"] = z(U.del2); U["zal"] = z(U.labels_raw); U["blend"] = C["w_del"] * U.zdel + U.zal
    mol, fp = {}, {}
    M = lambda cid, s: mol.setdefault(cid, Chem.MolFromSmiles(s)); F = lambda cid, s: fp.setdefault(cid, g2.GetFingerprint(M(cid, s)))
    def pick(df, score, exclude=(), n=50):
        o, f_ = [], []
        for r in df[~df.CatalogID.isin(exclude)].assign(_s=score).sort_values("_s", ascending=False).head(C["cand"]).itertuples():
            f = F(r.CatalogID, r.SMILES)
            if f_ and max(DataStructs.BulkTanimotoSimilarity(f, f_)) >= C["diversity"]: continue
            o.append(r.CatalogID); f_.append(f)
            if len(o) >= n: break
        return o
    def scaf(cid, s, generic=False):
        try:
            sc = MS.GetScaffoldForMol(M(cid, s))
            if generic: sc = MS.MakeScaffoldGeneric(sc)
            return Chem.MolToSmiles(sc)
        except Exception: return ""
    pool = U[U.del2.rank(ascending=False) <= C["pool_pct"] / 100 * len(U)].copy()
    a1 = pick(pool, pool.blend); A1 = pool[pool.CatalogID.isin(a1)]
    fs = {scaf(r.CatalogID, r.SMILES) for r in A1.itertuples()} - {""}; fg = {scaf(r.CatalogID, r.SMILES, True) for r in A1.itertuples()} - {""}; f1 = [F(r.CatalogID, r.SMILES) for r in A1.itertuples()]
    same_family = lambda r: scaf(r.CatalogID, r.SMILES) in fs or scaf(r.CatalogID, r.SMILES, True) in fg or max(DataStructs.BulkTanimotoSimilarity(F(r.CatalogID, r.SMILES), f1)) >= C["family_tani"]
    cand = pool.sort_values("blend", ascending=False).head(C["cand"] * 3); sa1 = set(a1)
    fam = {r.CatalogID for r in cand.itertuples() if r.CatalogID in sa1 or same_family(r)}
    a2 = pick(pool, pool.blend, exclude=fam)
    assert len(a1) == 50 and len(a2) == 50 and not set(a1) & set(a2) and not [r for r in pool[pool.CatalogID.isin(a2)].itertuples() if same_family(r)], "AL2 comparte familia con AL1"
    S = (U.blend - U.blend.min()) / (U.blend.max() - U.blend.min()); rep = {}
    for tag, sel in (("AL1", a1), ("AL2", a2)):
        m = U.CatalogID.isin(set(sel)).values; s = np.where(m, np.maximum(S, 0.999), S * 0.998)
        o = pd.DataFrame({"CatalogID": U.CatalogID, "Score": np.round(s, 6), "Sel_50": m.astype(int)}).sort_values("Score", ascending=False); o.to_csv(P(f"Team_candeleroj_{tag}_DELML.csv"), index=False)
        ref = pd.read_csv(C["files"][tag]); rs = set(ref[ref.Sel_50 == 1].CatalogID); rep[f"{tag}_sel_comunes_con_archivo_22oct"] = len(set(sel) & rs)
    r1 = pick(U, U.zal); r2 = pick(U, U.zal, exclude=set(fam) | sa1)
    rep.update(AL1_cambia_sin_DEL=50 - len(sa1 & set(r1)), AL2_cambia_sin_DEL=50 - len(set(a2) & set(r2)), AL1_interseccion_AL2=len(sa1 & set(a2)), pool_DEL=len(pool), config={k: v for k, v in C.items() if k != "files"},
               percentil_DEL_mediano=dict(AL1=float(U.del2.rank(pct=True)[U.CatalogID.isin(sa1)].median()), AL2=float(U.del2.rank(pct=True)[U.CatalogID.isin(set(a2))].median())))
    json.dump(rep, open(P("ablation_delml.json"), "w"), indent=1, ensure_ascii=False); log("DELML:", rep); return rep


# ---------------------------------------------------------------- stage: library (el embudo ejecutable: 428,960 -> pool de 178,052 -> 100)
def stage_library():
    """Ordena TODA la biblioteca (los dos archivos de estructuras: 428,960 compuestos) con el componente DEL y situa en ese ranking los 100 compuestos elegidos de AL1 y AL2."""
    V = pd.read_csv(RAW["validation"]); T = pd.read_csv(RAW["test"])
    lib = pd.concat([V.rename(columns={smiles_col(V): "SMILES"})[["CatalogID", "SMILES"]], T.rename(columns={smiles_col(T): "SMILES"})[["CatalogID", "SMILES"]]]).drop_duplicates("CatalogID").reset_index(drop=True)
    c = CONFIG["del_data"]
    d = pd.read_parquet(RAW["del_data"], columns=["SMILES", "count_PGK2", "count_NTC"])
    pos = d[(d.count_PGK2 >= c["min_count_PGK2"]) & (d.count_NTC <= c["max_count_NTC"])]
    neg = d[d.count_PGK2 <= c["max_count_neg"]].sample(c["n_negatives"], random_state=c["neg_seed"]); del d
    dm = LR().fit(feats(list(pos.SMILES) + list(neg.SMILES), g2), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    lib["del_raw"] = np.concatenate([dm.decision_function(feats(lib.SMILES[i:i + 20000], g2)) for i in range(0, len(lib), 20000)]); log("biblioteca puntuada:", len(lib))
    lib["rank"] = lib.del_raw.rank(ascending=False, method="min").astype(int); lib["pct"] = lib.del_raw.rank(pct=True)
    lib[["CatalogID", "del_raw", "rank", "pct"]].to_csv(P("library_del_scores.csv"), index=False)
    rk = dict(zip(lib.CatalogID, lib["rank"])); pc = dict(zip(lib.CatalogID, lib.pct)); N = len(lib)
    pool = pd.read_csv(P("pool_signals.csv")); pr = (-pool.del_raw).rank(method="min").astype(int); rp_ = dict(zip(pool.CatalogID, pr)); NP = len(pool)
    cuts = [100, 1000, 5000, 10000, 25000, 50000, 100000, N // 2]; cutp = [100, 1000, 5000, 10000, 25000, 50000, NP // 2]
    out = dict(embudo=dict(biblioteca=N, archivo_estructuras_1=len(V), archivo_estructuras_2=len(T), pool_a_puntuar=NP, elegidos=100))
    for tag in ("AL1", "AL2"):
        ids = list(pd.read_csv(P(f"selection_{tag}.csv")).CatalogID); r = np.array([rk[x] for x in ids]); q = np.array([rp_[x] for x in ids]); p = np.array([pc[x] for x in ids])
        out[tag] = dict(biblioteca=dict(min=int(r.min()), mediana=int(np.median(r)), max=int(r.max()), percentil_mediano=round(float(np.median(p)), 3), **{f"dentro_de_{c_}": int((r <= c_).sum()) for c_ in cuts}),
                        pool=dict(min=int(q.min()), mediana=int(np.median(q)), max=int(q.max()), **{f"dentro_de_{c_}": int((q <= c_).sum()) for c_ in cutp}))
    json.dump(out, open(P("library_positions.json"), "w"), indent=1); log("biblioteca ordenada:", json.dumps(out)); return out


def stage_lineage():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lineage; return lineage.build()


STAGES = dict(inputs=stage_inputs, signals=stage_signals, series=stage_series, al1=stage_al1, al2=stage_al2, ablation=stage_ablation, library=stage_library, verify=stage_verify, delml=stage_delml, lineage=stage_lineage)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all"] + list(STAGES)); a = ap.parse_args()
    CORE = ["inputs", "signals", "series", "al1", "al2", "ablation", "library", "verify"]   # 'delml' y 'lineage' son opcionales: necesitan archivos que no van en el repositorio
    for name in (CORE if a.stage == "all" else [a.stage]):
        log("== etapa:", name); STAGES[name]()
