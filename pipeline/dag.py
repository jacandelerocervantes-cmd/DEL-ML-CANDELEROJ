"""Grafo de dependencias entre scripts, por analisis estatico (solo lectura): que archivo escribe cada script y quien lo lee.
Heuristica: un literal de ruta (data/, results/, submissions/, active_learning_candidates/, reference/, models/) en una linea con una funcion de escritura
(to_csv, to_parquet, np.save*, json.dump, to_pickle, pickle.dump, open(..,'w')) es ESCRITURA; en cualquier otro caso es LECTURA. Las constantes NOMBRE = "ruta" se resuelven.
Las rutas con {..} o * se tratan como patrones. Un script que hace exec() de otro (exp43 -> exp42) hereda sus lecturas.
Salida: pipeline_out/dag.json  (aristas productor -> consumidor y ancestros de cada script final).   Uso: python pipeline/dag.py"""
import fnmatch, glob, json, os, re
from collections import defaultdict

OUT = "pipeline_out"
ROOTS = r"(?:data|results|submissions|active_learning_candidates|reference|models)"
LIT = re.compile(r"""(?:f?)["'](""" + ROOTS + r"""/[^"']+)["']""")
CONST = re.compile(r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*f?["'](""" + ROOTS + r"""/[^"']+)["']""")
WRITE = re.compile(r"\.to_csv\(|\.to_parquet\(|np\.save|\.to_pickle\(|pickle\.dump|json\.dump|savez|\.write\(|open\([^)]*[\"']w[b]?[\"']|savefig")
FINAL = ["exp51_final_AL_csv", "exp60_AL2_new_groups"]


def pat(p):
    return re.sub(r"\{[^}]*\}", "*", p)


def analyse():
    scripts = {os.path.basename(f)[:-3]: f for f in sorted(glob.glob("scripts/*.py") + glob.glob("earlier_workflow/*.py"))}
    info = {}
    for name, f in scripts.items():
        src = open(f, encoding="utf8", errors="ignore").read(); lines = src.splitlines(); consts = {}
        for ln in lines:
            m = CONST.match(ln)
            if m: consts[m.group(1)] = m.group(2)
        reads, writes = set(), set()
        for ln in lines:
            if ln.lstrip().startswith("#"): continue
            lits = [pat(x) for x in LIT.findall(ln)]
            for k, v in consts.items():
                if re.search(r"\b" + re.escape(k) + r"\b", ln) and not CONST.match(ln): lits.append(pat(v))
            if not lits: continue
            (writes if WRITE.search(ln) else reads).update(lits)
        # exec() de otro script: hereda sus lecturas (exp43 ejecuta la parte de funciones de exp42)
        for other in re.findall(r"scripts/([A-Za-z0-9_]+)\.py", src):
            if other in scripts and other != name and "exec(" in src: info.setdefault("_exec", {}).setdefault(name, set()).add(other)
        info[name] = dict(file=f, reads=reads, writes=writes)
    ex = info.pop("_exec", {})
    for a, bs in ex.items():
        for b in bs: info[a]["reads"] |= info[b]["reads"]
    return info


def match(w, r):
    """Arista solo si la lectura es una ruta concreta (sin comodin) que coincide con lo que escribe el productor (que puede ser un patron de f-string).
    Las listas de envio (submissions/) son DATOS de entrada, no productos del grafo."""
    if r.startswith("submissions/") or w.startswith("submissions/") or "*" in r or r.endswith("/"): return False
    return w == r or fnmatch.fnmatch(r, w)


def build():
    info = analyse(); prod = defaultdict(set); edges = []
    for a, d in info.items():
        for w in d["writes"]: prod[w].add(a)
    for b, d in info.items():
        for r in d["reads"]:
            for w, ps in prod.items():
                if match(w, r):
                    for a in ps:
                        if a != b: edges.append((a, b, r))
    parents = defaultdict(set)
    for a, b, r in edges: parents[b].add((a, r))
    def anc(n, seen=None):
        seen = seen if seen is not None else set()
        for a, _ in parents[n]:
            if a not in seen: seen.add(a); anc(a, seen)
        return seen
    # lecturas sin productor = datos de entrada (crudos o precalculados)
    produced = set(prod)
    def inputs_of(names):
        s = set()
        for n in names:
            for r in info[n]["reads"]:
                if not any(match(w, r) for w in produced): s.add(r)
        return s
    res = {}
    for f in FINAL:
        A = anc(f); res[f] = dict(ancestros=sorted(A), entradas_sin_productor=sorted(inputs_of(A | {f})))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(aristas=sorted(set(edges)), finales=res, scripts=len(info)), open(os.path.join(OUT, "dag.json"), "w"), indent=1, ensure_ascii=False)
    for f, d in res.items():
        print(f, "| ancestros:", d["ancestros"]); print("   entradas sin productor:", d["entradas_sin_productor"])
    return res


if __name__ == "__main__":
    build()
