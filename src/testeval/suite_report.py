"""Explicação por SUÍTE (por modelo gerador) e comparação entre modelos.

Organiza os resultados em pastas para o relatório:

  work/<run>/suites/<gerador>/
      testes/<unit_id>.py      cópia dos testes gerados (uma suíte por alvo × amostra)
      metricas_unidades.csv    métricas de cada suíte (alvo × amostra)
      metricas_testes.csv      métricas de cada teste dessa suíte (colunas do results.csv)
      explicacao.md / .json    explicação da suíte, com evidências verificáveis
  work/<run>/comparacao/
      comparacao_modelos.csv   uma linha por gerador (média ± desvio entre suítes)
      mutantes_por_alvo.csv    mutante × gerador: fração das amostras em que é morto
      relatorio_comparativo.md comparação explicada entre os modelos

Princípios (os mesmos de explain.py):
  - toda afirmação aponta para fatos medidos (alvo, mutante, linha, teste);
  - ``verify_suite_explanation`` confere cada referência contra os fatos;
  - métricas BRUTAS (sem normalização relativa ao melhor), para que a nota de um
    modelo não dependa dos outros;
  - o que vem dos juízes LLM (relevância, oráculo) é rotulado como tal.
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

UNIT_COLS = ["unit_id", "target", "n_tests", "n_validos", "n_falha_assercao", "n_redundantes",
             "cobertura_suite", "mutation_score_suite", "n_suite_minima", "mutantes",
             "mutantes_timeout", "relevancia_mediana", "pct_oraculo_inconsistente",
             "score_medio", "aceitar", "revisar", "recusar", "error"]

# métrica -> (rótulo, maior é melhor?)
COMPARE = {
    "pct_executaveis": ("Executáveis", True),
    "pct_falha_assercao": ("Falham no código original (oráculo errado)", False),
    "cobertura_suite": ("Cobertura da suíte", True),
    "mutation_score_suite": ("Mutation score da suíte", True),
    "pct_redundantes": ("Testes redundantes", False),
    "relevancia_mediana": ("Relevância (juízes LLM)", True),
    "score_medio": ("Score médio por teste", True),
}


# ---------------------------------------------------------------- utilidades
def _mean_sd(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return round(statistics.mean(vals), 4), round(sd, 4)


def _fmt(m: Optional[float], sd: Optional[float] = None, pct: bool = True) -> str:
    if m is None:
        return "—"
    s = f"{m:.0%}" if pct else f"{m:.2f}"
    if sd:
        s += f" ± {sd:.0%}" if pct else f" ± {sd:.2f}"
    return s


def _mut_label(muts: Dict[int, Dict], mid: int) -> str:
    m = muts.get(mid)
    return f"#{mid} (L{m['line']}: {m['desc']})" if m else f"#{mid}"


def _write_csv(path: Path, data: List[Dict], cols: Optional[List[str]] = None) -> None:
    cols = cols or (list(data[0]) if data else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in data:
            w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in cols})


# ---------------------------------------------------------------- fatos por unidade
def unit_metrics(unit: Dict, rows: List[Dict]) -> Dict:
    """Métricas de UMA suíte (alvo × gerador × amostra)."""
    s = unit["suite"]
    ex = [r for r in rows if r.get("executavel")]
    rel = [r["relevancia_juiz"] for r in ex if r.get("relevancia_juiz") is not None]
    orc = [r for r in ex if r.get("oraculo_juiz")]
    dec = Counter(r.get("decisao_sugerida") for r in rows)
    return {
        "unit_id": unit["unit_id"], "target": unit["target"],
        "n_tests": s["n_tests"], "n_validos": s["n_validos"],
        "n_executaveis": len(ex),
        "n_falha_assercao": sum(1 for r in rows if r.get("status") == "fail_assert"),
        "n_redundantes": sum(1 for r in rows if r.get("status") == "pass" and r.get("ganho_marginal", 0) == 0),
        "cobertura_suite": s["cobertura_suite"] if unit.get("facts") else None,
        "mutation_score_suite": s["mutation_score_suite"] if unit.get("facts") else None,
        "n_suite_minima": s["n_suite_minima"], "mutantes": s["mutantes"],
        "mutantes_timeout": s["mutantes_timeout"],
        "relevancia_mediana": round(statistics.median(rel), 4) if rel else None,
        "pct_oraculo_inconsistente": (round(sum(1 for r in orc if r["oraculo_juiz"] == "inconsistent") / len(orc), 4)
                                      if orc else None),
        "score_medio": round(statistics.mean(r.get("score") or 0 for r in rows), 4) if rows else None,
        "aceitar": dec.get("aceitar", 0), "revisar": dec.get("revisar", 0), "recusar": dec.get("recusar", 0),
        "error": (unit.get("error") or "")[:160],
    }


def _killed(unit: Dict) -> Set[int]:
    return {m for ms in unit["facts"].get("kills", {}).values() for m in ms}


def mutant_matrix(units: List[Dict], gens: List[str]) -> Tuple[Dict, Dict]:
    """Por alvo: {mutante: {gerador: fração das amostras que o matam}} e o catálogo de mutantes.

    Os mutantes são determinísticos por alvo (mesma semente), então o id é comparável
    entre geradores; unidades cujo catálogo diverge (código mudou) são ignoradas.
    """
    catalog: Dict[str, Dict[int, Dict]] = {}
    timeouts: Dict[str, Set[int]] = defaultdict(set)
    per: Dict[str, Dict[str, List[Set[int]]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        if not u.get("facts"):
            continue
        muts = {m["id"]: m for m in u["facts"]["mutants"]}
        ref = catalog.setdefault(u["target"], muts)
        if {k: (v["line"], v["desc"]) for k, v in ref.items()} != {k: (v["line"], v["desc"]) for k, v in muts.items()}:
            continue
        timeouts[u["target"]] |= set(u["facts"].get("timeouts", []))
        per[u["target"]][u.get("gerador", "")].append(_killed(u))
    freq: Dict[str, Dict[int, Dict[str, float]]] = {}
    for t, muts in catalog.items():
        freq[t] = {}
        for mid in sorted(muts):
            if mid in timeouts[t]:
                continue
            freq[t][mid] = {g: (round(sum(mid in k for k in per[t][g]) / len(per[t][g]), 4)
                                if per[t].get(g) else None) for g in gens}
    return freq, catalog


def _uncovered_always(units: List[Dict]) -> Dict[str, List[int]]:
    """Por alvo: linhas que NENHUMA amostra desse gerador cobre."""
    out: Dict[str, Set[int]] = {}
    for u in units:
        if not u.get("facts"):
            continue
        s = set(u["facts"].get("suite_uncovered_lines", []))
        out[u["target"]] = s if u["target"] not in out else out[u["target"]] & s
    return {t: sorted(v) for t, v in out.items()}


# ---------------------------------------------------------------- explicação por gerador
def explain_generator(g: str, gunits: List[Dict], grows: List[Dict], umetrics: List[Dict],
                      freq: Dict, catalog: Dict, gens: List[str]) -> Dict:
    n_tests = len(grows)
    ex = [r for r in grows if r.get("executavel")]
    fa = [r for r in grows if r.get("status") == "fail_assert"]
    passed = [r for r in grows if r.get("status") == "pass"]
    red = [r for r in passed if r.get("ganho_marginal", 0) == 0]
    bug = [r for r in grows if r.get("possivel_bug_no_codigo")]
    cov_m, cov_sd = _mean_sd([m["cobertura_suite"] for m in umetrics])
    ms_m, ms_sd = _mean_sd([m["mutation_score_suite"] for m in umetrics])
    rel = [r["relevancia_juiz"] for r in ex if r.get("relevancia_juiz") is not None]
    disc = [r["discordancia"] for r in ex if r.get("discordancia") is not None]
    dec = Counter(r.get("decisao_sugerida") for r in grows)
    others = [o for o in gens if o != g]

    resumo = {
        "gerador": g, "n_suites": len(gunits), "n_testes": n_tests,
        "pct_executaveis": round(len(ex) / n_tests, 4) if n_tests else None,
        "pct_falha_assercao": round(len(fa) / n_tests, 4) if n_tests else None,
        "cobertura_suite": cov_m, "cobertura_suite_dp": cov_sd,
        "mutation_score_suite": ms_m, "mutation_score_suite_dp": ms_sd,
        "pct_redundantes": round(len(red) / len(passed), 4) if passed else None,
        "relevancia_mediana": round(statistics.median(rel), 4) if rel else None,
        "discordancia_media_juizes": round(statistics.mean(disc), 4) if disc else None,
        "score_medio": round(statistics.mean(r.get("score") or 0 for r in grows), 4) if grows else None,
        "aceitar": dec.get("aceitar", 0), "revisar": dec.get("revisar", 0), "recusar": dec.get("recusar", 0),
        "n_possivel_bug": len(bug),
    }

    ev: List[Dict] = []
    miss: List[Dict] = []
    if any(u.get("error") for u in gunits):
        bad = [u["unit_id"] for u in gunits if u.get("error")]
        ev.append({"tipo": "execucao", "texto": f"{len(bad)} de {len(gunits)} suíte(s) não executam "
                   f"(erro de sintaxe/coleta): {', '.join(bad[:3])}.", "refs": {"unidades": bad}})
    ev.append({"tipo": "execucao", "texto": f"{len(ex)} de {n_tests} testes executam "
               f"({_fmt(resumo['pct_executaveis'])}).", "refs": {}})
    if fa:
        ids = [f"{r['unit_id']}::{r['teste_id']}" for r in fa]
        ev.append({"tipo": "correcao", "texto": f"{len(fa)} teste(s) falham no código original (valor esperado "
                   f"incorreto ou bug real), ex.: {', '.join(ids[:3])}.", "refs": {"testes": ids}})
    if bug:
        ids = [f"{r['unit_id']}::{r['teste_id']}" for r in bug]
        ev.append({"tipo": "correcao", "texto": f"{len(bug)} falha(s) têm oráculo considerado coerente com a "
                   f"especificação pelos juízes: possível bug no código, adjudicar à mão ({', '.join(ids[:3])}).",
                   "refs": {"testes": ids}})
    ev.append({"tipo": "cobertura", "texto": f"Cobertura média da suíte {_fmt(cov_m, cov_sd)} "
               f"(média ± desvio entre {len(umetrics)} suíte(s)).", "refs": {}})
    ev.append({"tipo": "deteccao", "texto": f"Mutation score médio {_fmt(ms_m, ms_sd)}.", "refs": {}})

    # mutantes exclusivos / perdidos em relação aos outros geradores
    for t, fm in sorted(freq.items()):
        muts = catalog[t]
        mine = [m for m, f in fm.items() if (f.get(g) or 0) > 0]
        excl = [m for m in mine if all(not (fm[m].get(o) or 0) for o in others)]
        lost = [m for m, f in fm.items() if f.get(g) == 0 and any((f.get(o) or 0) > 0 for o in others)]
        never = [m for m, f in fm.items() if all(not (v or 0) for v in f.values())]
        unstable = [m for m in mine if (fm[m].get(g) or 0) < 1]
        if excl:
            ev.append({"tipo": "deteccao", "texto": f"[{t}] Só este modelo detecta: "
                       + "; ".join(_mut_label(muts, m) for m in excl[:4]) + ".",
                       "refs": {"alvo": t, "mutantes": excl}})
        if lost:
            quem = {m: [o for o in others if (fm[m].get(o) or 0) > 0] for m in lost}
            ev.append({"tipo": "deteccao", "texto": f"[{t}] Outros modelos detectam, este não: "
                       + "; ".join(f"{_mut_label(muts, m)} [{', '.join(quem[m])}]" for m in lost[:4]) + ".",
                       "refs": {"alvo": t, "mutantes": lost}})
            miss.append({"tipo": "mutantes_perdidos", "texto": f"[{t}] Faltam casos que matem "
                         + ", ".join(f"#{m}" for m in lost[:6]) + " (outros modelos conseguem).",
                         "refs": {"alvo": t, "mutantes": lost}})
        if unstable:
            ev.append({"tipo": "estabilidade", "texto": f"[{t}] Detecção instável entre amostras: "
                       + ", ".join(f"#{m} ({fm[m][g]:.0%})" for m in unstable[:5]) + ".",
                       "refs": {"alvo": t, "mutantes": unstable}})
        if never:
            miss.append({"tipo": "mutantes_sobreviventes", "texto": f"[{t}] Nenhum modelo detecta: "
                         + "; ".join(_mut_label(muts, m) for m in never[:4]) + ".",
                         "refs": {"alvo": t, "mutantes": never}})
    for t, lines in _uncovered_always(gunits).items():
        if lines:
            miss.append({"tipo": "linhas_nao_cobertas", "texto": f"[{t}] Linhas nunca cobertas por este modelo: "
                         + ", ".join(f"L{l}" for l in lines[:8]) + ".", "refs": {"alvo": t, "linhas": lines}})

    if passed:
        ids = [f"{r['unit_id']}::{r['teste_id']}" for r in red]
        txt = f"{len(red)} de {len(passed)} testes que passam são redundantes (ganho marginal 0)"
        txt += f", ex.: {', '.join(ids[:3])}." if ids else "."
        ev.append({"tipo": "redundancia", "texto": txt, "refs": {"testes": ids}})
    if rel:
        ev.append({"tipo": "relevancia", "texto": f"[juízes LLM] Relevância mediana {_fmt(resumo['relevancia_mediana'])}; "
                   f"discordância média entre juízes {_fmt(resumo['discordancia_media_juizes'])}.", "refs": {}})
    ev.append({"tipo": "decisao", "texto": f"Decisões sugeridas: {resumo['aceitar']} aceitar, "
               f"{resumo['revisar']} revisar, {resumo['recusar']} recusar.", "refs": {}})
    return {"resumo": resumo, "evidencias": ev, "o_que_falta": miss}


def verify_suite_explanation(expl: Dict, catalog: Dict, gunits: List[Dict], grows: List[Dict]) -> List[str]:
    """Confere toda referência (alvo, mutante, linha, teste, unidade) contra os fatos medidos."""
    problems: List[str] = []
    tests = {f"{r['unit_id']}::{r['teste_id']}" for r in grows}
    units = {u["unit_id"] for u in gunits}
    stmts: Dict[str, Set[int]] = defaultdict(set)
    for u in gunits:
        if u.get("facts"):
            stmts[u["target"]] |= set(u["facts"].get("stmts", []))
    for item in expl["evidencias"] + expl["o_que_falta"]:
        r = item.get("refs", {})
        t = r.get("alvo")
        if t is not None and t not in catalog:
            problems.append(f"alvo inexistente: {t}")
            continue
        for m in r.get("mutantes", []):
            if m not in catalog.get(t, {}):
                problems.append(f"mutante inexistente: {t}#{m}")
        for l in r.get("linhas", []):
            if l not in stmts.get(t, set()):
                problems.append(f"linha inexistente: {t}:L{l}")
        for k in r.get("testes", []):
            if k not in tests:
                problems.append(f"teste inexistente: {k}")
        for k in r.get("unidades", []):
            if k not in units:
                problems.append(f"unidade inexistente: {k}")
    return problems


# ---------------------------------------------------------------- saída em pastas
def _expl_md(expl: Dict, ok: bool, problems: List[str]) -> str:
    r = expl["resumo"]
    lines = [f"# Suíte gerada por `{r['gerador']}`", "",
             f"{r['n_suites']} suíte(s) (alvo × amostra), {r['n_testes']} testes. "
             f"Explicação {'verificada ✓' if ok else 'COM REFERÊNCIAS NÃO VERIFICADAS ⚠'}.", "",
             "| Métrica | Valor |", "|---|---|"]
    for k, (lab, _) in COMPARE.items():
        lines.append(f"| {lab} | {_fmt(r.get(k), r.get(k + '_dp'))} |")
    lines += [f"| Decisões (aceitar / revisar / recusar) | {r['aceitar']} / {r['revisar']} / {r['recusar']} |",
              "", "## Evidências", ""]
    lines += [f"- **{e['tipo']}**: {e['texto']}" for e in expl["evidencias"]]
    lines += ["", "## O que falta", ""]
    lines += [f"- **{e['tipo']}**: {e['texto']}" for e in expl["o_que_falta"]] or ["- nada identificado"]
    if problems:
        lines += ["", "## Problemas de verificação", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def _ranking(resumos: List[Dict]) -> List[str]:
    out = []
    for k, (lab, higher) in COMPARE.items():
        vals = [(r[k], r["gerador"]) for r in resumos if r.get(k) is not None]
        if len(vals) < 2:
            continue
        vals.sort(reverse=higher)
        best, worst = vals[0], vals[-1]
        if best[0] == worst[0]:
            out.append(f"- **{lab}**: empate entre todos ({_fmt(best[0])}).")
        else:
            out.append(f"- **{lab}**: melhor `{best[1]}` ({_fmt(best[0])}), pior `{worst[1]}` ({_fmt(worst[0])}).")
    return out


def write_all(units: List[Dict], rows: List[Dict], gens: List[str], run_dir: Path,
              results_cols: List[str]) -> Dict[str, int]:
    gens = [g for g in gens if any(u.get("gerador") == g for u in units)] or \
        sorted({u.get("gerador", "") for u in units})
    rows_by_unit: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        rows_by_unit[r["unit_id"]].append(r)
    freq, catalog = mutant_matrix(units, gens)

    sroot, croot = run_dir / "suites", run_dir / "comparacao"
    resumos, n_bad = [], 0
    for g in gens:
        gunits = [u for u in units if u.get("gerador") == g]
        grows = [r for u in gunits for r in rows_by_unit[u["unit_id"]]]
        um = [unit_metrics(u, rows_by_unit[u["unit_id"]]) for u in gunits]
        gd = sroot / g
        (gd / "testes").mkdir(parents=True, exist_ok=True)
        for u in gunits:
            src = run_dir / "generated" / u["unit_id"] / "test_generated.py"
            if src.exists():
                shutil.copyfile(src, gd / "testes" / f"{u['unit_id']}.py")
        _write_csv(gd / "metricas_unidades.csv", um, UNIT_COLS)
        flat = [{**r, "reuso_em": ";".join(r.get("reuso_em") or [])} for r in grows]
        _write_csv(gd / "metricas_testes.csv", flat, results_cols)
        expl = explain_generator(g, gunits, grows, um, freq, catalog, gens)
        problems = verify_suite_explanation(expl, catalog, gunits, grows)
        n_bad += bool(problems)
        expl["verificada"], expl["problemas"] = not problems, problems
        (gd / "explicacao.json").write_text(json.dumps(expl, indent=2, ensure_ascii=False), encoding="utf-8")
        (gd / "explicacao.md").write_text(_expl_md(expl, not problems, problems), encoding="utf-8")
        resumos.append(expl["resumo"])

    croot.mkdir(parents=True, exist_ok=True)
    _write_csv(croot / "comparacao_modelos.csv", resumos)
    mrows = [{"alvo": t, "mutante": m, "linha": catalog[t][m]["line"], "operador": catalog[t][m]["op"],
              "descricao": catalog[t][m]["desc"], **{g: fm[m][g] for g in gens}}
             for t, fm in sorted(freq.items()) for m in fm]
    _write_csv(croot / "mutantes_por_alvo.csv", mrows,
               ["alvo", "mutante", "linha", "operador", "descricao"] + gens)

    md = ["# Comparação entre os modelos geradores", "",
          "Métricas brutas (sem normalização relativa ao melhor): média ± desvio entre as suítes "
          "(alvo × amostra) de cada modelo. Relevância vem dos juízes LLM e não tem gabarito objetivo.", "",
          "| Gerador | Suítes | Testes | " + " | ".join(lab for lab, _ in COMPARE.values()) + " |",
          "|---|---|---|" + "---|" * len(COMPARE)]
    for r in resumos:
        md.append(f"| `{r['gerador']}` | {r['n_suites']} | {r['n_testes']} | "
                  + " | ".join(_fmt(r.get(k), r.get(k + "_dp")) for k in COMPARE) + " |")
    md += ["", "## Quem se destaca em cada métrica", ""] + (_ranking(resumos) or ["- só um gerador."])
    md += ["", "## Detecção de defeitos por alvo", "",
           "Fração das amostras de cada modelo que matam o mutante (`mutantes_por_alvo.csv`).", ""]
    for t, fm in sorted(freq.items()):
        md += [f"### {t}", "", "| Mutante | " + " | ".join(gens) + " |", "|---|" + "---|" * len(gens)]
        for m in fm:
            md.append(f"| {_mut_label(catalog[t], m)} | "
                      + " | ".join("—" if fm[m][g] is None else f"{fm[m][g]:.0%}" for g in gens) + " |")
        md.append("")
    md += ["## Explicações por modelo", ""] + [f"- [`{g}`](../suites/{g}/explicacao.md)" for g in gens]
    (croot / "relatorio_comparativo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"geradores": len(gens), "explicacoes_com_problema": n_bad}
