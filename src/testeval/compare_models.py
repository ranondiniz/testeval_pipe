"""Comparação entre as suítes dos modelos geradores, por alvo (conjuntos de evidências).

Lê work/<run>/eval/*.json (campos "facts" e "suite") e, por alvo × gerador (unindo
amostras, se houver mais de uma), deriva:
  linhas cobertas   = facts.stmts    - facts.suite_uncovered_lines
  ramos cobertos    = facts.branches - facts.suite_uncovered_branches
  mutantes mortos   = ids de facts.mutants - facts.survivors - facts.timeouts

Os mutantes são os mesmos entre modelos para o mesmo alvo (semente fixa); isso é
verificado. Se os catálogos diferirem, compara-se pela descrição do mutante
(linha, operador, descrição de ``Mutant.brief``) em vez do id, com aviso.

Saídas em work/<run>/comparacao/:
  cross_model_overlap.csv    Jaccard de linhas, ramos e mutantes mortos por par de modelos
  cross_model_exclusive.csv  o que só um modelo cobre/mata, entre todos
  cross_model_survivors.csv  mutantes que sobreviveram a TODAS as suítes do alvo
  resumo_por_modelo.csv      agregado por modelo (todos os alvos)
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Set, Tuple


def jaccard(a: Set, b: Set) -> Optional[float]:
    """|a ∩ b| / |a ∪ b|; None quando os dois conjuntos são vazios (indefinido)."""
    u = a | b
    return round(len(a & b) / len(u), 4) if u else None


def _mkey(m: Dict, by_desc: bool) -> Hashable:
    return (m["line"], m["op"], m["desc"]) if by_desc else m["id"]


def suite_sets(unit: Dict, by_desc: bool = False) -> Dict[str, Set]:
    """Conjuntos cobertos/mortos de UMA unidade (vazios se a suíte não executou)."""
    f = unit.get("facts") or {}
    if not f:
        return {"linhas": set(), "ramos": set(), "mutantes": set()}
    lines = set(f.get("stmts", [])) - set(f.get("suite_uncovered_lines", []))
    branches = ({tuple(b) for b in f.get("branches", [])}
                - {tuple(b) for b in f.get("suite_uncovered_branches", [])})
    dead_ids = ({m["id"] for m in f.get("mutants", [])} - set(f.get("survivors", []))
                - set(f.get("timeouts", [])))
    by_id = {m["id"]: m for m in f.get("mutants", [])}
    return {"linhas": lines, "ramos": branches, "mutantes": {_mkey(by_id[i], by_desc) for i in dead_ids}}


def _catalog(unit: Dict) -> Dict[int, Tuple]:
    return {m["id"]: (m["line"], m["op"], m["desc"]) for m in (unit.get("facts") or {}).get("mutants", [])}


def build(units: List[Dict]) -> Dict:
    """Agrupa por alvo e gerador e calcula sobreposição, exclusivos e sobreviventes."""
    by_target: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        by_target[u["target"]][u.get("gerador", "")].append(u)
    warnings: List[str] = []
    overlap, exclusive, survivors = [], [], []
    per_model_excl: Dict[str, Dict[str, int]] = defaultdict(lambda: {"linhas": 0, "ramos": 0, "mutantes": 0})
    for t, gens in sorted(by_target.items()):
        cats = [_catalog(u) for us in gens.values() for u in us if u.get("facts")]
        same = all(c == cats[0] for c in cats) if cats else True
        if not same:
            warnings.append(f"{t}: catálogos de mutantes diferem entre modelos; comparando por descrição")
        by_desc = not same
        sets: Dict[str, Dict[str, Set]] = {}
        nvalid: Dict[str, int] = {}
        for g, us in sorted(gens.items()):
            acc = {"linhas": set(), "ramos": set(), "mutantes": set()}
            for u in us:   # mais de uma amostra: união dos conjuntos
                for k, v in suite_sets(u, by_desc).items():
                    acc[k] |= v
            sets[g] = acc
            nvalid[g] = sum(u["suite"].get("n_validos", 0) for u in us)
        for a, b in combinations(sorted(sets), 2):
            overlap.append({"alvo": t, "modelo_a": a, "modelo_b": b,
                            "jaccard_linhas": jaccard(sets[a]["linhas"], sets[b]["linhas"]),
                            "jaccard_ramos": jaccard(sets[a]["ramos"], sets[b]["ramos"]),
                            "jaccard_mutantes_mortos": jaccard(sets[a]["mutantes"], sets[b]["mutantes"])})
        for g in sorted(sets):
            others = [sets[o] for o in sets if o != g]
            ex = {k: sets[g][k] - set().union(*[o[k] for o in others]) if others else set(sets[g][k])
                  for k in ("linhas", "ramos", "mutantes")}
            for k in ex:
                per_model_excl[g][k] += len(ex[k])
            exclusive.append({"alvo": t, "modelo": g,
                              "linhas_exclusivas": ";".join(f"L{x}" for x in sorted(ex["linhas"])),
                              "ramos_exclusivos": ";".join(f"L{a}->L{b}" for a, b in sorted(ex["ramos"])),
                              "mutantes_exclusivos": ";".join(_fmt_mut(x) for x in sorted(ex["mutantes"], key=str)),
                              "n_linhas_exclusivas": len(ex["linhas"]), "n_ramos_exclusivos": len(ex["ramos"]),
                              "n_mutantes_exclusivos": len(ex["mutantes"]), "n_testes_validos": nvalid[g]})
        # sobreviventes comuns: mutantes (sem timeout em nenhuma suíte) que nenhum modelo matou
        ref = next((u for us in gens.values() for u in us if u.get("facts")), None)
        if ref is None:
            continue
        timeouts = {_mkey(m, by_desc) for us in gens.values() for u in us if u.get("facts")
                    for m in u["facts"]["mutants"] if m["id"] in set(u["facts"].get("timeouts", []))}
        executed = [u for us in gens.values() for u in us if u.get("facts")]
        if len({u.get("gerador") for u in executed}) < len(gens):
            warnings.append(f"{t}: algum modelo sem suíte executável; sobreviventes calculados só com os que executaram")
        dead_any = set().union(*[s["mutantes"] for s in sets.values()])
        for m in ref["facts"]["mutants"]:
            k = _mkey(m, by_desc)
            if k not in dead_any and k not in timeouts:
                survivors.append({"alvo": t, "mutante_id": m["id"], "linha": m["line"], "operador": m["op"],
                                  "descricao": m["desc"]})
    return {"overlap": overlap, "exclusive": exclusive, "survivors": survivors,
            "per_model_excl": per_model_excl, "warnings": warnings}


def _fmt_mut(x) -> str:
    return f"#{x}" if isinstance(x, int) else f"L{x[0]}:{x[2]}"


def summary_by_model(units: List[Dict], per_model_excl: Dict) -> List[Dict]:
    agg: Dict[str, Dict] = defaultdict(lambda: {"n_testes": 0, "n_executaveis": 0, "n_suite_minima": 0,
                                                 "cob": [], "ms": [], "red": [], "n_suites": 0})
    for u in units:
        a = agg[u.get("gerador", "")]
        a["n_suites"] += 1
        a["n_testes"] += u["suite"].get("n_tests", 0)
        a["n_executaveis"] += sum(1 for r in u.get("rows", []) if r.get("executavel"))
        a["n_suite_minima"] += u["suite"].get("n_suite_minima", 0)
        if u.get("facts"):
            a["cob"].append(u["suite"]["cobertura_suite"])
            a["ms"].append(u["suite"]["mutation_score_suite"])
        a["red"] += [r["redundancia"] for r in u.get("rows", []) if r.get("redundancia") is not None]
    mean = lambda v: round(statistics.mean(v), 4) if v else None  # noqa: E731
    out = []
    for g, a in sorted(agg.items()):
        ex = per_model_excl.get(g, {"linhas": 0, "ramos": 0, "mutantes": 0})
        out.append({"modelo": g, "n_suites": a["n_suites"], "n_testes": a["n_testes"],
                    "n_executaveis": a["n_executaveis"], "n_suite_minima": a["n_suite_minima"],
                    "cobertura_media_suite": mean(a["cob"]), "mutation_score_medio_suite": mean(a["ms"]),
                    "redundancia_media": mean(a["red"]),
                    "total_linhas_exclusivas": ex["linhas"], "total_ramos_exclusivos": ex["ramos"],
                    "total_mutantes_exclusivos": ex["mutantes"]})
    return out


def _write(path: Path, rows: List[Dict], cols: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def write_all(run_dir: Path, units: Optional[List[Dict]] = None) -> Dict:
    if units is None:
        units = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((run_dir / "eval").glob("*.json"))]
    res = build(units)
    out = run_dir / "comparacao"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "cross_model_overlap.csv", res["overlap"],
           ["alvo", "modelo_a", "modelo_b", "jaccard_linhas", "jaccard_ramos", "jaccard_mutantes_mortos"])
    _write(out / "cross_model_exclusive.csv", res["exclusive"],
           ["alvo", "modelo", "linhas_exclusivas", "ramos_exclusivos", "mutantes_exclusivos",
            "n_linhas_exclusivas", "n_ramos_exclusivos", "n_mutantes_exclusivos", "n_testes_validos"])
    _write(out / "cross_model_survivors.csv", res["survivors"],
           ["alvo", "mutante_id", "linha", "operador", "descricao"])
    summ = summary_by_model(units, res["per_model_excl"])
    _write(out / "resumo_por_modelo.csv", summ, list(summ[0]) if summ else ["modelo"])
    return {"pares": len(res["overlap"]), "sobreviventes_comuns": len(res["survivors"]),
            "avisos": res["warnings"]}
