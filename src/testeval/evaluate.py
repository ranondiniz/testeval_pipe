"""Avaliação objetiva de um arquivo de teste gerado (uma 'unidade').

Métricas calculadas aqui (todas verificáveis por execução/análise):
  executavel, cobertura, deteccao_defeitos, redundancia, correcao_obj
A relevância e a checagem semântica do oráculo vêm do juiz (judge.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from . import mutation
from .coverage_metrics import collect_coverage
from .execution import ERROR, FAIL_ASSERT, PASS, run_and_parse
from .redundancy import redundancy
from .similarity import most_similar
from .targets import Target
from .utils import list_tests


def _empty(unit_id: str, target: Target, reason: str, rows: List[Dict] | None = None) -> Dict:
    return {"unit_id": unit_id, "target": target.id, "error": reason, "rows": rows or [],
            "facts": {}, "suite": {"n_tests": len(rows or []), "n_validos": 0,
                                    "cobertura_suite": 0.0, "mutation_score_suite": 0.0,
                                    "n_suite_minima": 0, "mutantes": 0, "mutantes_timeout": 0}}


def evaluate_unit(cfg: Dict, target: Target, test_file: Path, unit_id: str,
                  unit_dir: Path, generator: str = "") -> Dict:
    ex = cfg["execution"]
    repo: Path = cfg["project"]["repo_path"]
    src = test_file.read_text(encoding="utf-8")
    try:
        tests = list_tests(src)
    except SyntaxError as e:
        return _empty(unit_id, target, f"SyntaxError: {e}")
    if not tests:
        return _empty(unit_id, target, "nenhum teste encontrado no arquivo gerado")

    out = run_and_parse(test_file, repo, unit_dir / "orig.xml", ex["timeout_tests"],
                        ex["python"], cov_target=repo / target.file)
    if out.collection_error or out.timeout:
        rows = [{"unit_id": unit_id, "target": target.id, "gerador": generator,
                 "teste_id": t["id"], "codigo": t["code"], "status": "collection_error",
                 "msg": out.message, "executavel": False} for t in tests]
        return _empty(unit_id, target, out.message or "erro de coleta", rows)

    status = {t["id"]: out.statuses.get(t["id"], ("missing", "")) for t in tests}
    valid: Set[str] = {tid for tid, (st, _) in status.items() if st == PASS}

    cov = {"per_test": {}, "stmts": set(), "branches": set(), "kind": "line"}
    if valid:
        cov = collect_coverage(out.cov_data_file, out.cov_rcfile, repo / target.file,
                               target.start, target.end, ranges=target.ranges())
    stmts, branches = cov["stmts"], cov["branches"]
    denom_cov = max(1, len(stmts) + len(branches))

    # --- mutação ---
    mutants = mutation.generate_mutants((repo / target.file).read_text(encoding="utf-8"),
                                        target.start, target.end,
                                        ex["max_mutants"], ex["mutation_seed"],
                                        ranges=target.ranges())
    mres = mutation.run_mutants(repo, target.file, mutants, test_file, valid,
                                ex["timeout_mutant"], ex["python"], ex["jobs"]) \
        if valid else {"kills": {}, "timeouts": set()}
    kills: Dict[int, Set[str]] = mres["kills"]
    timeouts: Set[int] = mres["timeouts"]
    denom_mut = max(1, len(mutants) - len(timeouts))
    killed_by: Dict[str, Set[int]] = {tid: set() for tid in valid}
    for mid, tset in kills.items():
        for tid in tset:
            killed_by.setdefault(tid, set()).add(mid)

    # --- pegadas e redundância (só testes válidos) ---
    foot: Dict[str, Set] = {}
    for tid in valid:
        pt = cov["per_test"].get(tid, {"lines": set(), "branches": set()})
        foot[tid] = ({("L", l) for l in pt["lines"]} | {("A",) + tuple(b) for b in pt["branches"]}
                     | {("M", m) for m in killed_by.get(tid, set())})
    red = redundancy(foot)

    # --- similaridade textual dentro da unidade ---
    sim = most_similar([t["id"] for t in tests], [t["code"] for t in tests],
                       cfg["similarity"]["backend"], cfg["similarity"]["embedding_model"])

    # --- linhas/mutantes exclusivos (para explicações) ---
    all_lines: Dict[int, List[str]] = {}
    for tid in valid:
        for l in cov["per_test"].get(tid, {"lines": set()})["lines"]:
            all_lines.setdefault(l, []).append(tid)
    unique_lines = {tid: sorted(l for l, ts in all_lines.items() if ts == [tid]) for tid in valid}
    unique_kills = {tid: sorted(m for m in killed_by.get(tid, set())
                                if kills.get(m, set()) == {tid}) for tid in valid}

    rows: List[Dict] = []
    for t in tests:
        tid = t["id"]
        st, msg = status[tid]
        ran = st in (PASS, FAIL_ASSERT)
        pt = cov["per_test"].get(tid, {"lines": set(), "branches": set()}) if st == PASS else \
            {"lines": set(), "branches": set()}
        r = red.get(tid)
        rows.append({
            "unit_id": unit_id, "target": target.id, "gerador": generator, "teste_id": tid,
            "codigo": t["code"], "status": st, "msg": msg, "executavel": ran,
            # as cinco métricas
            "cobertura": round((len(pt["lines"]) + len(pt["branches"])) / denom_cov, 4),
            "redundancia": r["redundancia"] if r else None,
            "deteccao_defeitos": round(len(killed_by.get(tid, set())) / denom_mut, 4),
            "correcao_obj": {PASS: 1.0, FAIL_ASSERT: 0.0}.get(st),
            # detalhes
            "ganho_marginal": r["ganho"] if r else 0,
            "na_suite_minima": r["na_suite_minima"] if r else False,
            "sim_textual_max": sim[tid][0], "teste_mais_similar": sim[tid][1],
            "linhas_cobertas": len(pt["lines"]), "ramos_cobertos": len(pt["branches"]),
            "mutantes_mortos": len(killed_by.get(tid, set())),
        })

    union_lines = set().union(*[cov["per_test"].get(t, {"lines": set()})["lines"] for t in valid]) if valid else set()
    union_br = set().union(*[cov["per_test"].get(t, {"branches": set()})["branches"] for t in valid]) if valid else set()
    killed_any = {m for m, s in kills.items() if s}
    survivors = [m.id for m in mutants if m.id not in killed_any and m.id not in timeouts]
    facts = {
        "cov_kind": cov["kind"],
        "stmts": sorted(stmts), "branches": sorted(list(b) for b in branches),
        "suite_uncovered_lines": sorted(stmts - union_lines),
        "suite_uncovered_branches": sorted(list(b) for b in branches - union_br),
        "mutants": [m.brief() for m in mutants], "survivors": survivors,
        "timeouts": sorted(timeouts), "unique_lines": unique_lines,
        "kills": {t: sorted(s) for t, s in killed_by.items()}, "unique_kills": unique_kills,
    }
    suite = {
        "n_tests": len(tests), "n_validos": len(valid),
        "cobertura_suite": round((len(union_lines) + len(union_br)) / denom_cov, 4),
        "mutation_score_suite": round(len(killed_any) / denom_mut, 4),
        "n_suite_minima": sum(1 for v in red.values() if v["na_suite_minima"]),
        "mutantes": len(mutants), "mutantes_timeout": len(timeouts),
    }
    return {"unit_id": unit_id, "target": target.id, "error": "", "rows": rows,
            "facts": facts, "suite": suite}
