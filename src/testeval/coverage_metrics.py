"""Cobertura por teste (linhas + ramos) lida do banco do coverage.py.

Usa ``dynamic_context = test_function``: cada teste vira um "contexto" no
arquivo .coverage, o que permite saber QUAL teste cobriu QUAL linha/ramo.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Set, Tuple

from .utils import norm_ctx

Arc = Tuple[int, int]


def collect_coverage(data_file: Path, rcfile: Path, target_file: Path,
                     start: int, end: int, ranges=None) -> Dict:
    """Retorna {'per_test': {id: {'lines': set, 'branches': set}},
                'stmts': set, 'branches': set, 'kind': 'branch'|'line'}.

    Apenas linhas/ramos dentro de [start, end] (a função alvo) ou de ``ranges``
    (alvo + auxiliares chamados) são considerados.
    """
    rs = ranges or [(start, end)]

    def inside(l: int, strict: bool = False) -> bool:
        return any((a < l if strict else a <= l) and l <= b for a, b in rs)

    from coverage import Coverage  # import tardio: só necessário nesta etapa

    target = str(target_file.resolve())
    cov = Coverage(data_file=str(data_file), config_file=str(rcfile))
    cov.load()
    # linhas executáveis (statements) da função alvo
    _, statements, _, _, _ = cov.analysis2(target)
    # a linha do `def` executa no import (contexto vazio): nenhum teste seria creditado por ela
    stmts: Set[int] = {l for l in statements if inside(l, strict=True)}

    # ramos possíveis: união de executados + não executados no relatório JSON
    branches: Set[Arc] = set()
    kind = "line"
    try:
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "cov.json"
            cov.json_report(morfs=[target], outfile=str(out_json))
            rep = json.loads(out_json.read_text(encoding="utf-8"))
            fdata = next(iter(rep["files"].values()))
            for key in ("executed_branches", "missing_branches"):
                for a, b in fdata.get(key, []):
                    if inside(a):
                        branches.add((a, b))
            kind = "branch"
    except Exception:  # noqa: BLE001  (versões antigas sem ramos no JSON)
        branches, kind = set(), "line"

    data = cov.get_data()
    fkey = next((f for f in data.measured_files() if Path(f).resolve() == Path(target)), None)
    per_test: Dict[str, Dict] = {}
    if fkey is not None:
        for ctx in data.measured_contexts():
            if not ctx:
                continue
            data.set_query_context(ctx)
            lines = {l for l in (data.lines(fkey) or []) if inside(l)}
            arcs = set(data.arcs(fkey) or []) if data.has_arcs() else set()
            tid = norm_ctx(ctx)
            entry = per_test.setdefault(tid, {"lines": set(), "branches": set()})
            entry["lines"] |= lines & stmts
            entry["branches"] |= {a for a in arcs if a in branches}
    return {"per_test": per_test, "stmts": stmts, "branches": branches, "kind": kind}
