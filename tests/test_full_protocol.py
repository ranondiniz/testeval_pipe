"""Protocolo da execução completa: seleção de alvos, repetição do juiz, comparação entre modelos, AC1."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from testeval import compare_models as cm  # noqa: E402
from testeval import hits, judge, selection  # noqa: E402
from testeval.targets import extract_targets  # noqa: E402

TOY = ROOT / "examples" / "toy"
GOOD = {"relevance": 4, "relevance_reason": "r", "oracle": "consistent", "oracle_reason": "o",
        "spec_quote": "", "detects_faults": "yes", "redundant_with": ""}


# ------------------------------------------------------------------ seleção de alvos
def test_decision_points():
    src = '''def f(x, ys):
    if x and x > 1 or x < -1:
        return 1
    elif x == 0:
        return [y for y in ys if y if y > 2]
    for y in ys:
        while y:
            y -= 1
    return 2 if x else 3
'''
    # if + elif + (and, or) + 2 ifs de compreensão + for + while + ternário = 9
    assert selection.decision_points(src) == 9
    assert selection.decision_points("def g():\n    return 1\n") == 0


def test_select_three_bands_two_near_median():
    items = [(f"t{i}", c) for i, c in enumerate([0, 0, 0, 0, 3, 4, 4, 6])]
    chosen, warn = selection.select(items)
    assert not warn and len(chosen) == 6
    assert [c["faixa"] for c in chosen] == ["simples"] * 2 + ["medio"] * 2 + ["complexo"] * 2
    assert [c["id"] for c in chosen] == ["t0", "t1", "t4", "t5", "t6", "t7"]
    few, warn = selection.select(items[:4])
    assert len(few) == 4 and warn


def test_project_targets_filter():
    all_t = extract_targets(TOY, ".", ["toy_pkg/validators.py"], max_targets=100)
    keep = [all_t[0].id]
    assert [t.id for t in extract_targets(TOY, ".", ["toy_pkg/validators.py"], max_targets=0,
                                          only_ids=keep)] == keep
    with pytest.raises(ValueError):
        extract_targets(TOY, ".", ["toy_pkg/validators.py"], only_ids=["nao_existe"])


# ------------------------------------------------------------------ juiz: repetição única
class Seq:
    """Juiz falso que devolve respostas em sequência."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def chat(self, system, user):
        self.calls += 1
        return self.answers.pop(0)


def _t():
    return next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")


def test_judge_retries_once_on_invalid():
    suite = [("test_a", "x")]
    j = Seq("não é json", json.dumps(GOOD))
    r = judge.judge_test(j, _t(), "test_a", "x", suite)
    assert r["juiz_ok"] and j.calls == 2 and r["invalida_1a_tentativa"] == 1
    assert r["respostas_invalidas"] == ["não é json"]
    missing = {k: v for k, v in GOOD.items() if k != "detects_faults"}   # campo faltando = inválido
    j = Seq(json.dumps(missing), json.dumps(missing), json.dumps(GOOD))
    r = judge.judge_test(j, _t(), "test_a", "x", suite)
    assert not r["juiz_ok"] and j.calls == 2 and r["tentativas"] == 2   # só UMA repetição
    j = Seq(json.dumps(GOOD))
    r = judge.judge_test(j, _t(), "test_a", "x", suite)
    assert r["juiz_ok"] and j.calls == 1 and r["invalida_1a_tentativa"] == 0


def test_write_validity():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        (rd / "judge").mkdir()
        res = {"t1": {"juiz_ok": True, "invalida_1a_tentativa": 1, "tentativas": 2},
               "t2": {"juiz_ok": False, "invalida_1a_tentativa": 1, "tentativas": 2},
               "t3": {"juiz_ok": True, "invalida_1a_tentativa": 0, "tentativas": 1},
               "t4": {"juiz_ok": True, "invalida_1a_tentativa": 0, "tentativas": 1}}
        (rd / "judge" / "u__j.json").write_text(json.dumps({"unit": "u", "judge": "j", "results": res}),
                                               encoding="utf-8")
        rows = judge.write_validity(rd)
        assert rows[0]["taxa_invalida_antes"] == 0.5 and rows[0]["taxa_invalida_final"] == 0.25
        assert rows[0]["chamadas"] == 6
        assert (rd / "judge_validity.csv").exists()


# ------------------------------------------------------------------ comparação entre modelos
MUTS = [{"id": i, "line": 10 + i, "op": "ROR", "desc": f"m{i}"} for i in range(1, 6)]


def _unit(g, stmts_unc, br_unc, survivors, timeouts=(), mutants=MUTS):
    return {"unit_id": f"T__{g}__s0", "target": "T", "gerador": g, "error": "",
            "rows": [{"executavel": True, "redundancia": 0.5}],
            "facts": {"stmts": [1, 2, 3, 4], "branches": [[1, 2], [1, 3]],
                      "suite_uncovered_lines": stmts_unc, "suite_uncovered_branches": br_unc,
                      "mutants": mutants, "survivors": list(survivors), "timeouts": list(timeouts)},
            "suite": {"n_tests": 3, "n_validos": 3, "cobertura_suite": 0.5, "mutation_score_suite": 0.4,
                      "n_suite_minima": 2}}


def _three():
    # A: linhas {1,2,3}, ramos {(1,2)}, mortos {1,2}
    # B: linhas {1,2},   ramos {(1,2),(1,3)}, mortos {1,3}       (#5 timeout em B)
    # C: linhas {1,2,4}, ramos {},      mortos {1}
    return [_unit("A", [4], [[1, 3]], [3, 4, 5]),
            _unit("B", [3, 4], [], [2, 4], timeouts=[5]),
            _unit("C", [3], [[1, 2], [1, 3]], [2, 3, 4, 5])]


def test_jaccard_exclusive_survivors():
    assert cm.jaccard({1, 2}, {2, 3}) == round(1 / 3, 4) and cm.jaccard(set(), set()) is None
    res = cm.build(_three())
    ov = {(o["modelo_a"], o["modelo_b"]): o for o in res["overlap"]}
    assert ov[("A", "B")]["jaccard_linhas"] == round(2 / 3, 4)
    assert ov[("A", "B")]["jaccard_ramos"] == 0.5
    assert ov[("A", "B")]["jaccard_mutantes_mortos"] == round(1 / 3, 4)
    assert ov[("A", "C")]["jaccard_ramos"] == 0.0 and ov[("B", "C")]["jaccard_linhas"] == round(2 / 3, 4)
    ex = {e["modelo"]: e for e in res["exclusive"]}
    assert ex["A"]["linhas_exclusivas"] == "L3" and ex["A"]["mutantes_exclusivos"] == "#2"
    assert ex["B"]["ramos_exclusivos"] == "L1->L3" and ex["B"]["mutantes_exclusivos"] == "#3"
    assert ex["C"]["linhas_exclusivas"] == "L4" and ex["C"]["n_mutantes_exclusivos"] == 0
    # #4 sobrevive a todos; #5 teve timeout em B -> fora dos sobreviventes
    assert [s["mutante_id"] for s in res["survivors"]] == [4] and not res["warnings"]


def test_catalog_mismatch_falls_back_to_description():
    other = [{**m, "id": m["id"] + 100} for m in MUTS]   # mesmos mutantes, ids diferentes
    units = [_unit("A", [], [], [3, 4, 5]), _unit("B", [], [], [103, 104, 105], mutants=other)]
    res = cm.build(units)
    assert res["warnings"] and "descrição" in res["warnings"][0]
    assert res["overlap"][0]["jaccard_mutantes_mortos"] == 1.0


def test_write_all_and_summary():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        out = cm.write_all(rd, _three())
        assert out["sobreviventes_comuns"] == 1 and out["pares"] == 3
        for f in ("cross_model_overlap.csv", "cross_model_exclusive.csv", "cross_model_survivors.csv",
                  "resumo_por_modelo.csv"):
            assert (rd / "comparacao" / f).exists(), f
        summ = list(csv.DictReader((rd / "comparacao" / "resumo_por_modelo.csv").open(encoding="utf-8")))
        a = next(s for s in summ if s["modelo"] == "A")
        assert a["n_testes"] == "3" and a["total_linhas_exclusivas"] == "1" and a["n_suite_minima"] == "2"


# ------------------------------------------------------------------ concordância
def test_raw_agreement_and_gwet_ac1():
    items = [["a", "a"], ["a", "a"], ["a", "b"], ["b", "b"]]
    assert hits.raw_agreement(items) == 0.75
    assert hits.gwet_ac1(items, 2) == round((0.75 - 0.46875) / (1 - 0.46875), 4)
    # prevalência alta: kappa baixo/indefinido, AC1 alto
    skew = [["c", "c", "c"]] * 9 + [["c", "c", "u"]]
    assert hits.gwet_ac1(skew, 3) > 0.9 and hits.fleiss_kappa(skew) < 0.1
