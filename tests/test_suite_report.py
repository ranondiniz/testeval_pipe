"""Explicação por suíte (por gerador) e pastas de comparação, com dados sintéticos."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from testeval import suite_report as sr  # noqa: E402
from testeval.report import COLS  # noqa: E402

MUTS = [{"id": i, "line": 10 + i, "op": "ROR", "desc": f"m{i}"} for i in (1, 2, 3, 4)]


def _unit(uid, gen, kills, uncovered, error=""):
    facts = {} if error else {"mutants": MUTS, "kills": kills, "timeouts": [], "stmts": [11, 12, 13, 14],
                              "suite_uncovered_lines": uncovered}
    return {"unit_id": uid, "target": "T", "gerador": gen, "error": error, "facts": facts,
            "suite": {"n_tests": 2, "n_validos": 2, "cobertura_suite": 0.5, "mutation_score_suite": 0.5,
                      "n_suite_minima": 1, "mutantes": 4, "mutantes_timeout": 0}}


def _row(uid, gen, tid, status="pass", gain=1):
    return {"unit_id": uid, "target": "T", "gerador": gen, "teste_id": tid, "status": status,
            "executavel": status in ("pass", "fail_assert"), "ganho_marginal": gain, "score": 0.5,
            "decisao_sugerida": "revisar", "relevancia_juiz": 0.75, "oraculo_juiz": "consistent",
            "discordancia": 0.0}


def _data():
    # A mata #1 nas duas amostras e #2 só na primeira; B mata #1 e #3; ninguém mata #4.
    units = [_unit("T__A__s0", "A", {"t1": [1, 2]}, [14]), _unit("T__A__s1", "A", {"t1": [1]}, [13, 14]),
             _unit("T__B__s0", "B", {"t1": [1, 3]}, [])]
    rows = [_row("T__A__s0", "A", "t1"), _row("T__A__s0", "A", "t2", gain=0),
            _row("T__A__s1", "A", "t1"), _row("T__A__s1", "A", "t2", status="fail_assert"),
            _row("T__B__s0", "B", "t1"), _row("T__B__s0", "B", "t2")]
    return units, rows


def test_mutant_matrix_frequencies():
    units, _ = _data()
    freq, catalog = sr.mutant_matrix(units, ["A", "B"])
    assert freq["T"][1] == {"A": 1.0, "B": 1.0}
    assert freq["T"][2] == {"A": 0.5, "B": 0.0}
    assert freq["T"][3] == {"A": 0.0, "B": 1.0}
    assert freq["T"][4] == {"A": 0.0, "B": 0.0}
    assert set(catalog["T"]) == {1, 2, 3, 4}


def test_explanation_evidence_and_verification():
    units, rows = _data()
    freq, catalog = sr.mutant_matrix(units, ["A", "B"])
    ua = [u for u in units if u["gerador"] == "A"]
    ra = [r for r in rows if r["gerador"] == "A"]
    um = [sr.unit_metrics(u, [r for r in ra if r["unit_id"] == u["unit_id"]]) for u in ua]
    e = sr.explain_generator("A", ua, ra, um, freq, catalog, ["A", "B"])
    refs = {(x["tipo"], tuple(x["refs"].get("mutantes", []))) for x in e["evidencias"] + e["o_que_falta"]}
    assert ("deteccao", (2,)) in refs                 # só A detecta #2
    assert ("mutantes_perdidos", (3,)) in refs        # B detecta #3, A não
    assert ("estabilidade", (2,)) in refs             # #2 morto em 50% das amostras de A
    assert ("mutantes_sobreviventes", (4,)) in refs   # ninguém detecta #4
    nunca = [x for x in e["o_que_falta"] if x["tipo"] == "linhas_nao_cobertas"]
    assert nunca and nunca[0]["refs"]["linhas"] == [14]   # interseção entre as amostras
    assert e["resumo"]["pct_falha_assercao"] == 0.25 and e["resumo"]["pct_redundantes"] == 0.3333
    assert sr.verify_suite_explanation(e, catalog, ua, ra) == []
    e["evidencias"].append({"tipo": "x", "texto": "", "refs": {"alvo": "T", "mutantes": [99], "testes": ["nao::existe"]}})
    assert len(sr.verify_suite_explanation(e, catalog, ua, ra)) == 2


def test_write_all_creates_folders():
    units, rows = _data()
    units.append(_unit("T__C__s0", "C", {}, [], error="SyntaxError"))
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        g = rd / "generated" / "T__A__s0"
        g.mkdir(parents=True)
        (g / "test_generated.py").write_text("def test_x(): pass\n", encoding="utf-8")
        out = sr.write_all(units, rows, ["A", "B", "C"], rd, COLS)
        assert out == {"geradores": 3, "explicacoes_com_problema": 0}
        for gen in ("A", "B", "C"):
            for f in ("explicacao.md", "explicacao.json", "metricas_unidades.csv", "metricas_testes.csv"):
                assert (rd / "suites" / gen / f).exists(), (gen, f)
        assert (rd / "suites" / "A" / "testes" / "T__A__s0.py").exists()
        comp = list(csv.DictReader((rd / "comparacao" / "comparacao_modelos.csv").open(encoding="utf-8")))
        assert [c["gerador"] for c in comp] == ["A", "B", "C"]
        md = (rd / "comparacao" / "relatorio_comparativo.md").read_text(encoding="utf-8")
        assert "Mutation score da suíte" in md and "| #2 (L12: m2) | 50% | 0% | — |" in md
        c = json.loads((rd / "suites" / "C" / "explicacao.json").read_text(encoding="utf-8"))
        assert any("não executam" in x["texto"] for x in c["evidencias"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
