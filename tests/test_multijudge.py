"""Testes da v2: vários juízes, autojulgamento, consenso, tabela de acertos, concordância.

Rode com: python tests/test_multijudge.py   (ou pytest). Mesmo aviso do test_pure.py:
usa executor falso em vez de pytest/coverage.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_pure as tp  # noqa: E402  (reaproveita o backend falso)
from testeval import cli, consensus, evaluate, hits, judge, mutation  # noqa: E402
from testeval.config import load_config, resolve_models  # noqa: E402
from testeval.llm import LLM, build_llms, check_independence, is_self  # noqa: E402
from testeval.targets import extract_targets  # noqa: E402

TOY = ROOT / "examples" / "toy"


def _cfg(**over):
    cfg = load_config(str(ROOT / "config.smoke.yaml"))
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    return cfg


def test_fleiss_kappa_known_value():
    # Exemplo clássico (Wikipedia): 10 itens, 14 avaliadores, 5 categorias -> kappa ≈ 0,210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    items = [[str(c) for c, k in enumerate(row) for _ in range(k)] for row in table]
    assert abs(hits.fleiss_kappa(items) - 0.210) < 0.002, hits.fleiss_kappa(items)
    assert hits.fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == 1.0
    assert hits.fleiss_kappa([["a", "a"], ["a", "a"]]) is None        # sem variação
    assert hits.fleiss_kappa([["a", "b"]]) is None                     # itens insuficientes


def test_hit_functions():
    p = {"status": "pass", "mutantes_mortos": 2, "ganho_marginal": 3, "redundancia": 0.1}
    f = {"status": "fail_assert", "mutantes_mortos": 0, "ganho_marginal": 0}
    ok = lambda **k: {"juiz_ok": True, **k}  # noqa: E731
    assert hits.hit_oracle(p, ok(oraculo_juiz="consistent")) == "acerto"
    assert hits.hit_oracle(p, ok(oraculo_juiz="inconsistent")) == "erro"
    assert hits.hit_oracle(f, ok(oraculo_juiz="inconsistent")) == "acerto"
    assert hits.hit_oracle(f, ok(oraculo_juiz="unclear")) == "parcial"
    assert hits.hit_oracle({"status": "error"}, ok(oraculo_juiz="consistent")) is None
    assert hits.hit_detects(p, ok(detecta_juiz="yes")) == "acerto"
    assert hits.hit_detects(p, ok(detecta_juiz="no")) == "erro"
    assert hits.hit_detects(p, ok(detecta_juiz="unsure")) == "parcial"
    assert hits.hit_detects(f, ok(detecta_juiz="yes")) is None          # só testes que passam
    dup = {"status": "pass", "ganho_marginal": 0, "redundancia": 1.0}
    assert hits.hit_redundant(dup, ok(redundante_com_juiz="test_x")) == "acerto"
    assert hits.hit_redundant(dup, ok(redundante_com_juiz="")) == "erro"
    assert hits.hit_redundant(p, ok(redundante_com_juiz="test_x")) == "erro"
    assert hits.hit_redundant(p, ok(redundante_com_juiz="")) == "acerto"


def test_independence_modes():
    base = {"judging": {"self_judging": "flag"}, "llm": {"models": [
        {"label": "a", "family": "f1", "provider": "x"}, {"label": "b", "family": "f2", "provider": "x"},
        {"label": "c", "family": "f3", "provider": "x"}, {"label": "d", "family": "f1", "provider": "x"}]}}
    gens, judges = resolve_models(base)
    assert len(gens) == len(judges) == 4
    assert is_self(gens[0], judges[0]) and is_self(gens[0], judges[3]) and not is_self(gens[0], judges[1])
    check_independence(base)                                              # flag: ok
    ex = {**base, "judging": {"self_judging": "exclude"}}
    check_independence(ex)
    fb = {**base, "judging": {"self_judging": "forbid"}}
    try:
        check_independence(fb)
        raise AssertionError("forbid deveria falhar")
    except ValueError:
        pass
    unk = {"judging": {"self_judging": "flag"}, "llm": {"models": [{"label": "a", "provider": "x"}]}}
    try:
        check_independence(unk)
        raise AssertionError("family ausente deveria falhar")
    except ValueError as e:
        assert "family" in str(e)
    only = {"judging": {"self_judging": "exclude"}, "llm": {"models": [{"label": "a", "family": "f", "provider": "x"}]}}
    try:
        check_independence(only)
        raise AssertionError("sem juiz elegível deveria falhar")
    except ValueError:
        pass
    _, js, _ = build_llms({**base, "judging": {"self_judging": "flag"}})
    assert all(j.c["temperature"] == 0.0 for j in js)                      # juiz: temperatura baixa


def test_targets_exclude_and_import_name():
    ts = extract_targets(TOY, ".", ["toy_pkg/validators.py"], exclude_names=["can_register"])
    assert [t.qualname for t in ts] == ["classify_age"] and ts[0].import_name == "classify_age"
    src = "class Doc:\n    def validate(self, x):\n        \"\"\"v\"\"\"\n        if x:\n            return True\n        return False\n    def generate(self):\n        a = 1\n        b = 2\n        return a + b\n"
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m.py").write_text(src)
        ts = extract_targets(Path(td), ".", ["m.py"], exclude_names=["generate"])
    assert [t.qualname for t in ts] == ["Doc.validate"] and ts[0].import_name == "Doc"


class FakeJudge:
    def __init__(self, payload):
        self.payload, self.label = payload, "fake"

    def chat(self, system, user):
        assert "JUDGE" in system and "Other tests in the same suite" in user
        return "Claro!\n" + json.dumps(self.payload)


def test_judge_parsing_and_quote_verification():
    t = next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")
    suite = [("test_a", "def test_a(): ..."), ("test_b", "def test_b(): ...")]
    good = {"relevance": 5, "relevance_reason": "r", "oracle": "consistent", "oracle_reason": "o",
            "spec_quote": 'Returns "child" for ages 0-12', "detects_faults": "YES", "redundant_with": "test_b"}
    r = judge.judge_test(FakeJudge(good), t, "test_a", "def test_a(): ...", suite)
    assert r["juiz_ok"] and r["relevancia_juiz"] == 1.0 and r["detecta_juiz"] == "yes"
    assert r["redundante_com_juiz"] == "test_b" and r["trecho_verificado"] is True
    fake_quote = {**good, "spec_quote": "isto não existe na docstring", "redundant_with": "test_zzz",
                  "relevance": 99, "oracle": "talvez"}
    r = judge.judge_test(FakeJudge(fake_quote), t, "test_a", "x", suite)
    assert r["trecho_verificado"] is False and r["redundante_com_juiz"] == "" \
        and r["relevancia_juiz"] == 1.0 and r["oraculo_juiz"] == "unclear"
    bad = judge.judge_test(FakeJudge({"foo": 1}), t, "test_a", "x", suite)
    assert bad["juiz_ok"] is False


def test_consensus_majority_and_self_flag():
    gen_meta = {"g1": {"label": "g1", "family": "A", "provider": "x"}}
    judge_meta = {"g1": {"label": "g1", "family": "A", "provider": "x"},
                  "j2": {"label": "j2", "family": "B", "provider": "x"},
                  "j3": {"label": "j3", "family": "C", "provider": "x"},
                  "j4": {"label": "j4", "family": "D", "provider": "x"}}
    mk = lambda o, rel: {"juiz_ok": True, "oraculo_juiz": o, "relevancia_juiz": rel,  # noqa: E731
                         "justificativa_relevancia": "j", "justificativa_oraculo": "", "trecho_spec": "",
                         "trecho_verificado": False, "detecta_juiz": "yes", "redundante_com_juiz": ""}
    rows = [{"unit_id": "u", "teste_id": "t", "gerador": "g1"}]
    res = {"g1": {"u::t": mk("consistent", 1.0)}, "j2": {"u::t": mk("consistent", 0.75)},
           "j3": {"u::t": mk("inconsistent", 0.25)}, "j4": {"u::t": mk("consistent", 0.5)}}
    consensus.aggregate(rows, res, judge_meta, gen_meta)
    r = rows[0]
    assert r["juizes"]["g1"]["autojulgamento"] is True and r["juizes"]["j2"]["autojulgamento"] is False
    assert r["consenso_base"] == "cruzado" and r["n_juizes"] == 3        # g1 (próprio) fica fora do consenso
    assert r["oraculo_juiz"] == "consistent" and abs(r["discordancia"] - 1 / 3) < 1e-3
    assert r["relevancia_juiz"] == 0.5 and r["rel_amplitude"] == 0.5
    # empate -> "unclear"
    rows2 = [{"unit_id": "u", "teste_id": "t", "gerador": "g1"}]
    consensus.aggregate(rows2, {"j2": {"u::t": mk("consistent", 1)}, "j3": {"u::t": mk("inconsistent", 1)}},
                        judge_meta, gen_meta)
    assert rows2[0]["oraculo_juiz"] == "unclear" and rows2[0]["discordancia"] == 0.5


def test_full_multijudge_flow_with_fake_backend():
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(workdir=Path(td))
        cfg["workdir"] = Path(td)
        cfg["run_dir"] = Path(td) / "smoke"
        cfg["run_dir"].mkdir()
        cfg["execution"].update(max_mutants=40, jobs=2)
        orig = (evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage)
        evaluate.run_and_parse = mutation.run_and_parse = tp.fake_run_and_parse
        evaluate.collect_coverage = tp.fake_collect_coverage
        try:
            cli.stage_targets(cfg)
            units = cli.stage_generate(cfg)
            assert len(units) == 2                                   # 1 alvo × 2 geradores × 1 amostra
            cli.stage_evaluate(cfg)
            cli.stage_judge(cfg)
            cli.stage_report(cfg)
        finally:
            evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage = orig
        rd = cfg["run_dir"]
        info = json.loads((rd / "run_info.json").read_text(encoding="utf-8"))
        assert info["repo"]["commit"] and info["ambiente"]["python"] and len(info["geradores"]) == 2
        assert len(info["juizes"]) == 3 and "api_key" not in json.dumps(info).lower()
        assert len(list((rd / "judge").glob("*.json"))) == 2 * 3     # 2 unidades × 3 juízes
        rows = list(csv.DictReader((rd / "results.csv").open(encoding="utf-8")))
        assert len(rows) == 16 and {r["n_juizes"] for r in rows} == {"2"}   # 3 juízes, menos o próprio gerador
        a = [r for r in rows if r["gerador"] == "mock_a" and r["teste_id"] == "test_child"][0]
        assert a["n_juizes"] == "2"                                   # mock_a julga a si mesmo (fora do consenso)
        assert a["consenso_base"] == "cruzado" and a["explicacao_verificada"] == "True"
        summ = list(csv.DictReader((rd / "hits_summary.csv").open(encoding="utf-8")))
        cell = [s for s in summ if s["gerador"] == "mock_a" and s["juiz"] == "mock_a" and s["dimensao"] == "oraculo"][0]
        assert cell["autojulgamento"] == "True" and cell["n"] == "8", cell
        # mock_a (sempre "consistent"): acerta nos testes que passam (7) e erra no oráculo errado (1)
        assert (cell["acertos"], cell["erros"], cell["parciais"]) == ("7", "1", "0"), cell
        cc = [s for s in summ if s["gerador"] == "mock_a" and s["juiz"] == "mock_c" and s["dimensao"] == "oraculo"][0]
        assert (cc["acertos"], cc["erros"]) == ("1", "7"), cc      # mock_c (sempre "inconsistent") é o inverso
        bb = [s for s in summ if s["gerador"] == "mock_a" and s["juiz"] == "mock_b" and s["dimensao"] == "oraculo"][0]
        assert bb["parciais"] == "8", bb                           # "unclear" => parcial
        md = (rd / "hits_matrix.md").read_text(encoding="utf-8")
        assert "†" in md and "mock_c" in md
        agr = list(csv.DictReader((rd / "agreement.csv").open(encoding="utf-8")))
        assert any(x["dimensao"] == "oraculo" and x["n_juizes"] == "3" for x in agr)
        html = (rd / "report.html").read_text(encoding="utf-8")
        assert "juízes avaliaram" in html


def test_exclude_mode_skips_self_pairs():
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg()
        cfg["workdir"] = Path(td)
        cfg["run_dir"] = Path(td) / "x"
        cfg["run_dir"].mkdir()
        cfg["judging"] = {"self_judging": "exclude"}
        orig = (evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage)
        evaluate.run_and_parse = mutation.run_and_parse = tp.fake_run_and_parse
        evaluate.collect_coverage = tp.fake_collect_coverage
        try:
            cli.stage_targets(cfg); cli.stage_generate(cfg); cli.stage_evaluate(cfg); cli.stage_judge(cfg)
        finally:
            evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage = orig
        files = sorted(p.name for p in (cfg["run_dir"] / "judge").glob("*.json"))
        assert len(files) == 2 * 3 - 2, files                        # cada gerador não julga a si mesmo
        assert not any("mock_a__s0__mock_a" in f for f in files)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print(f"FAIL  {name}: {e!r}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
