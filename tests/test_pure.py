"""Testes dos módulos puros e do fluxo de dados (com backend de execução falso).

Rode com:  python tests/test_pure.py   (sem dependências além de numpy/sklearn/pyyaml)
ou com pytest.  O backend falso usa um executor da stdlib em vez de pytest/coverage,
portanto NÃO valida a integração real com essas ferramentas — para isso use `make smoke`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from testeval import evaluate, execution, mutation, scoring  # noqa: E402
from testeval.config import load_config  # noqa: E402
from testeval.execution import Outcome, parse_junit  # noqa: E402
from testeval.explain import explain_row, verify_explanation  # noqa: E402
from testeval.redundancy import redundancy  # noqa: E402
from testeval.similarity import cross_target_reuse, most_similar  # noqa: E402
from testeval.targets import extract_targets  # noqa: E402
from testeval.utils import extract_code, list_tests, norm_ctx  # noqa: E402

TOY = ROOT / "examples" / "toy"


def test_norm_ctx():
    assert norm_ctx("test_generated.test_foo") == "test_foo"
    assert norm_ctx("test_generated.TestX.test_y[p1]") == "TestX.test_y"
    assert norm_ctx("x.y.test_generated.TestX.test_y") == "TestX.test_y"


def test_extract_code_and_list_tests():
    raw = "texto\n```python\ndef test_a():\n    assert 1\n\nclass TestB:\n    def test_c(self):\n        assert 2\n```\nfim"
    code = extract_code(raw)
    ids = [t["id"] for t in list_tests(code)]
    assert ids == ["test_a", "TestB.test_c"], ids
    assert list_tests(code)[1]["code"].startswith("def test_c")


def test_targets_toy():
    ts = extract_targets(TOY, ".", ["toy_pkg/validators.py"], min_lines=3, max_targets=5)
    names = {t.qualname for t in ts}
    assert names == {"classify_age", "can_register"}, names
    t = next(x for x in ts if x.qualname == "classify_age")
    assert t.module == "toy_pkg.validators" and "child" in t.doc and t.start < t.end


def test_mutants_are_valid_and_diverse():
    src = (TOY / "toy_pkg" / "validators.py").read_text(encoding="utf-8")
    t = next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")
    ms = mutation.generate_mutants(src, t.start, t.end, max_mutants=100)
    assert len(ms) >= 8, len(ms)
    ops = {m.op for m in ms}
    assert {"ROR", "NEG", "RVR"} <= ops, ops
    for m in ms:
        compile(m.source, "<m>", "exec")
        assert m.source != src
    # amostragem determinística
    a = mutation.generate_mutants(src, t.start, t.end, max_mutants=5, seed=7)
    b = mutation.generate_mutants(src, t.start, t.end, max_mutants=5, seed=7)
    assert [m.desc for m in a] == [m.desc for m in b] and len(a) == 5


def test_redundancy_greedy():
    foot = {"a": {1, 2, 3}, "b": {1, 2, 3}, "c": {3, 4}, "d": set()}
    r = redundancy(foot)
    kept = {k for k, v in r.items() if v["na_suite_minima"]}
    assert kept in ({"a", "c"}, {"b", "c"}), kept          # um dos duplicados fica
    dup = "b" if "a" in kept else "a"
    assert r[dup]["redundancia"] == 1.0 and r["d"]["redundancia"] == 1.0
    assert 0 < r["c"]["redundancia"] < 1


def test_similarity():
    ids = ["t1", "t2", "t3"]
    codes = ["def test_a():\n    assert f(5) == 'child'",
             "def test_b():\n    assert f(5) == 'child'",
             "def test_c():\n    with raises(ValueError):\n        g(-1)"]
    s = most_similar(ids, codes)
    assert s["t1"][1] == "t2" and s["t1"][0] > 0.5 and s["t3"][0] < s["t1"][0]
    reuse = cross_target_reuse([
        {"key": "u1::t", "target": "A", "code": "def test_x():\n    assert f(1) == 2"},
        {"key": "u2::t", "target": "B", "code": "def test_y():\n    assert g(7) == 9"},
        {"key": "u3::t", "target": "A", "code": "def test_z():\n    with raises(E):\n        f(1)\n    assert 1"},
    ], 0.9)
    assert reuse["u1::t"] == ["u2::t"] and reuse["u3::t"] == []


def test_parse_junit():
    xml = ('<testsuites><testsuite>'
           '<testcase classname="test_generated" name="test_ok"/>'
           '<testcase classname="test_generated" name="test_bad"><failure message="assert 1 == 2">x</failure></testcase>'
           '<testcase classname="test_generated" name="test_exc"><failure message="ValueError: boom">x</failure></testcase>'
           '<testcase classname="test_generated.TestK" name="test_p[1]"/>'
           '<testcase classname="test_generated.TestK" name="test_p[2]"><failure message="assert False">x</failure></testcase>'
           '</testsuite></testsuites>')
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.xml"
        p.write_text(xml, encoding="utf-8")
        r = parse_junit(p)
    assert r["test_ok"][0] == "pass" and r["test_bad"][0] == "fail_assert"
    assert r["test_exc"][0] == "error" and r["TestK.test_p"][0] == "fail_assert"


def test_scoring_and_decisions():
    base = dict(unit_id="u", target="T", executavel=True, status="pass", correcao_obj=1.0,
                oraculo_juiz="consistent", ganho_marginal=3)
    rows = [
        {**base, "teste_id": "good", "cobertura": 0.8, "deteccao_defeitos": 0.8, "relevancia_juiz": 1.0, "redundancia": 0.0},
        {**base, "teste_id": "dup", "cobertura": 0.8, "deteccao_defeitos": 0.8, "relevancia_juiz": 1.0, "redundancia": 1.0, "ganho_marginal": 0},
        {**base, "teste_id": "wrong", "status": "fail_assert", "correcao_obj": 0.0, "oraculo_juiz": "inconsistent",
         "cobertura": 0, "deteccao_defeitos": 0, "relevancia_juiz": 0.5, "redundancia": None, "ganho_marginal": 0},
        {**base, "teste_id": "bug?", "status": "fail_assert", "correcao_obj": 0.0, "oraculo_juiz": "consistent",
         "cobertura": 0, "deteccao_defeitos": 0, "relevancia_juiz": 1.0, "redundancia": None, "ganho_marginal": 0},
        {**base, "teste_id": "crash", "executavel": False, "status": "error", "correcao_obj": None,
         "cobertura": 0, "deteccao_defeitos": 0, "relevancia_juiz": None, "redundancia": None},
        {**base, "teste_id": "suspect", "oraculo_juiz": "inconsistent", "cobertura": 0.8, "deteccao_defeitos": 0.8,
         "relevancia_juiz": 1.0, "redundancia": 0.0},
    ]
    scoring.score_rows(rows, load_config(str(ROOT / "config.smoke.yaml"))["scoring"])
    d = {r["teste_id"]: r for r in rows}
    assert d["good"]["decisao_sugerida"] == "aceitar" and d["good"]["score"] > 0.9
    assert d["dup"]["decisao_sugerida"] == "recusar"
    assert d["wrong"]["decisao_sugerida"] == "recusar" and d["wrong"]["correcao"] == 0.0
    assert d["bug?"]["decisao_sugerida"] == "revisar" and d["bug?"]["possivel_bug_no_codigo"]
    assert d["crash"]["decisao_sugerida"] == "recusar" and d["crash"]["score"] == 0.0
    assert d["suspect"]["correcao"] == 0.5 and d["suspect"]["decisao_sugerida"] == "revisar"


# ----------------------------------------------------------------------------------------
# Backend falso: executa cada teste com a stdlib e rastreia linhas do alvo (settrace).
# ----------------------------------------------------------------------------------------
RUNNER = r'''
import sys, json, importlib.util
test_file, target_file, out_json = sys.argv[1:4]
target_file = __import__("os").path.realpath(target_file)
spec = importlib.util.spec_from_file_location("test_generated", test_file)
mod = importlib.util.module_from_spec(spec)
res = {"collection_error": None, "tests": {}}
try:
    spec.loader.exec_module(mod)
except BaseException as e:
    res["collection_error"] = repr(e)
    json.dump(res, open(out_json, "w", encoding="utf-8")); sys.exit(0)
def run(fn):
    lines = set()
    def local(frame, event, arg):
        if event == "line": lines.add(frame.f_lineno)
        return local
    def glob(frame, event, arg):
        if __import__("os").path.realpath(frame.f_code.co_filename) == target_file: return local
    sys.settrace(glob)
    try:
        fn(); st, msg = "pass", ""
    except AssertionError as e:
        st, msg = "fail_assert", "assert " + str(e)
    except BaseException as e:
        st, msg = "error", repr(e)
    finally:
        sys.settrace(None)
    return st, msg, sorted(lines)
for name in dir(mod):
    if name.startswith("test") and callable(getattr(mod, name)):
        st, msg, lines = run(getattr(mod, name))
        res["tests"][name] = {"status": st, "msg": msg, "lines": lines}
json.dump(res, open(out_json, "w", encoding="utf-8"))
'''


def fake_run_and_parse(test_file, workspace, junit_xml, timeout, python="python3", cov_target=None):
    junit_xml.parent.mkdir(parents=True, exist_ok=True)
    runner = junit_xml.parent / (junit_xml.stem + "_runner.py")  # um por execução: evita corrida entre threads
    runner.write_text(RUNNER, encoding="utf-8")
    out_json = junit_xml.parent / (junit_xml.stem + ".json")
    target = cov_target or (Path(workspace) / "toy_pkg" / "validators.py")
    env = {"PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
    try:
        subprocess.run([sys.executable, str(runner), str(test_file), str(Path(workspace) / "toy_pkg" / "validators.py"),
                        str(out_json)], cwd=workspace, env=env, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        return Outcome(timeout=True, message="timeout")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    o = Outcome()
    if data["collection_error"]:
        o.collection_error, o.message = True, data["collection_error"]
        return o
    o.statuses = {k: (v["status"], v["msg"]) for k, v in data["tests"].items()}
    o.cov_data_file = out_json
    o.cov_rcfile = out_json
    return o


def fake_collect_coverage(data_file, rcfile, target_file, start, end, ranges=None):
    import ast
    src = Path(target_file).read_text(encoding="utf-8")
    stmts = {n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.stmt)
             and start < n.lineno <= end and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))}
    data = json.loads(Path(data_file).read_text(encoding="utf-8"))["tests"]
    per = {k: {"lines": {l for l in v["lines"] if l in stmts}, "branches": set()} for k, v in data.items()}
    return {"per_test": per, "stmts": stmts, "branches": set(), "kind": "line"}


def test_end_to_end_with_fake_backend():
    cfg = load_config(str(ROOT / "config.smoke.yaml"))
    cfg["execution"].update(max_mutants=40, jobs=2)
    t = next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")
    orig = (evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage)
    evaluate.run_and_parse = mutation.run_and_parse = fake_run_and_parse
    evaluate.collect_coverage = fake_collect_coverage
    try:
        with tempfile.TemporaryDirectory() as td:
            udir = Path(td)
            tf = udir / "test_generated.py"
            tf.write_text((TOY / "canned_tests.py").read_text(encoding="utf-8"), encoding="utf-8")
            res = evaluate.evaluate_unit(cfg, t, tf, "u1", udir, "mock")
    finally:
        evaluate.run_and_parse, mutation.run_and_parse, evaluate.collect_coverage = orig

    rows = {r["teste_id"]: r for r in res["rows"]}
    assert res["error"] == "", res["error"]
    assert rows["test_wrong_oracle_teen"]["status"] == "fail_assert"
    assert rows["test_wrong_oracle_teen"]["correcao_obj"] == 0.0
    assert rows["test_child"]["status"] == "pass" and rows["test_child"]["cobertura"] > 0
    assert rows["test_trivial"]["cobertura"] == 0 and rows["test_trivial"]["redundancia"] == 1.0
    # duplicados: exatamente um permanece na suíte mínima
    pair = [rows["test_child"]["na_suite_minima"], rows["test_child_duplicate"]["na_suite_minima"]]
    assert sorted(pair) == [False, True], pair
    # a mutação mata de verdade: o teste de fronteira (18) deve matar algum mutante exclusivo
    assert rows["test_adult_lower_boundary"]["mutantes_mortos"] > 0
    assert res["suite"]["mutation_score_suite"] > 0.6, res["suite"]
    assert res["suite"]["cobertura_suite"] > 0.8, res["suite"]
    assert rows["test_wrong_oracle_teen"]["mutantes_mortos"] == 0   # nunca conta como 'matar' mutante

    # pontuação + explicação + verificador
    for r in res["rows"]:
        r.update({"relevancia_juiz": 0.75, "oraculo_juiz": "unclear", "juiz_ok": True,
                  "justificativa_relevancia": "ok", "trecho_spec": "", "trecho_verificado": False})
    scoring.score_rows(res["rows"], cfg["scoring"])
    for r in res["rows"]:
        e = explain_row(r, res["facts"], res["suite"], [])
        ok, problems = verify_explanation(e, res["facts"], list(rows), t.doc)
        assert ok, (r["teste_id"], problems)
    by = {r["teste_id"]: r for r in res["rows"]}
    assert by["test_wrong_oracle_teen"]["decisao_sugerida"] == "recusar"
    assert by["test_trivial"]["decisao_sugerida"] == "recusar"
    assert by["test_adult_lower_boundary"]["decisao_sugerida"] in ("aceitar", "revisar")
    # o verificador pega referência inventada
    bad = explain_row(by["test_child"], res["facts"], res["suite"], [])
    bad["evidencias"].append({"tipo": "cobertura", "texto": "x", "refs": {"linhas": [9999]}})
    assert not verify_explanation(bad, res["facts"], list(rows), t.doc)[0]


def test_collection_error_unit():
    cfg = load_config(str(ROOT / "config.smoke.yaml"))
    t = next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "test_generated.py"
        tf.write_text("def test_x(:\n")
        res = evaluate.evaluate_unit(cfg, t, tf, "u2", Path(td), "mock")
    assert res["error"].startswith("SyntaxError") and res["rows"] == []


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
