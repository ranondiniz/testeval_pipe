"""Juiz retomável: gravação por teste, falha_infra, espera crescente e filtro --only (LLM falso, sem rede)."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from testeval import cli, judge  # noqa: E402
from testeval import llm as L  # noqa: E402
from testeval.targets import extract_targets  # noqa: E402

TOY = ROOT / "examples" / "toy"
GOOD = json.dumps({"relevance": 4, "relevance_reason": "r", "oracle": "consistent", "oracle_reason": "o",
                   "spec_quote": "", "detects_faults": "yes", "redundant_with": ""})
TESTS = [(f"test_{i}", f"def test_{i}():\n    assert True\n") for i in range(5)]


def _t():
    return next(x for x in extract_targets(TOY, ".", ["toy_pkg/validators.py"]) if x.qualname == "classify_age")


class Fake:
    """Juiz falso: `fail_at` = {nº da chamada: exceção}; conta chamadas e o teste pedido."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at or {}

    def chat(self, system, user):
        self.calls += 1
        if self.calls in self.fail_at:
            raise self.fail_at[self.calls]
        return GOOD


def test_interrupt_and_resume_does_not_repeat_calls():
    with tempfile.TemporaryDirectory() as td:
        part = judge.partial_path(Path(td), "unidade", "juiz/x")
        crash = Fake({3: KeyboardInterrupt()})   # queda no meio da 3ª chamada
        with pytest.raises(KeyboardInterrupt):
            judge.judge_suite(crash, _t(), TESTS, TESTS, 1, part, "unidade", "juiz/x")
        assert len(judge.load_partial(part)) == 2   # os 2 primeiros já gravados
        lines = [json.loads(x) for x in part.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["respostas_brutas"] == [GOOD] and lines[0]["resultado"]["juiz_ok"]
        again = Fake()
        res, pend = judge.judge_suite(again, _t(), TESTS, TESTS, 1, part, "unidade", "juiz/x")
        assert again.calls == 3 and not pend   # só os 3 que faltavam
        assert list(res) == [t for t, _ in TESTS]   # ordem da suíte, mesmo formato de judge_test
        assert set(res["test_0"]) == set(judge.judge_test(Fake(), _t(), "test_0", "x", TESTS))
        third = Fake()
        res2, _ = judge.judge_suite(third, _t(), TESTS, TESTS, 1, part)
        assert third.calls == 0 and res2 == res


def test_infra_failure_marks_pending_and_continues():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        part = judge.partial_path(rd / "judge", "u", "gemma")
        f = Fake({2: L.InfraError("Falha ao chamar gemma após 15 tentativas: HTTP 503 high demand")})
        res, pend = judge.judge_suite(f, _t(), TESTS, TESTS, 1, part, "u", "gemma")
        assert res is None and pend == ["test_1"] and f.calls == 5   # seguiu para os próximos
        rows = judge.write_validity(rd)
        g = next(r for r in rows if r["juiz"] == "gemma")
        assert g["falha_infra_eventos"] == 1 and g["pendentes_falha_infra"] == 1
        assert g["invalidos_final"] == 0   # falha_infra não é resposta inválida
        again = Fake()
        res, pend = judge.judge_suite(again, _t(), TESTS, TESTS, 1, part, "u", "gemma")
        assert again.calls == 1 and not pend and len(res) == 5
        g = next(r for r in judge.write_validity(rd) if r["juiz"] == "gemma")
        assert g["falha_infra_eventos"] == 1 and g["pendentes_falha_infra"] == 0
        cols = next(csv.reader((rd / "judge_validity.csv").open(encoding="utf-8")))
        assert "falha_infra_eventos" in cols and "pendentes_falha_infra" in cols


def test_invalid_answer_still_retried_once_not_infra():
    with tempfile.TemporaryDirectory() as td:
        part = Path(td) / "p.jsonl"

        class Bad(Fake):
            def chat(self, s, u):
                self.calls += 1
                return "não é json"
        b = Bad()
        res, pend = judge.judge_suite(b, _t(), TESTS[:1], TESTS[:1], 1, part)
        assert b.calls == 2 and not pend and res["test_0"]["juiz_ok"] is False
        assert json.loads(part.read_text(encoding="utf-8"))["respostas_brutas"] == ["não é json"] * 2


# ------------------------------------------------------------------ espera crescente
class R:
    def __init__(self, status, text):
        self.status_code, self.text, self.headers = status, text, {}

    def json(self):
        return json.loads(self.text)


def _llm(monkeypatch, responses, **c):
    sleeps, it = [], iter(responses)
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: next(it))
    monkeypatch.setattr(L.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("K", "x")
    return L.LLM({"label": "g", "model": "m", "base_url": "http://x", "api_key_env": "K", **c}), sleeps


def test_backoff_configurable_and_infra_error(monkeypatch):
    busy = R(503, '{"error": {"code": 503, "message": "high demand"}}')
    m, sleeps = _llm(monkeypatch, [busy] * 15, retries=14, backoff_max=300)
    with pytest.raises(L.InfraError) as e:
        m.chat("s", "u")
    assert "15 tentativas" in str(e.value)
    assert sleeps == sorted(sleeps) and max(sleeps) == 300 and len(sleeps) == 15
    # padrão (outros modelos): 7 tentativas, espera máxima 60 s, como antes
    m, sleeps = _llm(monkeypatch, [busy] * 7)
    with pytest.raises(L.InfraError):
        m.chat("s", "u")
    assert len(sleeps) == 7 and max(sleeps) == 60


def test_high_demand_in_http200_body_is_infra(monkeypatch):
    body = R(200, json.dumps({"error": {"message": "This model is currently experiencing high demand"}}))
    m, _ = _llm(monkeypatch, [body] * 2, retries=1)
    with pytest.raises(L.InfraError):
        m.chat("s", "u")


def test_empty_answer_exhausted_is_not_infra(monkeypatch):
    empty = R(200, json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
    m, _ = _llm(monkeypatch, [empty] * 2, retries=1)
    with pytest.raises(L.LLMError) as e:
        m.chat("s", "u")
    assert not isinstance(e.value, L.InfraError)


# ------------------------------------------------------------------ --only
class J:
    def __init__(self, label):
        self.label = label


def test_select_judges_only():
    js = [J("gpt6_luna"), J("haiku_5_5"), J("gemma4_31b"), J("nemotron_ultra")]
    assert cli.select_judges(js, None) == js and cli.select_judges(js, "") == js
    assert [j.label for j in cli.select_judges(js, "gpt6_luna,haiku_5_5, nemotron_ultra")] == \
        ["gpt6_luna", "haiku_5_5", "nemotron_ultra"]
    assert [j.label for j in cli.select_judges(js, ["gemma4_31b"])] == ["gemma4_31b"]
    with pytest.raises(SystemExit):
        cli.select_judges(js, "nao_existe")
