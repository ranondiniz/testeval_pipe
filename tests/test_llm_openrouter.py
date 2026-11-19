import json
import pytest
from testeval import llm as L


class R:
    def __init__(self, status=200, data=None, headers=None, text=""):
        self.status_code, self._d, self.headers, self.text = status, data, headers or {}, text or json.dumps(data)
    def json(self):
        if self._d is None:
            raise ValueError
        return self._d


def ok(content="hi", fr="stop"):
    return R(200, {"provider": "P", "model": "m", "usage": {"total_tokens": 3},
                   "choices": [{"message": {"content": content}, "finish_reason": fr}]})


def mk(monkeypatch, responses, **c):
    calls = []
    it = iter(responses)
    def post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return next(it)
    monkeypatch.setattr(L.requests, "post", post)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setenv("K", "x")
    return L.LLM({"label": "a", "model": "m", "base_url": "http://x", "api_key_env": "K", **c}), calls


def test_temperature_optional(monkeypatch):
    m, calls = mk(monkeypatch, [ok()], temperature=0.5, supports_temperature=False)
    m.chat("s", "u")
    assert "temperature" not in calls[0]
    m, calls = mk(monkeypatch, [ok()], temperature=0.5)
    m.chat("s", "u")
    assert calls[0]["temperature"] == 0.5


def test_retry_429_and_http200_error_and_empty(monkeypatch):
    m, calls = mk(monkeypatch, [R(429, {}, {"Retry-After": "1"}), R(200, {"error": {"code": 502}}),
                                ok(""), ok("final")], extra_body={"provider": {"order": ["x"]}})
    assert m.chat("s", "u") == "final"
    assert len(calls) == 4 and calls[0]["provider"] == {"order": ["x"]}
    assert m.last["provider"] == "P"


def test_no_retry_on_400(monkeypatch):
    m, calls = mk(monkeypatch, [R(400, {}, text="bad temperature")])
    with pytest.raises(L.LLMError):
        m.chat("s", "u")
    assert len(calls) == 1


def test_missing_key(monkeypatch):
    m, _ = mk(monkeypatch, [])
    monkeypatch.delenv("K")
    with pytest.raises(L.LLMError):
        m.chat("s", "u")


def test_strip_reasoning_blocks():
    t, n = L.strip_reasoning('<thought>pensando {"x": 1}</thought>\n{"ok": true, "n": 3}')
    assert t == '{"ok": true, "n": 3}' and n > 0
    assert L.strip_reasoning("<think>a</think>b")[0] == "b"
    assert L.strip_reasoning("sem bloco")[0] == "sem bloco"
    assert L.strip_reasoning("resposta <thought>truncado sem fechar")[0] == "resposta"


def test_chat_strips_thought_and_json_parses(monkeypatch):
    from testeval.utils import extract_json
    m, _ = mk(monkeypatch, [ok('<thought>raciocínio com {chaves}</thought>{"ok": true, "n": 3}')])
    out = m.chat("s", "u")
    assert extract_json(out) == {"ok": True, "n": 3}
    assert m.last["raciocinio_removido_chars"] > 0


def test_only_thought_is_retried_as_empty(monkeypatch):
    m, calls = mk(monkeypatch, [ok("<thought>sem fim"), ok("resposta final")])
    assert m.chat("s", "u") == "resposta final"
    assert len(calls) == 2
