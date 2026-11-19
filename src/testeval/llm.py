"""Cliente de LLM agnóstico de provedor + regras de independência gerador/juiz.

Fala com qualquer endpoint compatível com a API de "chat completions"
(OpenAI, OpenRouter, Ollama, vLLM, LM Studio, ...). O modo ``mock`` permite
rodar a pipeline sem rede (teste de fumaça).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from .config import resolve_models

_LOCK = threading.Lock()
_LAST_CALL: Dict[str, float] = {}
NO_RETRY = {400, 401, 402, 403, 404}


def _throttle(group: str, rpm: float) -> None:
    """Espaça chamadas de um mesmo grupo (limite compartilhado, ex.: modelos :free)."""
    gap = 60.0 / rpm
    with _LOCK:
        wait = _LAST_CALL.get(group, 0.0) + gap - time.monotonic()
        _LAST_CALL[group] = time.monotonic() + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def _retry_after(r: requests.Response, default: float, cap: float = 120.0) -> float:
    try:
        return min(float(r.headers.get("Retry-After", default)), cap)
    except ValueError:
        return default


class LLMError(RuntimeError):
    pass


class InfraError(LLMError):
    """Falha de infraestrutura que persistiu após todas as tentativas: 429, 5xx, timeout, rede
    ou erro no corpo do tipo "high demand". Não é resposta inválida do modelo."""


INFRA_MARKERS = ("high demand", "overloaded", "unavailable", "rate limit", "rate-limit", "timeout")


def _infra_body(err: Any, code: Any) -> bool:
    return (isinstance(code, int) and (code == 429 or code >= 500)) or any(
        m in str(err).lower() for m in INFRA_MARKERS)


_REASONING = re.compile(r"<(thought|think|thinking)>.*?</\1>", re.S | re.I)
_OPEN_ONLY = re.compile(r"<(thought|think|thinking)>.*", re.S | re.I)


def strip_reasoning(text: str) -> Tuple[str, int]:
    """Remove blocos <thought>/<think> que alguns modelos (ex.: Gemma 4) escrevem antes da resposta.
    Bloco aberto e nunca fechado (resposta truncada) é descartado até o fim. Devolve (texto, nº de caracteres removidos)."""
    out = _REASONING.sub("", text)
    out = _OPEN_ONLY.sub("", out)
    return out.strip(), len(text) - len(out.strip())


class LLM:
    def __init__(self, c: Dict[str, Any]):
        self.c = c
        self.label = c.get("label") or c.get("model", "model")
        self.family = c.get("family", "unknown")
        self.role = c.get("role", "generator")
        self.last: Dict[str, Any] = {}   # metadados da última chamada (quem serviu, tokens, truncamento)

    def meta(self) -> Dict[str, Any]:
        keys = ("provider", "family", "model", "temperature", "max_tokens", "base_url")
        return {**{k: self.c.get(k) for k in keys}, "ultima_chamada": self.last}

    def _body(self, system: str, user: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.c["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            self.c.get("token_param", "max_tokens"): self.c.get("max_tokens", 2000),
        }
        # Alguns modelos de raciocínio rejeitam 'temperature': supports_temperature: false (ou temperature: null).
        if self.c.get("supports_temperature", True) is not False and self.c.get("temperature") is not None:
            body["temperature"] = self.c["temperature"]
        body.update(self.c.get("extra_body") or {})   # ex.: provider routing, reasoning
        return body

    def _log(self) -> None:
        path = self.c.get("calls_log")
        if path:
            with _LOCK, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": self.label,
                                    "role": self.role, **self.last}, ensure_ascii=False) + "\n")

    def chat(self, system: str, user: str) -> str:
        if self.c.get("provider") == "mock":
            return self._mock(system, user)
        url = self.c["base_url"].rstrip("/") + "/chat/completions"
        key = os.environ.get(self.c.get("api_key_env", ""), "")
        if self.c.get("api_key_env") and not key:
            raise LLMError(f"Variável de ambiente {self.c['api_key_env']} não definida ({self.label}).")
        headers = {"Content-Type": "application/json", **(self.c.get("headers") or {})}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = self._body(system, user)
        # Espera crescente configurável por modelo: backoff_base * 2^tentativa, limitada a backoff_max (s).
        retries = int(self.c.get("retries", 6))
        base = float(self.c.get("backoff_base", 1))
        cap = float(self.c.get("backoff_max", 60))
        ra_cap = max(120.0, cap)
        last: str = ""
        infra = False
        for attempt in range(retries + 1):
            wait = min(base * 2 ** attempt, cap)
            if self.c.get("rpm"):
                _throttle(self.c.get("rate_group") or self.label, float(self.c["rpm"]))
            try:
                r = requests.post(url, headers=headers, json=body,
                                  timeout=self.c.get("timeout", 180))
            except (requests.Timeout, requests.ConnectionError) as e:
                last = f"rede: {e}"
                infra = True
                time.sleep(wait)
                continue
            if r.status_code in NO_RETRY:
                raise LLMError(f"{self.label}: HTTP {r.status_code} (não repetível): {r.text[:300]}")
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                infra = True
                time.sleep(_retry_after(r, wait, ra_cap))
                continue
            try:
                data = r.json()
            except ValueError:
                last = f"resposta não-JSON: {r.text[:200]}"
                infra = False
                time.sleep(wait)
                continue
            if data.get("error"):   # o OpenRouter pode devolver erro dentro de HTTP 200
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else None
                last = f"erro no corpo: {err}"
                if isinstance(code, int) and code in NO_RETRY:
                    raise LLMError(f"{self.label}: {last}")
                infra = _infra_body(err, code)
                time.sleep(wait)
                continue
            choice = (data.get("choices") or [{}])[0]
            raw_content = (choice.get("message") or {}).get("content") or ""
            content, removed = strip_reasoning(raw_content)
            self.last = {"provider": data.get("provider"), "model_servido": data.get("model"),
                         "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                         "truncado": choice.get("finish_reason") == "length",
                         "raciocinio_removido_chars": removed}
            self._log()
            if not content.strip():   # modelo de raciocínio pode gastar todo o max_tokens pensando
                last = f"resposta vazia (finish_reason={choice.get('finish_reason')}); aumente max_tokens"
                infra = False
                time.sleep(wait)
                continue
            return content
        raise (InfraError if infra else LLMError)(
            f"Falha ao chamar {self.label} após {retries + 1} tentativas: {last}")

    def _mock(self, system: str, user: str) -> str:
        if self.role == "judge":
            one = {
                "relevance": int(self.c.get("mock_relevance", 4)), "relevance_reason": "mock",
                "oracle": self.c.get("mock_oracle", "unclear"), "oracle_reason": "mock",
                "spec_quote": "", "detects_faults": self.c.get("mock_detects", "unsure"),
                "redundant_with": "",
            }
            return json.dumps(one)
        return Path(self.c["canned"]).read_text(encoding="utf-8")


def is_self(gen: Dict[str, Any], judge: Dict[str, Any]) -> bool:
    """O juiz é o próprio modelo gerador ou da mesma família? (mocks só comparam o label)."""
    if gen.get("label") == judge.get("label"):
        return True
    if gen.get("provider") == "mock" or judge.get("provider") == "mock":
        return False
    gf, jf = gen.get("family", "unknown"), judge.get("family", "unknown")
    return gf == jf and gf != "unknown"


def check_independence(cfg: Dict[str, Any]) -> List[str]:
    """Valida a configuração de papéis. Retorna avisos (strings); levanta ValueError se inválida."""
    gens, judges = resolve_models(cfg)
    mode = cfg["judging"]["self_judging"]
    if mode not in ("flag", "exclude", "forbid"):
        raise ValueError("judging.self_judging deve ser flag, exclude ou forbid")
    if not gens or not judges:
        raise ValueError("Configure ao menos um modelo gerador e um juiz.")
    for m in gens + judges:
        if m.get("provider") != "mock" and m.get("family", "unknown") == "unknown":
            raise ValueError(f"Declare 'family' do modelo '{m.get('label')}' "
                             "(ex.: openai, google, anthropic, deepseek, meta).")
    warnings: List[str] = []
    for g in gens:
        selfs = [j for j in judges if is_self(g, j)]
        if mode == "forbid" and selfs:
            raise ValueError(f"Gerador '{g['label']}' tem juiz da mesma família: "
                             f"{[j['label'] for j in selfs]} (self_judging: forbid).")
        eligible = [j for j in judges if not (mode == "exclude" and is_self(g, j))]
        if not eligible:
            raise ValueError(f"Gerador '{g['label']}' ficou sem nenhum juiz elegível.")
        if len(eligible) < 3:
            warnings.append(f"'{g['label']}' terá só {len(eligible)} juiz(es); "
                            "com ≥3 (número ímpar) fica mais fácil medir concordância.")
    return warnings


def build_llms(cfg: Dict[str, Any]) -> Tuple[List[LLM], List[LLM], List[str]]:
    warnings = check_independence(cfg)
    gens, judges = resolve_models(cfg)
    # o mesmo modelo pode gerar (temperatura mais alta, variedade) e julgar (temperatura baixa, estabilidade)
    log = str(cfg["run_dir"] / "llm_calls.jsonl") if cfg.get("run_dir") else None
    return ([LLM({**g, "role": "generator", "calls_log": log}) for g in gens],
            [LLM({**j, "role": "judge", "calls_log": log,
                  "temperature": j.get("judge_temperature", 0.0)}) for j in judges],
            warnings)
