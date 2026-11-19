"""Juiz LLM: relevância, oráculo e PREVISÕES de propriedades mensuráveis.

Além de relevância e oráculo, o juiz prevê (a) se o teste detectaria falhas e
(b) se é redundante com outro teste da suíte. Essas duas previsões e a do oráculo
têm GABARITO objetivo (mutação, ganho marginal, execução), o que permite a tabela
de acertos (acerto / parcial / erro) por par gerador × juiz.

Controles contra viés:
 - o juiz NÃO vê métricas, resultado de execução nem o nome do gerador;
 - a citação da especificação é verificada (precisa ser substring da docstring);
 - ``judge.repeats`` > 1 usa mediana/maioria para reduzir variância.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from .llm import InfraError
from .targets import Target
from .utils import extract_json, safe_name

PROMPTS = Path(__file__).parent / "prompts"
VALID_ORACLE = {"consistent", "inconsistent", "unclear"}
VALID_DETECT = {"yes", "no", "unsure"}
MAX_SUITE_CHARS = 6000


def _suite_text(test_id: str, suite: List[Tuple[str, str]]) -> str:
    parts = [f"# id: {i}\n{c}" for i, c in suite if i != test_id]
    return ("\n\n".join(parts))[:MAX_SUITE_CHARS] or "(none)"


# chaves exigidas pelo prompt do juiz (judge_system.txt); faltando alguma, a resposta é inválida
REQUIRED_KEYS = ("relevance", "relevance_reason", "oracle", "oracle_reason", "spec_quote",
                 "detects_faults", "redundant_with")


def parse_verdict(text: str, test_id: str, suite_ids) -> Dict | None:
    """Interpreta a resposta do juiz. None = JSON inválido, campo faltando ou relevância não inteira."""
    data = extract_json(text or "")
    if not isinstance(data, dict) or any(k not in data for k in REQUIRED_KEYS):
        return None
    try:
        rel = int(data.get("relevance"))
    except (TypeError, ValueError):
        return None
    oracle = str(data.get("oracle", "unclear")).lower()
    detects = str(data.get("detects_faults", "unsure")).lower()
    red = str(data.get("redundant_with", "") or "").strip()
    return {"relevance": min(5, max(1, rel)),
            "relevance_reason": str(data.get("relevance_reason", ""))[:400],
            "oracle": oracle if oracle in VALID_ORACLE else "unclear",
            "oracle_reason": str(data.get("oracle_reason", ""))[:400],
            "spec_quote": str(data.get("spec_quote", ""))[:300],
            "detects": detects if detects in VALID_DETECT else "unsure",
            "redundant_with": red if (red in suite_ids and red != test_id) else ""}


def _one(judge, target: Target, test_id: str, test_code: str,
         suite: List[Tuple[str, str]], stats: Dict) -> Dict | None:
    """Uma avaliação; se a resposta for inválida, repete UMA vez (mesmo prompt)."""
    system = (PROMPTS / "judge_system.txt").read_text(encoding="utf-8")
    user = Template((PROMPTS / "judge_user.txt").read_text(encoding="utf-8")).safe_substitute(
        doc=target.doc or "(no docstring)", source=target.source, test=test_code,
        test_id=test_id, suite=_suite_text(test_id, suite))
    ids = {i for i, _ in suite}
    for attempt in (1, 2):
        text = judge.chat(system, user)
        res = parse_verdict(text, test_id, ids)
        stats["tentativas"] += 1
        if res is not None:
            return res
        stats["respostas_invalidas"].append(str(text)[:1500])   # guarda a resposta crua para diagnóstico
        if attempt == 1:
            stats["invalida_1a_tentativa"] += 1
    return None


def judge_test(judge, target: Target, test_id: str, test_code: str,
               suite: List[Tuple[str, str]], repeats: int = 1) -> Dict:
    stats: Dict = {"tentativas": 0, "invalida_1a_tentativa": 0, "respostas_invalidas": []}
    runs: List[Dict] = [r for r in (_one(judge, target, test_id, test_code, suite, stats)
                                    for _ in range(max(1, repeats))) if r]
    if not runs:
        return {"relevancia_juiz": None, "oraculo_juiz": "unclear", "detecta_juiz": "unsure",
                "redundante_com_juiz": "", "justificativa_relevancia": "",
                "justificativa_oraculo": "", "trecho_spec": "", "trecho_verificado": False,
                "juiz_ok": False, **stats}
    rel = statistics.median(r["relevance"] for r in runs)
    oracle = Counter(r["oracle"] for r in runs).most_common(1)[0][0]
    detects = Counter(r["detects"] for r in runs).most_common(1)[0][0]
    red_votes = Counter(bool(r["redundant_with"]) for r in runs).most_common(1)[0][0]
    red = next((r["redundant_with"] for r in runs if r["redundant_with"]), "") if red_votes else ""
    ref = next(r for r in runs if r["oracle"] == oracle)
    quote = ref["spec_quote"].strip()
    verified = bool(quote) and quote.lower() in (target.doc or "").lower()
    return {"relevancia_juiz": round((rel - 1) / 4, 4), "oraculo_juiz": oracle,
            "detecta_juiz": detects, "redundante_com_juiz": red,
            "justificativa_relevancia": ref["relevance_reason"],
            "justificativa_oraculo": ref["oracle_reason"], "trecho_spec": quote,
            "trecho_verificado": verified, "juiz_ok": True, **stats}


# ------------------------------------------------------------------ gravação incremental
class _Recorder:
    """Repassa as chamadas ao juiz e guarda as respostas brutas (para o arquivo parcial)."""

    def __init__(self, judge):
        self.judge, self.raw = judge, []

    def chat(self, system: str, user: str) -> str:
        text = self.judge.chat(system, user)
        self.raw.append(str(text))
        return text


def partial_path(judge_dir: Path, unit: str, judge_label: str) -> Path:
    """Arquivo parcial (JSON Lines) de uma suíte × juiz; fica fora do glob judge/*.json."""
    return judge_dir / "parciais" / f"{unit}__{safe_name(judge_label)}.jsonl"


def load_partial(path: Path) -> Dict[str, Dict]:
    """Testes já julgados no arquivo parcial (linhas com "resultado"). Linha cortada é ignorada."""
    done: Dict[str, Dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and "resultado" in d:
            done[d["teste_id"]] = d["resultado"]
    return done


def judge_suite(judge, target: Target, tests: List[Tuple[str, str]], suite: List[Tuple[str, str]],
                repeats: int, part: Path, unit: str = "", judge_label: str = ""
                ) -> Tuple[Optional[Dict[str, Dict]], List[str]]:
    """Julga os testes de UMA suíte gravando cada resultado assim que chega (retomável).

    Testes já gravados em ``part`` são pulados. Falha de infraestrutura (InfraError) marca o
    teste como falha_infra no parcial e segue para o próximo. Devolve (resultados na ordem
    da suíte, se todos prontos; senão None, lista de pendentes).
    """
    part.parent.mkdir(parents=True, exist_ok=True)
    done = load_partial(part)
    pending: List[str] = []
    for tid, code in tests:
        if tid in done:
            continue
        rec = _Recorder(judge)
        line = {"unit": unit, "judge": judge_label, "teste_id": tid, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            res = judge_test(rec, target, tid, code, suite, repeats)
            line.update(resultado=res, respostas_brutas=rec.raw)
            done[tid] = res
        except InfraError as e:
            line.update(falha_infra=True, erro=str(e)[:300], respostas_brutas=rec.raw)
            pending.append(tid)
        with part.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    if pending:
        return None, pending
    return {tid: done[tid] for tid, _ in tests}, []


def write_validity(run_dir: Path) -> List[Dict]:
    """judge_validity.csv: por juiz, taxa de resposta inválida antes da repetição e taxa final,
    e, à parte, falhas de infraestrutura (não contam como resposta inválida)."""
    import csv
    agg: Dict[str, Dict[str, int]] = {}
    new = lambda: {"n_veredictos": 0, "invalidos_antes": 0, "invalidos_final": 0,  # noqa: E731
                   "chamadas": 0, "sem_registro": 0, "falha_infra_eventos": 0, "pendentes_falha_infra": 0}
    for f in sorted((run_dir / "judge" / "parciais").glob("*.jsonl")) if (run_dir / "judge" / "parciais").exists() else []:
        infra, ok, label = set(), set(), None
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            label = d.get("judge") or label
            if d.get("falha_infra"):
                infra.add(d["teste_id"])
                a = agg.setdefault(label, new())
                a["falha_infra_eventos"] += 1
            elif "resultado" in d:
                ok.add(d["teste_id"])
        if label and infra - ok:
            agg.setdefault(label, new())["pendentes_falha_infra"] += len(infra - ok)
    for f in sorted((run_dir / "judge").glob("*.json")) if (run_dir / "judge").exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        a = agg.setdefault(d["judge"], new())
        for r in d["results"].values():
            a["n_veredictos"] += 1
            if "invalida_1a_tentativa" not in r:   # arquivo de versão antiga (sem repetição)
                a["sem_registro"] += 1
            a["invalidos_antes"] += 1 if r.get("invalida_1a_tentativa") else 0
            a["invalidos_final"] += 0 if r.get("juiz_ok") else 1
            a["chamadas"] += int(r.get("tentativas", 1))
    rows = [{"juiz": j, **a,
             "taxa_invalida_antes": round(a["invalidos_antes"] / a["n_veredictos"], 4) if a["n_veredictos"] else None,
             "taxa_invalida_final": round(a["invalidos_final"] / a["n_veredictos"], 4) if a["n_veredictos"] else None}
            for j, a in sorted(agg.items())]
    cols = ["juiz", "n_veredictos", "invalidos_antes", "taxa_invalida_antes", "invalidos_final",
            "taxa_invalida_final", "chamadas", "sem_registro", "falha_infra_eventos", "pendentes_falha_infra"]
    with (run_dir / "judge_validity.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows
