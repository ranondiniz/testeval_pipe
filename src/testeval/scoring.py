"""Combina as colunas individuais em um score consolidado e uma decisão sugerida.

Colunas (sempre preservadas): cobertura, redundancia, relevancia,
deteccao_defeitos, correcao -> score.

ATENÇÃO: pesos e limiares são HIPÓTESES de partida. Calibre-os contra um rótulo de
verdade (ex.: o teste detecta o bug real?) e reporte análise de sensibilidade.
"""
from __future__ import annotations

from typing import Dict, List


def correcao(row: Dict) -> float | None:
    """Correção do oráculo = execução no código original (objetiva) + checagem do juiz.

    pass + (consistente|incerto) -> 1.0
    pass + inconsistente         -> 0.5  (o teste passa, mas contradiz a especificação:
                                          pode estar fixando um comportamento bugado)
    falha de asserção            -> 0.0  (oráculo errado OU bug real no código)
    (com vários juízes, "oraculo_juiz" é o consenso; juízes divergentes impedem "aceitar")
    erro de execução             -> None
    """
    obj = row.get("correcao_obj")
    if obj is None:
        return None
    if obj == 1.0 and row.get("oraculo_juiz") == "inconsistent":
        return 0.5
    return obj


def score_rows(rows: List[Dict], scoring: Dict) -> None:
    """Preenche in-place: correcao, score, decisao_sugerida, possivel_bug_no_codigo."""
    w, th = scoring["weights"], scoring["thresholds"]
    relative = scoring.get("normalize", "relative_to_best") == "relative_to_best"
    best: Dict[str, Dict[str, float]] = {}
    for r in rows:
        b = best.setdefault(r["target"], {"cobertura": 0.0, "deteccao_defeitos": 0.0})
        for k in b:
            b[k] = max(b[k], r.get(k) or 0.0)

    for r in rows:
        r["correcao"] = correcao(r)
        r["relevancia"] = r.get("relevancia_juiz")
        r["possivel_bug_no_codigo"] = bool(
            r.get("status") == "fail_assert" and r.get("oraculo_juiz") == "consistent")
        if not r.get("executavel") or r["correcao"] is None:
            r["score"], r["decisao_sugerida"] = 0.0, "recusar"
            continue
        if r["possivel_bug_no_codigo"]:
            # falha + oráculo coerente com a especificação: pode ser um bug real no código.
            r["score"], r["decisao_sugerida"] = 0.0, "revisar"
            continue

        def norm(k: str) -> float:
            v = r.get(k) or 0.0
            if relative and best[r["target"]][k] > 0:
                return v / best[r["target"]][k]
            return v

        parts = {
            "cobertura": norm("cobertura"),
            "deteccao_defeitos": norm("deteccao_defeitos"),
            "relevancia": r["relevancia"] if r["relevancia"] is not None else 0.0,
            "correcao": r["correcao"],
            "redundancia": 1.0 - (r.get("redundancia") if r.get("redundancia") is not None else 1.0),
        }
        total_w = sum(w.values()) or 1.0
        r["score"] = round(sum(w[k] * parts[k] for k in parts) / total_w, 4)
        if r["correcao"] == 0.0:      # veto: oráculo errado não vale nada como teste
            r["score"] = 0.0

        red = r.get("redundancia") if r.get("redundancia") is not None else 1.0
        if r["correcao"] == 0.0 or r["score"] < th["reject"] or r.get("ganho_marginal", 0) == 0:
            r["decisao_sugerida"] = "recusar"
        elif (r["score"] >= th["accept"] and r["correcao"] == 1.0
              and red < th["max_redundancy_accept"]
              and (r.get("discordancia") or 0.0) <= th.get("max_disagreement_accept", 1.0)):
            r["decisao_sugerida"] = "aceitar"
        else:
            r["decisao_sugerida"] = "revisar"
