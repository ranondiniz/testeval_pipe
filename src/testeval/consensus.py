"""Consolida as avaliações de vários juízes em campos únicos por teste.

Base do consenso: juízes "cruzados" (de outra família que o gerador), se houver;
caso contrário, todos. As avaliações individuais ficam em ``row['juizes']``.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List

from .llm import is_self


def aggregate(rows: List[Dict], judge_results: Dict[str, Dict[str, Dict]],
              judge_meta: Dict[str, Dict], gen_meta: Dict[str, Dict]) -> None:
    """judge_results: {juiz: {teste_id_unico: resultado}} onde a chave é 'unit_id::teste_id'."""
    for r in rows:
        key = f"{r['unit_id']}::{r['teste_id']}"
        g = gen_meta.get(r.get("gerador", ""), {"label": r.get("gerador", "")})
        per: Dict[str, Dict] = {}
        for jl, res in judge_results.items():
            if key in res:
                per[jl] = {**res[key], "autojulgamento": is_self(g, judge_meta[jl])}
        r["juizes"] = per
        ok = {k: v for k, v in per.items() if v.get("juiz_ok")}
        if not ok:
            r["n_juizes"] = 0
            continue
        cross = {k: v for k, v in ok.items() if not v["autojulgamento"]}
        base = cross or ok
        r["consenso_base"] = "cruzado" if cross else "todos"
        r["n_juizes"] = len(base)
        rels = [v["relevancia_juiz"] for v in base.values() if v["relevancia_juiz"] is not None]
        r["relevancia_juiz"] = round(statistics.median(rels), 4) if rels else None
        r["rel_amplitude"] = round(max(rels) - min(rels), 4) if rels else None
        labels = [v["oraculo_juiz"] for v in base.values()]
        top = Counter(labels).most_common()
        winner = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "unclear"
        r["oraculo_juiz"] = winner
        r["discordancia"] = round(1 - top[0][1] / len(labels), 4)
        ref = next((v for v in base.values() if v["oraculo_juiz"] == winner), next(iter(base.values())))
        for k in ("justificativa_relevancia", "justificativa_oraculo", "trecho_spec", "trecho_verificado"):
            r[k] = ref[k]
        r["juiz_ok"] = True
