"""Tabela de acertos (acerto / parcial / erro) por par gerador × juiz + concordância.

Gabaritos objetivos:
  oraculo    : teste passa no código original  -> "consistent";  falha de asserção -> "inconsistent"
  detecta    : mata ao menos 1 mutante         -> "yes";          nenhum            -> "no"
  redundante : redundância >= limiar (padrão 0,5) ou ganho marginal 0
"unclear"/"unsure" contam como PARCIAL. 'Parcial' não entra como acerto na taxa estrita.
Limite: oráculo "inconsistent" + teste que passa é ambíguo (o juiz pode ter razão e o código
estar errado) — adjudique esses casos manualmente.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

OK, PARTIAL, ERR = "acerto", "parcial", "erro"
DIMS = ("oraculo", "detecta", "redundante")


def hit_oracle(row: Dict, res: Dict) -> Optional[str]:
    if row.get("status") not in ("pass", "fail_assert") or not res.get("juiz_ok"):
        return None
    o = res.get("oraculo_juiz")
    if o == "unclear":
        return PARTIAL
    truth = "consistent" if row["status"] == "pass" else "inconsistent"
    return OK if o == truth else ERR


def hit_detects(row: Dict, res: Dict) -> Optional[str]:
    if row.get("status") != "pass" or not res.get("juiz_ok"):
        return None
    d = res.get("detecta_juiz")
    if d == "unsure":
        return PARTIAL
    return OK if (d == "yes") == (row.get("mutantes_mortos", 0) > 0) else ERR


def hit_redundant(row: Dict, res: Dict, threshold: float = 0.5) -> Optional[str]:
    if row.get("status") != "pass" or not res.get("juiz_ok"):
        return None
    truth = row.get("ganho_marginal", 1) == 0 or (row.get("redundancia") or 0) >= threshold
    return OK if bool(res.get("redundante_com_juiz")) == truth else ERR


def build_records(rows: List[Dict], threshold: float = 0.5) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        for jl, res in (r.get("juizes") or {}).items():
            for dim, fn in (("oraculo", hit_oracle), ("detecta", hit_detects),
                            ("redundante", lambda a, b: hit_redundant(a, b, threshold))):
                h = fn(r, res)
                if h is not None:
                    out.append({"gerador": r.get("gerador", ""), "juiz": jl, "dimensao": dim,
                                "teste": f"{r['unit_id']}::{r['teste_id']}", "resultado": h,
                                "autojulgamento": bool(res.get("autojulgamento"))})
    return out


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 1,
                 alpha: float = 0.05) -> tuple:
    if not values:
        return (None, None)
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return (round(means[int(alpha / 2 * n_boot)], 4), round(means[int((1 - alpha / 2) * n_boot) - 1], 4))


def summarize(records: List[Dict], n_boot: int = 2000) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for rec in records:
        groups[(rec["gerador"], rec["juiz"], rec["dimensao"])].append(rec)
    rows = []
    for (g, j, d), recs in sorted(groups.items()):
        c = Counter(x["resultado"] for x in recs)
        n = len(recs)
        strict = [1.0 if x["resultado"] == OK else 0.0 for x in recs]
        lo, hi = bootstrap_ci(strict, n_boot)
        rows.append({"gerador": g, "juiz": j, "dimensao": d, "n": n, "acertos": c[OK],
                     "parciais": c[PARTIAL], "erros": c[ERR],
                     "taxa_acerto": round(c[OK] / n, 4),
                     "taxa_acerto_com_parcial": round((c[OK] + 0.5 * c[PARTIAL]) / n, 4),
                     "ic95_baixo": lo, "ic95_alto": hi,
                     "autojulgamento": recs[0]["autojulgamento"]})
    return rows


def fleiss_kappa(items: List[List[str]]) -> Optional[float]:
    """Kappa de Fleiss. Usa só os itens com o número de avaliadores mais frequente."""
    items = [r for r in items if r]
    if not items:
        return None
    n = Counter(len(r) for r in items).most_common(1)[0][0]
    items = [r for r in items if len(r) == n]
    if n < 2 or len(items) < 2:
        return None
    cats = sorted({c for r in items for c in r})
    N = len(items)
    p_e = sum((sum(r.count(c) for r in items) / (N * n)) ** 2 for c in cats)
    p_bar = sum((sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in items) / N
    if p_e >= 1:
        return None   # sem variação: kappa indefinido
    return round((p_bar - p_e) / (1 - p_e), 4)


# nº de categorias possíveis por dimensão (para o AC1 de Gwet)
N_CATS = {"oraculo": 3, "detecta": 3, "redundante": 2}


def raw_agreement(items: List[List[str]]) -> Optional[float]:
    """Concordância bruta média entre pares de juízes (aceita nº de juízes variável por item)."""
    items = [r for r in items if len(r) >= 2]
    if not items:
        return None
    pa = [sum(r.count(c) * (r.count(c) - 1) for c in set(r)) / (len(r) * (len(r) - 1)) for r in items]
    return round(sum(pa) / len(pa), 4)


def gwet_ac1(items: List[List[str]], q: int) -> Optional[float]:
    """AC1 de Gwet para vários avaliadores; q = nº de categorias possíveis da escala.

    Menos sensível que o kappa ao "paradoxo" de prevalência alta (quase todos os
    votos numa categoria), comum aqui (ex.: oráculo quase sempre "consistent").
    """
    items = [r for r in items if len(r) >= 2]
    if not items or q < 2:
        return None
    pa = raw_agreement(items)
    cats = {c for r in items for c in r}
    pi = {c: sum(r.count(c) / len(r) for r in items) / len(items) for c in cats}
    pe = sum(v * (1 - v) for v in pi.values()) / (q - 1)
    if pe >= 1:
        return None
    return round((pa - pe) / (1 - pe), 4)


def agreement(rows: List[Dict]) -> List[Dict]:
    """Concordância entre os juízes de cada gerador, por dimensão (todos os juízes que avaliaram)."""
    getters = {"oraculo": "oraculo_juiz", "detecta": "detecta_juiz",
               "redundante": lambda res: "sim" if res.get("redundante_com_juiz") else "nao"}
    by_gen: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r.get("juizes"):
            by_gen[r.get("gerador", "")].append(r)
    out = []
    if len(by_gen) > 1:   # linha agregada com todos os geradores
        by_gen = {**by_gen, "TODOS": [r for rs in by_gen.values() for r in rs]}
    for g, rs in sorted(by_gen.items()):
        for dim, getter in getters.items():
            items = []
            for r in rs:
                vals = [(getter(v) if callable(getter) else v.get(getter))
                        for v in r["juizes"].values() if v.get("juiz_ok")]
                if len(vals) >= 2:
                    items.append(vals)
            if not items:
                continue
            unan = sum(1 for v in items if len(set(v)) == 1) / len(items)
            out.append({"gerador": g, "dimensao": dim, "n_itens": len(items),
                        "n_juizes": Counter(len(v) for v in items).most_common(1)[0][0],
                        "pct_unanimidade": round(unan, 4), "concordancia_bruta": raw_agreement(items),
                        "kappa_fleiss": fleiss_kappa(items), "ac1_gwet": gwet_ac1(items, N_CATS[dim])})
    return out


def matrix_markdown(summary: List[Dict], gens: List[str], judges: List[str]) -> str:
    lines = ["# Tabela de acertos por gerador × juiz",
             "", "Célula: ✓ acertos · ● parciais · ✗ erros (n). `†` = autojulgamento "
             "(juiz da mesma família do gerador). Gabaritos: ver `hits.py`.", ""]
    for dim in DIMS:
        lines += [f"## {dim}", "", "| gerador \\ juiz | " + " | ".join(judges) + " |",
                  "|---|" + "---|" * len(judges)]
        for g in gens:
            cells = []
            for j in judges:
                s = next((x for x in summary if x["gerador"] == g and x["juiz"] == j
                          and x["dimensao"] == dim), None)
                cells.append("—" if not s else
                             f"{s['acertos']}✓ {s['parciais']}● {s['erros']}✗ ({s['n']})"
                             + (" †" if s["autojulgamento"] else ""))
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_all(rows: List[Dict], gens: List[str], judges: List[str], run_dir: Path,
              threshold: float = 0.5, n_boot: int = 2000) -> Dict[str, int]:
    recs = build_records(rows, threshold)
    summ = summarize(recs, n_boot)
    agr = agreement(rows)

    def dump(name: str, data: List[Dict]) -> None:
        if not data:
            (run_dir / name).write_text("", encoding="utf-8")
            return
        with (run_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    dump("hits_long.csv", recs)
    dump("hits_summary.csv", summ)
    dump("agreement.csv", agr)
    (run_dir / "hits_matrix.md").write_text(matrix_markdown(summ, gens, judges), encoding="utf-8")
    return {"registros": len(recs), "celulas": len(summ), "concordancias": len(agr)}
