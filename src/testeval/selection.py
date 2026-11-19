"""Seleção de alvos por regra fixa (registrada antes de ver resultados).

Regra:
  1. Elegíveis: alvos com docstring, com o mínimo de linhas do config, excluindo
     ``project.exclude_names`` (ex.: generate, generate_list).
  2. Complexidade = nº de pontos de decisão NA PRÓPRIA FUNÇÃO, contados via ast:
     if/elif, for, while, cada operador booleano and/or, cada condição de
     compreensão (``if`` dentro de list/set/dict/generator comprehension) e ternários.
  3. Ordena por (complexidade, id) e divide em três faixas de tamanho igual
     (sobra vai para as primeiras faixas).
  4. Em cada faixa, escolhe os 2 alvos mais próximos da mediana da faixa
     (empate: ordem alfabética do id). Total: 6 (2 simples, 2 médios, 2 complexos).
     Com menos de 6 elegíveis, usa todos e avisa.
"""
from __future__ import annotations

import ast
import statistics
from typing import Dict, List, Tuple

RULE = __doc__.split("Regra:", 1)[1].strip()
FAIXAS = ("simples", "medio", "complexo")


def decision_points(source: str) -> int:
    """Conta pontos de decisão no código de UMA função (texto da função)."""
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp)):
            n += 1
        elif isinstance(node, ast.BoolOp):
            n += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            n += len(node.ifs)
    return n


def split_bands(items: List[Tuple[str, int]], k: int = 3) -> List[List[Tuple[str, int]]]:
    items = sorted(items, key=lambda x: (x[1], x[0]))
    q, r = divmod(len(items), k)
    out, i = [], 0
    for b in range(k):
        size = q + (1 if b < r else 0)
        out.append(items[i:i + size])
        i += size
    return out


def select(items: List[Tuple[str, int]], per_band: int = 2) -> Tuple[List[Dict], List[str]]:
    """items: [(id, complexidade)]. Retorna (escolhidos com faixa e mediana, avisos)."""
    warnings: List[str] = []
    if len(items) < 3 * per_band:
        warnings.append(f"apenas {len(items)} alvo(s) elegível(is) (< {3 * per_band}): usando todos")
        bands = split_bands(items)
        return [{"id": i, "complexidade": c, "faixa": FAIXAS[b], "mediana_faixa": None}
                for b, band in enumerate(bands) for i, c in band], warnings
    chosen: List[Dict] = []
    for b, band in enumerate(split_bands(items)):
        med = statistics.median(c for _, c in band)
        best = sorted(band, key=lambda x: (abs(x[1] - med), x[0]))[:per_band]
        chosen += [{"id": i, "complexidade": c, "faixa": FAIXAS[b], "mediana_faixa": med}
                   for i, c in sorted(best, key=lambda x: (x[1], x[0]))]
    return chosen, warnings
