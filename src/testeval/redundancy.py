"""Redundância e suíte mínima por cobertura gulosa (greedy set cover).

Cada teste tem uma "pegada" (footprint): linhas + ramos cobertos + mutantes
mortos. Em cada passo escolhemos o teste com maior ganho marginal. Um teste cujo
ganho é zero não acrescenta nada ao que os anteriores já cobrem/detectam.

    redundancia = 1 - ganho_marginal / tamanho_da_pegada    (0 = útil, 1 = redundante)
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Set, Tuple


def greedy_order(foot: Dict[str, Set[Hashable]]) -> List[Tuple[str, int]]:
    remaining = {k: set(v) for k, v in foot.items()}
    covered: Set[Hashable] = set()
    order: List[Tuple[str, int]] = []
    while remaining:
        best = max(remaining, key=lambda t: (len(remaining[t] - covered), len(remaining[t]), t))
        gain = len(remaining[best] - covered)
        order.append((best, gain))
        covered |= remaining.pop(best)
    return order


def redundancy(foot: Dict[str, Set[Hashable]]) -> Dict[str, Dict]:
    """Retorna {id: {'redundancia': float, 'ganho': int, 'na_suite_minima': bool}}."""
    out: Dict[str, Dict] = {}
    for tid, gain in greedy_order(foot):
        size = len(foot[tid])
        red = 1.0 if size == 0 else 1.0 - gain / size
        out[tid] = {"redundancia": round(red, 4), "ganho": gain, "na_suite_minima": gain > 0}
    return out
