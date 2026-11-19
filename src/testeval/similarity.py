"""Similaridade entre testes: textual, estrutural e reuso entre alvos diferentes.

Backends: ``tfidf`` (padrão, local, sem rede) ou ``embeddings`` (requer
``sentence-transformers``; mais sensível a semântica).
"""
from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CODE_TOKEN = r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|[^\sA-Za-z_0-9]"
SKEL_TOKEN = r"[A-Za-z]+"


class _Norm(ast.NodeTransformer):
    """Reduz um teste ao seu 'esqueleto': nomes, atributos e literais viram marcadores."""

    def visit_Name(self, n):
        return ast.copy_location(ast.Name(id="N", ctx=n.ctx), n)

    def visit_Attribute(self, n):
        self.generic_visit(n)
        n.attr = "A"
        return n

    def visit_Constant(self, n):
        return ast.copy_location(ast.Constant(value=type(n.value).__name__), n)

    def visit_arg(self, n):
        n.arg, n.annotation = "a", None
        return n

    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        n.name, n.decorator_list = "f", []
        return n


def skeleton(code: str) -> str:
    try:
        tree = _Norm().visit(ast.parse(code))
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return ""


def _cosine_tfidf(texts: List[str], token_pattern: str, ngram=(1, 1)) -> np.ndarray:
    if len(texts) < 2 or not any(t.strip() for t in texts):
        return np.zeros((len(texts), len(texts)))
    vec = TfidfVectorizer(token_pattern=token_pattern, ngram_range=ngram, lowercase=False)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # vocabulário vazio
        return np.zeros((len(texts), len(texts)))
    return (X @ X.T).toarray()  # TF-IDF já vem L2-normalizado => produto = cosseno


def _cosine_embeddings(texts: List[str], model: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # dependência opcional
    emb = SentenceTransformer(model).encode(texts, normalize_embeddings=True)
    return np.asarray(emb) @ np.asarray(emb).T


def similarity_matrix(texts: List[str], backend: str = "tfidf",
                      model: str = "all-MiniLM-L6-v2", structural: bool = False) -> np.ndarray:
    if backend == "embeddings":
        return _cosine_embeddings(texts, model)
    if structural:
        return _cosine_tfidf(texts, SKEL_TOKEN, ngram=(1, 3))
    return _cosine_tfidf(texts, CODE_TOKEN)


def most_similar(ids: List[str], codes: List[str], backend: str = "tfidf",
                 model: str = "all-MiniLM-L6-v2") -> Dict[str, Tuple[float, str]]:
    """Para cada teste: (similaridade máxima com outro teste do mesmo alvo, id desse teste)."""
    out: Dict[str, Tuple[float, str]] = {}
    S = similarity_matrix(codes, backend, model)
    for i, tid in enumerate(ids):
        best, who = 0.0, ""
        for j, other in enumerate(ids):
            if i != j and S[i, j] > best:
                best, who = float(S[i, j]), other
        out[tid] = (round(best, 3), who)
    return out


def cross_target_reuse(items: List[Dict], threshold: float = 0.9) -> Dict[str, List[str]]:
    """Detecta testes com ESTRUTURA parecida em alvos diferentes (candidatos a reuso).

    items: [{'key': 'unit::test', 'target': 'alvo', 'code': '...'}]
    Retorna {key: [keys semelhantes de OUTROS alvos]}.
    """
    keys = [it["key"] for it in items]
    skels = [skeleton(it["code"]) for it in items]
    S = _cosine_tfidf(skels, SKEL_TOKEN, ngram=(1, 3))
    out: Dict[str, List[str]] = {k: [] for k in keys}
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i != j and a["target"] != b["target"] and S[i, j] >= threshold:
                out[keys[i]].append(keys[j])
    return out
