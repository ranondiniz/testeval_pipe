"""Funções utilitárias compartilhadas (sem dependências externas)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from typing import Dict, List, Optional

TEST_MODULE = "test_generated"  # nome fixo do módulo de teste gerado


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")


def norm_ctx(name: str) -> str:
    """Normaliza identificadores de teste vindos do coverage.py e do JUnit XML.

    'test_generated.TestX.test_y[p1]' -> 'TestX.test_y'
    'test_generated.test_foo'         -> 'test_foo'
    """
    name = re.sub(r"\[.*\]$", "", name)
    parts = name.split(".")
    if TEST_MODULE in parts:
        parts = parts[parts.index(TEST_MODULE) + 1:]
    if not parts:
        return ""
    if len(parts) >= 2 and parts[-2].startswith("Test"):
        return ".".join(parts[-2:])
    return parts[-1]


def list_tests(source: str) -> List[Dict]:
    """Lista funções/métodos de teste de um módulo pytest (levanta SyntaxError se inválido)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out: List[Dict] = []

    def add(tid: str, node: ast.AST) -> None:
        seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
        out.append({"id": tid, "code": textwrap.dedent(seg), "lineno": node.lineno})

    fn = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, fn) and node.name.startswith("test"):
            add(node.name, node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, fn) and sub.name.startswith("test"):
                    add(f"{node.name}.{sub.name}", sub)
    return out


def extract_code(text: str) -> str:
    """Extrai o maior bloco ```python da resposta do modelo (ou o texto inteiro)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
