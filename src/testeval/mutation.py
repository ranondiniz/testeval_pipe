"""Motor de mutação próprio (AST) com matriz de mortes POR TESTE.

Por que não usar mutmut/cosmic-ray diretamente? Eles informam se um mutante
sobreviveu, mas não QUAL teste o matou — informação necessária para medir a
contribuição individual de cada teste e a redundância. Rodamos a suíte inteira
uma vez por mutante (sem ``-x``) e lemos o resultado teste a teste.

Limitações (declare-as na dissertação): mutantes equivalentes não são
detectados; o conjunto de operadores é pequeno (ROR/AOR/LCR/UOD/CRP/RVR/NEG).
Valide uma amostra com mutmut ou cosmic-ray para comparar.
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import copy
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from .execution import ERROR, FAIL_ASSERT, PASS, run_and_parse

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
       ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
       ast.In: ast.NotIn, ast.NotIn: ast.In}
ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
         ast.FloorDiv: ast.Mult, ast.Mod: ast.Mult, ast.Pow: ast.Mult}
SYM = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
       ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
       ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
       ast.Mod: "%", ast.Pow: "**", ast.And: "and", ast.Or: "or"}


@dataclass
class Mutant:
    id: int
    line: int
    op: str
    desc: str
    source: str

    def brief(self) -> Dict:
        return {"id": self.id, "line": self.line, "op": self.op, "desc": self.desc}


class _Replace(ast.NodeTransformer):
    def __init__(self, target: ast.AST, new: ast.AST):
        self.target, self.new = target, new

    def visit(self, node: ast.AST):  # noqa: D401
        if node is self.target:
            return self.new
        return self.generic_visit(node)


def _in(ln: int, ranges) -> bool:
    return any(a <= ln <= b for a, b in ranges)


def _candidates(tree: ast.AST, ranges):
    """Gera (índice_na_árvore, op, descrição, builder(node_copiado)->novo_nó)."""
    for idx, node in enumerate(ast.walk(tree)):
        ln = getattr(node, "lineno", None)
        if ln is None or not _in(ln, ranges):
            continue
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
            old, new = type(node.ops[0]), CMP[type(node.ops[0])]

            def b(n, new=new):
                n.ops = [new()]
                return n
            yield idx, "ROR", f"{SYM[old]} → {SYM[new]}", b
        elif isinstance(node, ast.BinOp) and type(node.op) in ARITH:
            old, new = type(node.op), ARITH[type(node.op)]

            def b(n, new=new):
                n.op = new()
                return n
            yield idx, "AOR", f"{SYM[old]} → {SYM[new]}", b
        elif isinstance(node, ast.BoolOp):
            old = type(node.op)
            new = ast.Or if old is ast.And else ast.And

            def b(n, new=new):
                n.op = new()
                return n
            yield idx, "LCR", f"{SYM[old]} → {SYM[new]}", b
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield idx, "UOD", "not x → x", (lambda n: n.operand)
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield idx, "CRP", f"{node.value} → {not node.value}", \
                (lambda n: ast.Constant(value=not n.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            yield idx, "CRP", f"{node.value} → {node.value + 1}", \
                (lambda n: ast.Constant(value=n.value + 1))
        elif isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            def b(n):
                n.value = ast.Constant(value=None)
                return n
            yield idx, "RVR", "return valor → return None", b
        elif isinstance(node, (ast.If, ast.While)):
            def b(n):
                n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
                return n
            yield idx, "NEG", "condição → not condição", b


def generate_mutants(source: str, start: int, end: int, max_mutants: int = 40,
                     seed: int = 1, ranges=None) -> List[Mutant]:
    """Mutantes nas linhas [start, end] ou, se dado, em ``ranges`` (alvo + auxiliares)."""
    tree = ast.parse(source)
    baseline = ast.unparse(tree)
    cands = list(_candidates(tree, ranges or [(start, end)]))
    if len(cands) > max_mutants:
        rnd = random.Random(seed)
        cands = sorted(rnd.sample(cands, max_mutants), key=lambda c: c[0])
    mutants: List[Mutant] = []
    for idx, op, desc, build in cands:
        t2 = copy.deepcopy(tree)
        node2 = list(ast.walk(t2))[idx]  # mesma ordem de ast.walk => mesmo nó
        line = getattr(node2, "lineno", start)
        new = build(node2)
        if new is not node2:
            t2 = _Replace(node2, new).visit(t2)
        ast.fix_missing_locations(t2)
        try:
            msrc = ast.unparse(t2)
            compile(msrc, "<mutant>", "exec")
        except Exception:  # noqa: BLE001
            continue
        if msrc == baseline:
            continue
        mutants.append(Mutant(id=len(mutants) + 1, line=line, op=op, desc=desc, source=msrc))
    return mutants


def _copy_repo(repo: Path, dest: Path) -> None:
    shutil.copytree(repo, dest, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", ".venv", "venv", "work", ".tox", ".mypy_cache",
        ".pytest_cache", "node_modules", "build", "dist", "*.egg-info"))


def _worker(ws: Path, rel_file: str, mutants: List[Mutant], test_file: Path,
            valid: Set[str], timeout: int, python: str, tag: str):
    path = ws / rel_file
    original = path.read_text(encoding="utf-8")
    results = []
    try:
        for m in mutants:
            path.write_text(m.source, encoding="utf-8")
            out = run_and_parse(test_file, ws, test_file.parent / f"mut_{tag}_{m.id}.xml",
                                timeout, python)
            if out.timeout:
                results.append((m.id, None, True))
                continue
            killed: Set[str] = set()
            if out.collection_error:   # mutante quebrou o import: todos os testes válidos "morrem"
                killed = set(valid)
            else:
                for tid in valid:
                    st = out.statuses.get(tid, (PASS, ""))[0]
                    if st in (FAIL_ASSERT, ERROR):
                        killed.add(tid)
            results.append((m.id, killed, False))
    finally:
        path.write_text(original, encoding="utf-8")
    return results


def run_mutants(repo: Path, rel_file: str, mutants: List[Mutant], test_file: Path,
                valid: Set[str], timeout: int = 60, python: str = "python3",
                jobs: int = 1) -> Dict:
    """Executa a suíte contra cada mutante. Retorna kills por mutante e timeouts."""
    kills: Dict[int, Set[str]] = {}
    timeouts: Set[int] = set()
    if not mutants or not valid:
        return {"kills": kills, "timeouts": timeouts}
    jobs = max(1, min(jobs, len(mutants)))
    chunks = [mutants[i::jobs] for i in range(jobs)]
    with tempfile.TemporaryDirectory(prefix="testeval_mut_") as td:
        wss = []
        for i in range(jobs):
            ws = Path(td) / f"ws{i}"
            _copy_repo(repo, ws)
            wss.append(ws)
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_worker, wss[i], rel_file, chunks[i], test_file, valid,
                                timeout, python, str(i)) for i in range(jobs)]
            for f in futs:
                for mid, killed, to in f.result():
                    if to:
                        timeouts.add(mid)
                    else:
                        kills[mid] = killed or set()
    return {"kills": kills, "timeouts": timeouts}


__all__ = ["Mutant", "generate_mutants", "run_mutants"]
