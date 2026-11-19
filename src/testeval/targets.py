"""Seleção das unidades sob teste (funções/métodos) a partir do repositório."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .utils import safe_name

MAX_CONTEXT_CHARS = 12000


@dataclass
class Target:
    id: str
    file: str          # relativo à raiz do repositório
    module: str        # caminho de import (ex.: validate_docbr.CPF)
    qualname: str      # ex.: CPF.validate
    start: int
    end: int
    source: str        # código da função
    doc: str           # docstring (a "especificação" em texto)
    signature: str
    context: str       # módulo inteiro (se curto) ou classe/função
    import_name: str = ""   # nome a importar nos testes (a classe, no caso de métodos)
    # auxiliares do MESMO arquivo chamados (transitivamente) pelo alvo: [[nome, início, fim], ...]
    # entram no escopo de mutação e cobertura (ex.: CPF._generate_first_digit)
    scope: List[list] = field(default_factory=list)

    def ranges(self) -> List[Tuple[int, int]]:
        """Intervalos de linhas sob avaliação: o alvo + auxiliares chamados."""
        return [(self.start, self.end)] + [(int(a), int(b)) for _, a, b in self.scope]

    def to_dict(self) -> Dict:
        return asdict(self)


def _called_names(fn: ast.AST, cls_name: str | None) -> Set[str]:
    """Nomes chamados como self.x()/cls.x()/Classe.x() (métodos) ou x() (funções do módulo)."""
    out: Set[str] = set()
    owners = {"self", "cls"} | ({cls_name} if cls_name else set())
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in owners:
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def callee_scope(fn: ast.AST, cls: ast.ClassDef | None, tree: ast.Module) -> List[list]:
    """Fecho transitivo dos auxiliares do mesmo arquivo chamados por ``fn``.

    Métodos da mesma classe e funções de nível de módulo. Métodos herdados de outros
    arquivos (ex.: DocumentBase) ficam de fora (limitação declarada).
    """
    methods = {s.name: s for s in (cls.body if cls else [])
               if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
    funcs = {s.name: s for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
    seen: Dict[str, ast.AST] = {}
    todo = [fn]
    while todo:
        cur = todo.pop()
        for name in _called_names(cur, cls.name if cls else None):
            node = methods.get(name) or funcs.get(name)
            if node is None or node is fn or name in seen:
                continue
            seen[name] = node
            todo.append(node)
    return sorted(([k, v.lineno, v.end_lineno] for k, v in seen.items()), key=lambda x: x[1])


def _module_name(rel: str, import_root: str) -> str:
    p = Path(rel)
    if import_root not in ("", "."):
        p = p.relative_to(import_root)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_targets(repo: Path, import_root: str, source_files: List[str],
                    min_lines: int = 3, max_targets: int = 5,
                    include_private: bool = False,
                    exclude_names: List[str] | None = None,
                    mutation_scope: str = "callees",
                    only_ids: List[str] | None = None) -> List[Target]:
    found: List[Target] = []
    exclude = set(exclude_names or [])
    for rel in source_files:
        src = (repo / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        lines = src.splitlines()
        module = _module_name(rel, import_root)
        small_module = len(src) <= MAX_CONTEXT_CHARS

        def add(fn: ast.AST, cls: ast.ClassDef | None) -> None:
            name = fn.name
            if name.startswith("_") and not include_private:
                return
            if name in exclude or (f"{cls.name}.{name}" if cls else name) in exclude:
                return
            n = fn.end_lineno - fn.lineno + 1
            if n < min_lines:
                return
            qual = f"{cls.name}.{name}" if cls else name
            fsrc = "\n".join(lines[fn.lineno - 1: fn.end_lineno])
            if small_module:
                ctx = src
            elif cls is not None:
                ctx = "\n".join(lines[cls.lineno - 1: cls.end_lineno])[:MAX_CONTEXT_CHARS]
            else:
                ctx = fsrc
            ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
            found.append(Target(
                id=safe_name(f"{module}.{qual}"), file=rel, module=module, qualname=qual,
                start=fn.lineno, end=fn.end_lineno, source=fsrc,
                doc=ast.get_docstring(fn) or "",
                signature=f"def {name}({ast.unparse(fn.args)}){ret}", context=ctx,
                import_name=cls.name if cls else name,
                scope=callee_scope(fn, cls, tree) if mutation_scope == "callees" else []))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node, None)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(sub, node)
    if only_ids:   # lista fixa de alvos (project.targets): ignora max_targets
        missing = set(only_ids) - {t.id for t in found}
        if missing:
            raise ValueError(f"project.targets com id(s) inexistente(s) ou não elegível(is): {sorted(missing)}")
        chosen = [t for t in found if t.id in set(only_ids)]
        chosen.sort(key=lambda t: (t.file, t.start))
        return chosen
    # prioriza funções maiores (mais ramos), mas mantém ordem estável de leitura
    found.sort(key=lambda t: -(t.end - t.start))
    chosen = found[:max_targets]
    chosen.sort(key=lambda t: (t.file, t.start))
    return chosen
