"""Escopo de mutação/cobertura: alvo + auxiliares do mesmo arquivo chamados transitivamente."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from testeval import mutation  # noqa: E402
from testeval.targets import extract_targets  # noqa: E402

SRC = '''\
from base import Base


def helper(x):
    return x + 1


class Doc(Base):
    def validate(self, doc):
        """Valida."""
        if not self._inherited(doc):
            return False
        return self._digit(doc) == doc[-1]

    def _digit(self, doc):
        total = sum(int(c) for c in doc[:-1])
        return str(self._mod(total))

    def _mod(self, total):
        return helper(total) % 10

    def _unused(self):
        return 1 + 1
'''


def _targets(scope):
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "doc.py").write_text(SRC, encoding="utf-8")
        return extract_targets(Path(td), ".", ["doc.py"], min_lines=3, max_targets=5,
                               mutation_scope=scope)


def test_callees_are_transitive_and_same_file_only():
    t = next(x for x in _targets("callees") if x.qualname == "Doc.validate")
    names = [n for n, _, _ in t.scope]
    assert names == ["helper", "_digit", "_mod"]          # transitivo; _unused e _inherited fora
    assert t.ranges()[0] == (t.start, t.end) and len(t.ranges()) == 4


def test_function_scope_and_mutants_in_helpers():
    t0 = next(x for x in _targets("function") if x.qualname == "Doc.validate")
    assert t0.scope == []
    t = next(x for x in _targets("callees") if x.qualname == "Doc.validate")
    only = mutation.generate_mutants(SRC, t.start, t.end, 10**6)
    wide = mutation.generate_mutants(SRC, t.start, t.end, 10**6, ranges=t.ranges())
    assert len(wide) > len(only)
    assert any(m.line > t.end for m in wide) and all(t.start <= m.line <= t.end for m in only)
    unused = next(a for n, a, _ in [("_unused", i + 1, 0) for i, l in enumerate(SRC.splitlines())
                                   if "def _unused" in l])
    assert all(m.line < unused for m in wide)              # método não chamado não é mutado


if __name__ == "__main__":
    test_callees_are_transitive_and_same_file_only()
    test_function_scope_and_mutants_in_helpers()
    print("ok")
