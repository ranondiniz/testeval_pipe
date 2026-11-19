"""Execução dos testes gerados (pytest em subprocesso) e leitura do JUnit XML.

``run_and_parse`` é o ÚNICO ponto de contato com pytest/coverage; os demais
módulos dependem apenas do objeto ``Outcome``.
"""
from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .utils import TEST_MODULE, norm_ctx

PASS, FAIL_ASSERT, ERROR, SKIP = "pass", "fail_assert", "error", "skip"
_PRIORITY = {ERROR: 3, FAIL_ASSERT: 2, SKIP: 1, PASS: 0}


@dataclass
class Outcome:
    statuses: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # id -> (status, msg)
    collection_error: bool = False
    timeout: bool = False
    message: str = ""
    returncode: Optional[int] = None
    cov_data_file: Optional[Path] = None
    cov_rcfile: Optional[Path] = None


def parse_junit(xml_path: Path) -> Dict[str, Tuple[str, str]]:
    """Converte JUnit XML em {test_id: (status, mensagem)}; testes parametrizados são agregados."""
    if not xml_path.exists():
        return {}
    root = ET.parse(xml_path).getroot()
    out: Dict[str, Tuple[str, str]] = {}
    for tc in root.iter("testcase"):
        tid = norm_ctx(f'{tc.get("classname", "")}.{tc.get("name", "")}')
        if not tid:
            continue
        status, msg = PASS, ""
        for child in tc:
            m = (child.get("message") or child.text or "").strip()
            if child.tag == "failure":
                is_assert = m.lower().startswith(("assert", "assertionerror"))
                status, msg = (FAIL_ASSERT if is_assert else ERROR), m[:300]
            elif child.tag == "error":
                status, msg = ERROR, m[:300]
            elif child.tag == "skipped":
                status, msg = SKIP, m[:300]
        prev = out.get(tid)
        if prev is None or _PRIORITY[status] > _PRIORITY[prev[0]]:
            out[tid] = (status, msg)
    return out


def _env(workspace: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # evita .pyc obsoleto ao trocar o arquivo mutado
    return env


def run_and_parse(test_file: Path, workspace: Path, junit_xml: Path, timeout: int,
                  python: str = "python3", cov_target: Optional[Path] = None) -> Outcome:
    """Roda o arquivo de teste em ``workspace``. Se ``cov_target`` for dado, mede cobertura
    de ramos com contexto dinâmico por teste (``dynamic_context = test_function``)."""
    junit_xml.parent.mkdir(parents=True, exist_ok=True)
    if junit_xml.exists():
        junit_xml.unlink()
    out = Outcome()
    cmd = [python]
    if cov_target is not None:
        rc = junit_xml.parent / "cov.rc"
        data = junit_xml.parent / ".coverage"
        if data.exists():
            data.unlink()
        rc.write_text(
            "[run]\nbranch = True\ndynamic_context = test_function\n"
            f"data_file = {data}\ninclude = {cov_target.resolve()}\n", encoding="utf-8")
        cmd += ["-m", "coverage", "run", f"--rcfile={rc}"]
        out.cov_data_file, out.cov_rcfile = data, rc
    cmd += ["-m", "pytest", str(test_file), "-q", "-p", "no:cacheprovider", "--tb=short",
            f"--junitxml={junit_xml}", f"--rootdir={test_file.parent}"]
    try:
        p = subprocess.run(cmd, cwd=workspace, env=_env(workspace), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        out.timeout = True
        out.message = "timeout"
        return out
    out.returncode = p.returncode
    out.statuses = parse_junit(junit_xml)
    # rc 0 = tudo passou; 1 = alguma falha; demais = erro de coleta/uso
    if p.returncode not in (0, 1) or not out.statuses:
        out.collection_error = True
        out.message = (p.stdout + "\n" + p.stderr).strip()[-600:]
    return out


__all__ = ["Outcome", "parse_junit", "run_and_parse", "PASS", "FAIL_ASSERT", "ERROR", "SKIP", "TEST_MODULE"]
