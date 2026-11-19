"""Registro de reprodutibilidade: commit do alvo, ambiente, pacotes e modelos (sem chaves)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from .config import resolve_models


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "não instalado"


def collect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    repo: Path = cfg["project"]["repo_path"]
    gens, judges = resolve_models(cfg)
    pick = ("label", "family", "provider", "model", "temperature", "supports_temperature", "max_tokens", "base_url", "extra_body", "rpm")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_hash": cfg.get("_config_hash"),
        "repo": {"path": str(repo), "commit": _git(repo, "rev-parse", "HEAD") or "desconhecido",
                 "remote": _git(repo, "remote", "get-url", "origin"),
                 "sujo": bool(_git(repo, "status", "--porcelain"))},
        "ambiente": {"python": sys.version.split()[0], "so": platform.platform(),
                     "pacotes": {p: _ver(p) for p in ("pytest", "coverage", "PyYAML", "requests",
                                                      "scikit-learn", "numpy")}},
        "geradores": [{k: m.get(k) for k in pick} for m in gens],
        "juizes": [{k: m.get(k) for k in pick} for m in judges],
        "self_judging": cfg["judging"]["self_judging"],
        "execucao": cfg["execution"], "geracao": cfg["generation"],
    }


def write(cfg: Dict[str, Any]) -> Path:
    path = cfg["run_dir"] / "run_info.json"
    path.write_text(json.dumps(collect(cfg), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
