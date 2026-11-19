"""Carregamento da configuração (YAML) com valores padrão."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from .utils import short_hash

DEFAULTS: Dict[str, Any] = {
    "run_name": "run",
    "workdir": "work",
    "project": {
        "repo_path": "work/repo",
        "import_root": ".",
        "source_files": [],
        "min_lines": 3,
        "max_targets": 5,
        "include_private": False,
        "mutation_scope": "callees",   # "callees" = alvo + auxiliares do mesmo arquivo; "function" = só o alvo
        "exclude_names": [],
    },
    "llm": {
        # Forma recomendada: lista única de modelos (label obrigatório) + papéis opcionais.
        "models": [], "generator_labels": None, "judge_labels": None,
        # Forma antiga (compatível): generators + judge/judges
        "generators": [], "judge": {}, "judges": [],
    },
    # self_judging: flag    = o juiz avalia todos e os pares "próprio modelo/família" são marcados
    #               exclude = pares de mesma família são pulados
    #               forbid  = erro se existir algum par de mesma família configurado
    "judging": {"self_judging": "flag"},
    "hits": {"redundant_threshold": 0.5, "bootstrap": 2000},
    "generation": {"samples_per_target": 1},
    "execution": {
        "python": "auto",   # "auto" = sys.executable (evita o atalho python3 da Microsoft Store)
        "timeout_tests": 120,
        "timeout_mutant": 60,
        "max_mutants": 40,
        "mutation_seed": 1,
        "jobs": 1,
    },
    "similarity": {"backend": "tfidf", "embedding_model": "all-MiniLM-L6-v2",
                   "cross_target_threshold": 0.9},
    "judge": {"repeats": 1},
    "scoring": {
        "weights": {"cobertura": 0.20, "deteccao_defeitos": 0.30,
                    "relevancia": 0.20, "correcao": 0.20, "redundancia": 0.10},
        "normalize": "relative_to_best",   # ou "raw"
        "thresholds": {"accept": 0.70, "reject": 0.35, "max_redundancy_accept": 0.80,
                       "max_disagreement_accept": 0.30},
    },
}


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_hash"] = short_hash(p.read_text(encoding="utf-8"))
    base = p.parent
    cfg["_base"] = base
    cfg["workdir"] = (base / cfg["workdir"]).resolve()
    cfg["project"]["repo_path"] = (base / cfg["project"]["repo_path"]).resolve()
    if cfg["execution"].get("python") in (None, "", "auto"):
        cfg["execution"]["python"] = sys.executable
    cfg["run_dir"] = cfg["workdir"] / cfg["run_name"]
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    # caminhos relativos de arquivos "canned" (modo mock) ficam relativos ao YAML
    llm = cfg["llm"]
    every = (list(llm.get("models", [])) + list(llm.get("generators", []))
             + list(llm.get("judges", [])) + [llm.get("judge", {})])
    for c in every:
        if c.get("canned"):
            c["canned"] = str((base / c["canned"]).resolve())
    return cfg


def resolve_models(cfg: Dict[str, Any]):
    """Retorna (geradores, juízes): listas de dicts de modelo (o mesmo modelo pode estar em ambas)."""
    llm = cfg["llm"]
    if llm.get("models"):
        by = {m["label"]: m for m in llm["models"]}
        gl = llm.get("generator_labels") or list(by)
        jl = llm.get("judge_labels") or list(by)
        return [by[x] for x in gl], [by[x] for x in jl]
    judges = list(llm.get("judges", [])) or ([llm["judge"]] if llm.get("judge") else [])
    return list(llm.get("generators", [])), judges
