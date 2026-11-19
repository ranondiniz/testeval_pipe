"""Etapa 1-2: enviar o código ao modelo gerador e salvar os testes retornados."""
from __future__ import annotations

import json
import time
from pathlib import Path
from string import Template
from typing import Dict, List

from .llm import LLM
from .targets import Target
from .utils import extract_code, safe_name, short_hash

PROMPTS = Path(__file__).parent / "prompts"


def build_prompt(t: Target) -> str:
    tpl = Template((PROMPTS / "generate.txt").read_text(encoding="utf-8"))
    return tpl.safe_substitute(
        module=t.module, file=t.file, qualname=t.qualname, signature=t.signature,
        import_name=t.import_name or t.qualname.split('.')[0],
        doc=t.doc or "(no docstring: infer the intended behaviour from the name and signature, and be conservative)",
        context=t.context)


def unit_id(target: Target, gen_label: str, sample: int) -> str:
    return f"{target.id}__{safe_name(gen_label)}__s{sample}"


def generate_all(cfg: Dict, targets: List[Target], generators: List[LLM]) -> List[Dict]:
    run_dir: Path = cfg["run_dir"]
    n = cfg["generation"]["samples_per_target"]
    system = "You write high-quality, deterministic pytest unit tests."
    units: List[Dict] = []
    for t in targets:
        prompt = build_prompt(t)
        for g in generators:
            for k in range(n):
                uid = unit_id(t, g.label, k)
                d = run_dir / "generated" / uid
                d.mkdir(parents=True, exist_ok=True)
                f = d / "test_generated.py"
                if not f.exists():  # permite retomar execuções interrompidas
                    raw = g.chat(system, prompt)
                    f.write_text(extract_code(raw), encoding="utf-8")
                    (d / "raw_response.txt").write_text(raw, encoding="utf-8")
                    (d / "meta.json").write_text(json.dumps({
                        "unit_id": uid, "target": t.id, "generator": g.label,
                        "llm": g.meta(), "sample": k, "prompt_hash": short_hash(prompt),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, indent=2), encoding="utf-8")
                units.append({"unit_id": uid, "target": t.id, "generator": g.label,
                              "sample": k, "dir": str(d), "test_file": str(f)})
    (run_dir / "units.json").write_text(json.dumps(units, indent=2), encoding="utf-8")
    return units
