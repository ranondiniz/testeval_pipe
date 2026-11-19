"""Linha de comando: python -m testeval <etapa> --config config.yaml"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from . import compare_models, consensus, hits, runinfo, suite_report
from .config import load_config, resolve_models
from .evaluate import evaluate_unit
from .explain import explain_row, verify_explanation
from .generate import generate_all
from .judge import judge_suite, partial_path, write_validity
from .llm import build_llms, is_self
from .report import COLS, write_csv, write_html, write_jsonl, write_suite_csv
from .scoring import score_rows
from .similarity import cross_target_reuse
from .targets import Target, extract_targets
from .utils import safe_name


def _targets(cfg: Dict, fresh: bool = False) -> List[Target]:
    f: Path = cfg["run_dir"] / "targets.json"
    if f.exists() and not fresh:
        return [Target(**d) for d in json.loads(f.read_text(encoding="utf-8"))]
    p = cfg["project"]
    ts = extract_targets(p["repo_path"], p["import_root"], p["source_files"],
                         p["min_lines"], p["max_targets"], p["include_private"],
                         p.get("exclude_names"), p.get("mutation_scope", "callees"),
                         p.get("targets"))
    f.write_text(json.dumps([t.to_dict() for t in ts], indent=2, ensure_ascii=False), encoding="utf-8")
    return ts


def _llms(cfg: Dict):
    gens, judges, warnings = build_llms(cfg)
    for w in warnings:
        print("AVISO:", w)
    return gens, judges


def stage_targets(cfg: Dict) -> List[Target]:
    ts = _targets(cfg, fresh=True)
    print(f"{len(ts)} alvo(s):")
    for t in ts:
        print(f"  {t.id}  ({t.file}:{t.start}-{t.end})  doc={'sim' if t.doc else 'NÃO (relevância/oráculo menos confiáveis)'}")
        if t.scope:
            print("      + auxiliares no escopo: " + ", ".join(f"{n} ({a}-{b})" for n, a, b in t.scope))
    return ts


def stage_generate(cfg: Dict) -> List[Dict]:
    gens, _ = _llms(cfg)
    runinfo.write(cfg)
    units = generate_all(cfg, _targets(cfg), gens)
    print(f"{len(units)} unidade(s) de teste geradas/reaproveitadas. Ambiente/commit em run_info.json")
    return units


def stage_evaluate(cfg: Dict) -> None:
    ts = {t.id: t for t in _targets(cfg)}
    units = json.loads((cfg["run_dir"] / "units.json").read_text(encoding="utf-8"))
    ed = cfg["run_dir"] / "eval"
    ed.mkdir(exist_ok=True)
    for u in units:
        out = ed / f"{u['unit_id']}.json"
        if out.exists():
            continue
        print(f"avaliando {u['unit_id']} ...")
        res = evaluate_unit(cfg, ts[u["target"]], Path(u["test_file"]), u["unit_id"],
                            Path(u["dir"]), u["generator"])
        res["gerador"] = u["generator"]
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=list), encoding="utf-8")


def select_judges(judges: List, only) -> List:
    """Filtra juízes por label (``--only`` ou ``judge.only``). Sem filtro: todos."""
    if isinstance(only, str):
        only = only.split(",")
    only = [x.strip() for x in (only or []) if x and x.strip()]
    if not only:
        return judges
    unknown = set(only) - {j.label for j in judges}
    if unknown:
        raise SystemExit(f"--only: juiz(es) desconhecido(s): {sorted(unknown)}")
    return [j for j in judges if j.label in only]


def stage_judge(cfg: Dict) -> None:
    gens_m, judges_m = resolve_models(cfg)
    gen_by = {g["label"]: g for g in gens_m}
    _, judges = _llms(cfg)
    judges = select_judges(judges, cfg["judge"].get("only"))
    if cfg["judge"].get("only"):
        print("juízes nesta execução (--only): " + ", ".join(j.label for j in judges))
    pending_total = 0
    ts = {t.id: t for t in _targets(cfg)}
    jd = cfg["run_dir"] / "judge"
    jd.mkdir(exist_ok=True)
    mode = cfg["judging"]["self_judging"]
    for f in sorted((cfg["run_dir"] / "eval").glob("*.json")):
        ev = json.loads(f.read_text(encoding="utf-8"))
        g = gen_by.get(ev.get("gerador", ""), {"label": ev.get("gerador", "")})
        suite = [(r["teste_id"], r["codigo"]) for r in ev["rows"] if r.get("executavel")]
        for j in judges:
            if mode == "exclude" and is_self(g, j.c):
                continue
            out = jd / f"{f.stem}__{safe_name(j.label)}.json"
            if out.exists():
                continue
            tests = [(r["teste_id"], r["codigo"]) for r in ev["rows"] if r.get("executavel")]
            res, pend = judge_suite(j, ts[ev["target"]], tests, suite, cfg["judge"]["repeats"],
                                    partial_path(jd, f.stem, j.label), f.stem, j.label)
            if res is None:
                pending_total += len(pend)
                print(f"PENDENTE {f.stem} por {j.label}: {len(pend)} teste(s) com falha_infra (fica para a retomada)")
                continue
            out.write_text(json.dumps({"unit": f.stem, "judge": j.label, "results": res}, indent=2,
                                      ensure_ascii=False), encoding="utf-8")
            print(f"julgado {f.stem} por {j.label}: {len(res)} teste(s)")
    v = write_validity(cfg["run_dir"])
    print("validade do juiz (judge_validity.csv): " + "; ".join(
        f"{x['juiz']} inválidas antes {x['taxa_invalida_antes']}, final {x['taxa_invalida_final']}, "
        f"falha_infra {x['falha_infra_eventos']} (pendentes {x['pendentes_falha_infra']})" for x in v))
    if pending_total:
        print(f"!!! {pending_total} teste(s) pendentes por falha de infraestrutura; repita o mesmo comando")
        raise SystemExit(3)


def stage_report(cfg: Dict) -> None:
    ts = {t.id: t for t in _targets(cfg)}
    gens_m, judges_m = resolve_models(cfg)
    gen_meta = {g["label"]: g for g in gens_m}
    judge_meta = {j["label"]: j for j in judges_m}
    units, rows = [], []
    jfiles = list((cfg["run_dir"] / "judge").glob("*.json")) if (cfg["run_dir"] / "judge").exists() else []
    judge_results: Dict[str, Dict[str, Dict]] = {}
    for jf in jfiles:
        data = json.loads(jf.read_text(encoding="utf-8"))
        unit = data["unit"]
        bucket = judge_results.setdefault(data["judge"], {})
        for tid, res in data["results"].items():
            bucket[f"{unit}::{tid}"] = res
    for f in sorted((cfg["run_dir"] / "eval").glob("*.json")):
        ev = json.loads(f.read_text(encoding="utf-8"))
        units.append(ev)
        rows.extend(ev["rows"])
    consensus.aggregate(rows, judge_results, judge_meta, gen_meta)
    score_rows(rows, cfg["scoring"])
    reuse = cross_target_reuse(
        [{"key": f"{r['unit_id']}::{r['teste_id']}", "target": r["target"], "code": r["codigo"]}
         for r in rows if r.get("executavel")], cfg["similarity"]["cross_target_threshold"])
    by_unit = {u["unit_id"]: u for u in units}
    for r in rows:
        u = by_unit[r["unit_id"]]
        r["reuso_em"] = reuse.get(f"{r['unit_id']}::{r['teste_id']}", [])
        if not u["facts"]:
            r["explicacao"] = {"resumo": f"Recusar: {u.get('error', '')[:120]}", "evidencias": [], "o_que_falta": []}
            r["explicacao_verificada"] = True
            continue
        expl = explain_row(r, u["facts"], u["suite"], r["reuso_em"])
        ok, problems = verify_explanation(expl, u["facts"], [x["teste_id"] for x in u["rows"]],
                                          ts[r["target"]].doc)
        r["explicacao"], r["explicacao_verificada"] = expl, ok
        if problems:
            r["explicacao_problemas"] = problems
    rd = cfg["run_dir"]
    write_csv(rows, rd / "results.csv")
    write_jsonl(rows, rd / "results.jsonl")
    write_suite_csv(units, rd / "suite_summary.csv")
    write_html(rows, rd / "report.html")
    h = hits.write_all(rows, [g["label"] for g in gens_m], [j["label"] for j in judges_m], rd,
                       cfg["hits"]["redundant_threshold"], cfg["hits"]["bootstrap"])
    print(f"{len(rows)} teste(s) em {rd}: results.csv, results.jsonl, suite_summary.csv, report.html")
    print(f"tabela de acertos: {h['registros']} registros -> hits_matrix.md, hits_summary.csv, "
          f"hits_long.csv, agreement.csv")
    sr = suite_report.write_all(units, rows, [g["label"] for g in gens_m], rd, COLS)
    cm = compare_models.write_all(rd, units)
    print(f"comparação entre modelos: {cm['pares']} par(es), {cm['sobreviventes_comuns']} sobrevivente(s) comuns "
          "-> comparacao/cross_model_*.csv, resumo_por_modelo.csv")
    for w in cm["avisos"]:
        print("AVISO:", w)
    write_validity(rd)
    print(f"suítes por modelo: {sr['geradores']} gerador(es) -> suites/<gerador>/, comparacao/"
          + (f"  AVISO: {sr['explicacoes_com_problema']} explicação(ões) com referência não verificada"
             if sr["explicacoes_com_problema"] else ""))


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="testeval")
    ap.add_argument("stage", choices=["targets", "generate", "evaluate", "judge", "report", "all"])
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--only", default="", help="etapa judge: só estes juízes (labels separados por vírgula)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.only:
        cfg["judge"]["only"] = a.only.split(",")
    stages = {"targets": stage_targets, "generate": stage_generate, "evaluate": stage_evaluate,
              "judge": stage_judge, "report": stage_report}
    if a.stage == "all":
        for s in ("targets", "generate", "evaluate", "judge", "report"):
            stages[s](cfg)
    else:
        stages[a.stage](cfg)


if __name__ == "__main__":
    main()
