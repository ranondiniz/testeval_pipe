"""Saídas: results.csv (colunas + score), results.jsonl (com explicações),
suite_summary.csv e report.html (planilha de revisão humana)."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List

COLS = ["unit_id", "target", "gerador", "teste_id", "executavel",
        "cobertura", "redundancia", "relevancia", "deteccao_defeitos", "correcao", "score",
        "decisao_sugerida", "na_suite_minima", "ganho_marginal", "sim_textual_max",
        "teste_mais_similar", "reuso_em", "possivel_bug_no_codigo", "oraculo_juiz",
        "n_juizes", "discordancia", "rel_amplitude", "consenso_base", "explicacao_verificada"]


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["reuso_em"] = ";".join(r.get("reuso_em") or [])
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})


def write_suite_csv(units: List[Dict], path: Path) -> None:
    cols = ["unit_id", "target", "gerador", "n_tests", "n_validos", "cobertura_suite",
            "mutation_score_suite", "n_suite_minima", "mutantes", "mutantes_timeout", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for u in units:
            w.writerow({**u["suite"], "unit_id": u["unit_id"], "target": u["target"],
                        "gerador": u.get("gerador", ""), "error": (u.get("error") or "")[:120]})


def write_jsonl(rows: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def write_html(rows: List[Dict], path: Path, title: str = "Revisão de testes gerados por IA") -> None:
    e = html.escape
    body: List[str] = []
    for i, r in enumerate(rows):
        x = r.get("explicacao") or {}
        ev = "".join(f"<li class='{e(i_['tipo'])}'>{e(i_['texto'])}</li>" for i_ in x.get("evidencias", []))
        mi = "".join(f"<li>{e(i_['texto'])}</li>" for i_ in x.get("o_que_falta", []))
        warn = "" if r.get("explicacao_verificada", True) else "<p class='warn'>⚠ explicação com referência não verificada</p>"
        body.append(
            f"<tr data-id='{e(r['unit_id'])}::{e(r['teste_id'])}'>"
            f"<td><small>{e(r['target'])}<br>{e(r.get('gerador', ''))}</small></td>"
            f"<td><b>{e(r['teste_id'])}</b><pre>{e(r.get('codigo', ''))}</pre></td>"
            f"<td class='m'>{_pct(r.get('cobertura'))}</td><td class='m'>{_pct(r.get('redundancia'))}</td>"
            f"<td class='m'>{_pct(r.get('relevancia'))}</td><td class='m'>{_pct(r.get('deteccao_defeitos'))}</td>"
            f"<td class='m'>{_pct(r.get('correcao'))}</td>"
            f"<td class='s'><b>{r.get('score', 0):.2f}</b><br><small>{e(r.get('decisao_sugerida', ''))}</small></td>"
            f"<td class='x'><div><b>{e(x.get('resumo', ''))}</b></div>{warn}<ul>{ev}</ul>"
            f"<details><summary>O que falta</summary><ul>{mi}</ul></details></td>"
            f"<td><select data-row='{i}'><option value=''>—</option><option>aceitar</option>"
            f"<option>editar</option><option>recusar</option></select></td></tr>")
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>{e(title)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:1rem}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #bbb;padding:.4rem;vertical-align:top}} pre{{white-space:pre-wrap;margin:.3rem 0;font-size:12px}}
.m,.s{{text-align:center;white-space:nowrap}} .warn{{color:#a00}} li.correcao,li.execucao{{color:#a00}}
.hideM .m,.hideM .hm{{display:none}} .hideS .s,.hideS .hs{{display:none}} .hideX .x,.hideX .hx{{display:none}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}td,th{{border-color:#555}}}}
</style>
<h2>{e(title)}</h2>
<p>Condição: <b id="cond"></b> &nbsp; <button id="exp">Exportar decisões (CSV)</button>
<small> (use ?cond=A só código · B código + métricas/score · C tudo)</small></p>
<table id="t"><thead><tr><th>Alvo / gerador</th><th>Teste</th>
<th class="hm">Cobertura</th><th class="hm">Redundância</th><th class="hm">Relevância</th>
<th class="hm">Detecção</th><th class="hm">Correção</th><th class="hs">Score</th>
<th class="hx">Explicação</th><th>Decisão</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<script>
const q=new URLSearchParams(location.search);const cond=(q.get('cond')||'C').toUpperCase();
document.getElementById('cond').textContent=cond;
if(cond==='A')document.body.classList.add('hideM','hideS','hideX');
if(cond==='B')document.body.classList.add('hideX');
const log={{}};let last=performance.now();
document.querySelectorAll('select[data-row]').forEach(s=>s.addEventListener('change',ev=>{{
  const tr=ev.target.closest('tr');const now=performance.now();
  log[tr.dataset.id]={{decisao:ev.target.value,ms:Math.round(now-last),ts:new Date().toISOString()}};last=now;}}));
document.getElementById('exp').onclick=()=>{{
  const lines=['id,decisao,ms_desde_ultima_decisao,timestamp,condicao'];
  for(const [k,v] of Object.entries(log))lines.push([JSON.stringify(k),v.decisao,v.ms,v.ts,cond].join(','));
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{{type:'text/csv'}}));
  a.download='decisoes_'+cond+'.csv';a.click();}};
</script></html>"""
    path.write_text(page, encoding="utf-8")
