"""Explicações ancoradas em evidência (XAI de artefato de teste).

Princípio: TODA afirmação da explicação aponta para um fato rastreável
(linha, ramo, mutante, teste similar, trecho da especificação). O verificador
confere cada referência contra os fatos medidos; a única parte originada de LLM
(trecho da especificação) é checada como substring da docstring.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def _fmt_lines(lines: List[int], limit: int = 6) -> str:
    s = ", ".join(f"L{l}" for l in lines[:limit])
    return s + (f" (+{len(lines) - limit})" if len(lines) > limit else "")


def _mut_txt(facts: Dict, ids: List[int], limit: int = 3) -> str:
    by_id = {m["id"]: m for m in facts["mutants"]}
    return "; ".join(f"#{i} (L{by_id[i]['line']}: {by_id[i]['desc']})" for i in ids[:limit] if i in by_id)


def explain_row(row: Dict, facts: Dict, suite: Dict, reuse: List[str]) -> Dict:
    """Monta a explicação de um teste. Retorna {resumo, evidencias[], o_que_falta[]}."""
    ev: List[Dict] = []
    miss: List[Dict] = []
    tid = row["teste_id"]
    st = row.get("status")

    if not row.get("executavel"):
        ev.append({"tipo": "execucao", "texto": f"O teste não executa corretamente ({row.get('msg', '')[:160]}).",
                   "refs": {}})
    elif st == "fail_assert":
        txt = "A asserção falha no código original."
        if row.get("possivel_bug_no_codigo"):
            txt += " O juiz considera o oráculo coerente com a especificação: pode haver BUG no código — revise antes de descartar."
        else:
            txt += " Isso indica oráculo incorreto (valor esperado errado)."
        ev.append({"tipo": "correcao", "texto": txt, "refs": {}})
    else:
        if row.get("oraculo_juiz") == "inconsistent":
            ev.append({"tipo": "correcao", "texto": "O teste passa, mas o juiz aponta que o valor esperado contradiz a "
                       "especificação (pode estar fixando comportamento bugado).", "refs": {}})
        ul = facts["unique_lines"].get(tid, [])
        cob = f"Cobre {row['linhas_cobertas']} linha(s) e {row['ramos_cobertos']} ramo(s) do alvo (função + auxiliares chamados)"
        cob += f"; exclusivas deste teste: {_fmt_lines(ul)}." if ul else "; nenhuma linha exclusiva."
        ev.append({"tipo": "cobertura", "texto": cob, "refs": {"linhas": ul}})
        km = facts["kills"].get(tid, [])
        uk = facts["unique_kills"].get(tid, [])
        det = f"Mata {row['mutantes_mortos']} de {len(facts['mutants'])} mutantes ({row['deteccao_defeitos']:.0%})"
        det += f"; exclusivos: {_mut_txt(facts, uk)}." if uk else "; nenhum mutante exclusivo."
        ev.append({"tipo": "deteccao", "texto": det, "refs": {"mutantes": uk}})
        if row.get("ganho_marginal", 0) == 0:
            sim = row.get("teste_mais_similar") or "outros testes"
            ev.append({"tipo": "redundancia", "texto": "Redundante: tudo o que cobre e detecta já é coberto pelos "
                       f"demais testes (mais parecido textualmente: {sim}).",
                       "refs": {"testes": [row["teste_mais_similar"]] if row.get("teste_mais_similar") else []}})
        elif (row.get("redundancia") or 0) >= 0.5:
            ev.append({"tipo": "redundancia", "texto": f"Parcialmente redundante ({row['redundancia']:.0%} da pegada já coberta).",
                       "refs": {"testes": []}})
        if row.get("juiz_ok"):
            q = row.get("trecho_spec", "")
            note = f' Trecho da especificação: "{q}".' if q and row.get("trecho_verificado") else (
                " (citação da especificação NÃO verificada — ignore)" if q else "")
            ev.append({"tipo": "relevancia",
                       "texto": f"Relevância {row.get('relevancia_juiz', 0):.0%} — {row.get('justificativa_relevancia', '')}{note}",
                       "refs": {"trecho": q if row.get("trecho_verificado") else ""}})
    juizes = row.get("juizes") or {}
    if len(juizes) > 1:
        det = ", ".join(f"{k}{' (autojulgamento)' if v.get('autojulgamento') else ''}={v.get('oraculo_juiz', '?')}"
                        for k, v in juizes.items() if v.get("juiz_ok"))
        disc = row.get("discordancia") or 0
        txt = f"{len(juizes)} juízes avaliaram o oráculo: {det}."
        if disc > 0:
            txt += f" Há divergência ({disc:.0%}): recomenda-se revisão humana."
        ev.append({"tipo": "juizes", "texto": txt, "refs": {}})
    if reuse:
        ev.append({"tipo": "reuso", "texto": f"Estrutura semelhante a testes de outros alvos ({', '.join(reuse[:3])}): "
                   "candidato a parametrização/reuso.", "refs": {"chaves": reuse[:3]}})

    # o que falta (nível da suíte, igual para todos os testes da unidade)
    ul_all = facts.get("suite_uncovered_lines", [])
    if ul_all:
        miss.append({"tipo": "linhas_nao_cobertas", "texto": f"Nenhum teste cobre {_fmt_lines(ul_all)}.",
                     "refs": {"linhas": ul_all}})
    ub = facts.get("suite_uncovered_branches", [])
    if ub:
        miss.append({"tipo": "ramos_nao_cobertos", "texto": "Ramos não cobertos: " +
                     ", ".join(f"L{a}→L{b}" for a, b in ub[:4]) + ".", "refs": {"ramos": ub}})
    sv = facts.get("survivors", [])
    if sv:
        miss.append({"tipo": "mutantes_sobreviventes",
                     "texto": f"Mutantes que nenhum teste detecta: {_mut_txt(facts, sv)}. "
                              "Sugestão: adicionar um caso cujo resultado mude com essa alteração.",
                     "refs": {"mutantes": sv[:3]}})

    razoes = {"execucao": "não executa", "correcao": "oráculo suspeito/incorreto",
              "redundancia": "redundante"}
    main = [razoes[e["tipo"]] for e in ev if e["tipo"] in razoes]
    dec = row.get("decisao_sugerida", "revisar")
    resumo = f"{dec.capitalize()} (score {row.get('score', 0):.2f})" + (": " + "; ".join(main) if main else "")
    return {"resumo": resumo, "evidencias": ev, "o_que_falta": miss}


def verify_explanation(expl: Dict, facts: Dict, all_test_ids: List[str], doc: str) -> Tuple[bool, List[str]]:
    """Confere cada referência contra os fatos medidos. Retorna (ok, problemas)."""
    problems: List[str] = []
    lines_u = set(facts.get("stmts", []))
    mut_u = {m["id"] for m in facts.get("mutants", [])}
    br_u = {tuple(b) for b in facts.get("branches", [])}
    for item in expl["evidencias"] + expl["o_que_falta"]:
        r = item.get("refs", {})
        for l in r.get("linhas", []):
            if l not in lines_u:
                problems.append(f"linha inexistente: {l}")
        for m in r.get("mutantes", []):
            if m not in mut_u:
                problems.append(f"mutante inexistente: {m}")
        for b in r.get("ramos", []):
            if tuple(b) not in br_u:
                problems.append(f"ramo inexistente: {b}")
        for t in r.get("testes", []):
            if t and t not in all_test_ids:
                problems.append(f"teste inexistente: {t}")
        q = r.get("trecho", "")
        if q and q.lower() not in (doc or "").lower():
            problems.append("trecho da especificação não encontrado na docstring")
    return (not problems), problems
