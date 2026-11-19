# testeval — avaliação explicável de testes unitários gerados por IA

`testeval` é uma pipeline de pesquisa para **avaliar suítes de testes unitários geradas por modelos de linguagem
(LLMs)** e **explicar** cada avaliação com evidências verificáveis. Ela não para em "a cobertura é 90%": para cada
teste, diz *o que* ele cobre, *quais* defeitos (mutantes) só ele detecta, *com qual* outro teste é redundante e
*se* o seu oráculo é coerente com a especificação. Toda afirmação da explicação é conferida contra os fatos medidos.

O objetivo é apoiar a revisão humana de testes gerados por IA (aceitar, revisar ou recusar) e comparar modelos
geradores de forma transparente.

## Como funciona

```
código-alvo ──► geração (N LLMs) ──► execução isolada ──► métricas objetivas ──► juízes LLM (cegos)
   (AST)          prompt fixo          pytest            cobertura, mutação,      relevância, oráculo,
                  zero-shot                              redundância, reuso       previsões com gabarito
                                                                │                         │
                                                                └──────► consenso + score ◄┘
                                                                              │
                                     explicações verificadas ◄────────────────┤
                                     tabela de acertos (juiz × gabarito) ◄────┤
                                     comparação entre modelos ◄───────────────┘
```

| Etapa | Módulo | O que faz |
|---|---|---|
| `targets` | `targets.py` | Extrai funções e métodos (com docstring) do projeto-alvo via AST, com as funções auxiliares do mesmo arquivo que elas chamam. |
| `generate` | `generate.py`, `llm.py` | Cada modelo gera uma suíte pytest a partir de um prompt único e fixo (zero-shot), com assinatura, docstring e código do alvo. |
| `evaluate` | `execution.py`, `coverage_metrics.py`, `mutation.py`, `redundancy.py`, `similarity.py` | Executa a suíte no código original, mede cobertura de linhas e ramos **por teste**, roda mutação com matriz de mortes **por teste**, calcula a suíte mínima (cobertura de conjuntos gulosa) e a similaridade textual. |
| `judge` | `judge.py` | Juízes LLM avaliam cada teste **sem ver métricas, resultado de execução nem o nome do gerador**. |
| `report` | `consensus.py`, `scoring.py`, `explain.py`, `hits.py`, `suite_report.py`, `compare_models.py`, `report.py` | Consenso dos juízes, score, explicações verificadas, tabela de acertos, concordância, relatório por modelo e comparação entre modelos. |

## Desenho experimental

- **Vários modelos, julgamento cruzado.** Todos os modelos geram; cada suíte é julgada pelos outros modelos.
  O autojulgamento é configurável (`judging.self_judging`): `flag` (marcado † e fora do consenso),
  `exclude` (juiz da mesma família do gerador é pulado) ou `forbid`.
- **Um único prompt zero-shot**, fixo, em `src/testeval/prompts/`. Few-shot e chain-of-thought ficam como
  trabalho futuro.
- **Juiz cego e com temperatura baixa** (`judge_temperature`, padrão 0.0, quando o modelo aceita temperatura).
  A citação da especificação feita pelo juiz é verificada (precisa ser trecho literal da docstring). Resposta
  inválida (JSON quebrado ou campo faltando) é repetida **uma vez** com o mesmo prompt.
- **Previsões com gabarito objetivo.** Além de relevância (1–5) e justificativa, o juiz *prevê* três propriedades
  que a pipeline também *mede*, o que permite uma tabela de acertos por par gerador × juiz:

  | Dimensão | Previsão do juiz | Gabarito medido |
  |---|---|---|
  | oráculo | consistent / inconsistent / unclear | passa no código original → consistent; falha de asserção → inconsistent |
  | detecta | yes / no / unsure | mata ≥ 1 mutante → yes |
  | redundante | id de outro teste ou vazio | ganho marginal 0 na suíte mínima ou redundância ≥ 0,5 |

  `unclear`/`unsure` contam como **parcial**. A taxa de acerto tem IC95% por bootstrap. A concordância entre
  juízes é reportada como concordância bruta, **kappa de Fleiss** e **AC1 de Gwet** (estável quando uma
  categoria domina). Concordância não é correção: juízes LLM compartilham vieses.
- **Seleção de alvos por regra fixa**, registrada antes de ver resultados (`selection.py`): complexidade =
  pontos de decisão da função (if/elif, laços, and/or, ternários, filtros de compreensão); três faixas de
  tamanho igual; dois alvos mais próximos da mediana de cada faixa.
- **Reprodutibilidade.** `run_info.json` guarda o commit do projeto-alvo, ambiente, pacotes, modelos e hash da
  configuração (nunca as chaves). Cada unidade guarda a resposta bruta do modelo, e cada chamada de LLM fica em
  `llm_calls.jsonl` (provedor que serviu, modelo, tokens, truncamento).

## Métricas e score

Cada teste recebe métricas individuais e um score consolidado (pesos padrão em `config.py`, tratados como
hipótese de partida a calibrar):

| Métrica | Peso | Origem |
|---|---|---|
| cobertura (linhas e ramos do alvo) | 0,20 | `coverage_metrics.py` com contexto dinâmico por teste |
| detecção de defeitos (mutantes mortos) | 0,30 | motor de mutação próprio (AST), operadores ROR, AOR, LCR, UOD, CRP, RVR, NEG |
| relevância | 0,20 | juízes (mediana / consenso) |
| correção do oráculo | 0,20 | execução no original + consenso dos juízes |
| não redundância | 0,10 | ganho marginal e sobreposição com a suíte |

A decisão sugerida (`aceitar` / `revisar` / `recusar`) usa limiares configuráveis; oráculo errado veta o teste,
e divergência alta entre juízes impede "aceitar". Teste que falha no código original é marcado como
`possivel_bug_no_codigo` para adjudicação manual (o teste pode estar certo e o código errado).

## Explicabilidade

Para cada teste e para cada suíte, a pipeline gera uma explicação com:

- **resumo** e **evidências**: linhas e mutantes exclusivos, teste equivalente, trecho da especificação citado,
  votos dos juízes;
- **o que falta**: linhas e ramos não cobertos, mutantes sobreviventes.

`verify_explanation` e `verify_suite_explanation` conferem toda referência da explicação contra os fatos
medidos (linhas, mutantes, ids de teste). O `report.html` permite comparar três condições de apresentação para
estudos com usuários (`?cond=A` só código; `B` + métricas e score; `C` + explicação) e exporta as decisões
com o tempo gasto.

## Instalação

Requer Python ≥ 3.11 e git.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest tests -q
python -m testeval all -c config.smoke.yaml
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest tests -q
python -m testeval all -c config.smoke.yaml
```

Os dois últimos comandos executam os testes unitários e o fluxo completo offline. As saídas do teste de
fumaça ficam em `work/smoke/`; comece por `work/smoke/report.html`.

## Uso

```bash
# 1. testes unitários (LLM e executor falsos, sem rede)
python -m pytest tests -q

# 2. teste de fumaça: pipeline completa SEM rede (modelos "mock", projeto de exemplo em examples/toy)
python -m testeval all -c config.smoke.yaml          # saídas em work/smoke/

# 3. projeto-alvo do estudo
git clone --depth 1 https://github.com/alvarofpp/validate-docbr work/validate-docbr

# 4. chaves de API apenas por variável de ambiente (nunca em arquivo)
export OPENROUTER_API_KEY=...          # PowerShell: $env:OPENROUTER_API_KEY = "..."
export GOOGLE_API_KEY=...

# 5. execução real
python -m testeval all -c config.full.yaml
```

Com `make` disponível: `make setup`, `make test`, `make smoke`, `make clone`, `make run`.

**Etapas isoladas e retomáveis:** `python -m testeval <targets|generate|evaluate|judge|report> -c <config>`.
Cada etapa reaproveita o que já foi produzido; repetir o mesmo comando continua de onde parou.

- O juiz grava cada teste julgado assim que a resposta chega (`judge/parciais/*.jsonl`, com a resposta bruta)
  e, ao completar a suíte, o arquivo final `judge/<suíte>__<juiz>.json`.
- Falhas de infraestrutura (429, 5xx, timeout, "high demand") têm espera crescente configurável por modelo
  (`retries`, `backoff_base`, `backoff_max`). Se persistirem, o teste fica como `falha_infra` (não conta como
  resposta inválida), a etapa segue para o próximo e termina com código 3 para indicar que há pendências.
- `--only gpt6_luna,haiku_5_5` roda a etapa `judge` só com os juízes indicados (ou `judge.only` no config).

**Docker:** no Linux/macOS:

```bash
docker build -t testeval .
docker run --rm --network none --memory 2g --cpus 2 \
  -v "$PWD:/app" testeval evaluate -c config.full.yaml
```

No Windows (PowerShell), use `$PWD` ou o caminho absoluto do diretório:

```powershell
docker build -t testeval .
docker run --rm --network none --memory 2g --cpus 2 `
  -v "${PWD}:/app" testeval evaluate -c config.full.yaml
```

`--network none` impede que os testes gerados façam chamadas de rede durante a etapa `evaluate`.

## Configuração

| Arquivo | Uso |
|---|---|
| `config.smoke.yaml` | Teste de fumaça sem rede (3 modelos mock, projeto de exemplo). |
| `config.yaml` | Configuração de referência com os 4 modelos; ponto de partida para novas rodadas. |
| `config.full.yaml` | Protocolo da rodada completa: 6 alvos fixos do validate-docbr, 1 amostra por alvo, autojulgamento excluído, todos os mutantes. |

Cada modelo é declarado em `llm.models` com `label`, `family` (real e distinta: é ela que define o
autojulgamento), `provider: openai_compat`, `base_url`, `api_key_env` e `model`. Qualquer endpoint compatível
com a API de *chat completions* funciona (OpenRouter, Google AI Studio, Ollama, vLLM...). Opções úteis:
`supports_temperature: false` (modelos de raciocínio), `max_tokens`, `rpm` e `rate_group` (limite de taxa
compartilhado), `extra_body` (roteamento do provedor), `timeout`, `retries`, `backoff_max`.

Modelo e identificador usados pela configuração de referência (podem mudar; consulte o provedor antes de uma
rodada nova):

| label | modelo | acesso |
|---|---|---|
| gpt6_luna | `openai/gpt-6-luna` | OpenRouter (pago) |
| haiku_5_5 | `anthropic/claude-haiku-5.5` | OpenRouter (pago) |
| gemma4_31b | `models/gemma-4-31b-it` | Google AI Studio (gratuito) |
| nemotron_ultra | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter (gratuito) |

## Saídas (`work/<run_name>/`)

| Arquivo | Conteúdo |
|---|---|
| `results.csv`, `results.jsonl` | Uma linha por teste: métricas, score, decisão sugerida, consenso dos juízes, explicação. |
| `suite_summary.csv` | Métricas por suíte (alvo × gerador). |
| `report.html` | Relatório navegável e interface do estudo com usuários. |
| `hits_matrix.md`, `hits_summary.csv`, `hits_long.csv` | Tabela de acertos gerador × juiz. |
| `agreement.csv` | Concordância entre juízes (bruta, Fleiss, Gwet AC1). |
| `judge_validity.csv` | Respostas inválidas do juiz antes e depois da repetição; falhas de infraestrutura. |
| `suites/<gerador>/` | Testes, métricas por unidade e por teste, explicação verificada da suíte. |
| `comparacao/` | Comparação entre modelos: Jaccard de linhas, ramos e mutantes mortos; itens exclusivos; mutantes que sobrevivem a todos. |
| `generated/`, `eval/`, `judge/` | Artefatos intermediários (respostas brutas, fatos medidos, vereditos). |
| `run_info.json`, `llm_calls.jsonl` | Proveniência da execução e registro de cada chamada de LLM. |

## Limitações

1. Pressupõe que o **código original está correto**; uma falha de asserção é tratada como oráculo suspeito,
   mas pode ser bug real (por isso a marcação `possivel_bug_no_codigo`).
2. Mutantes equivalentes não são removidos e o conjunto de operadores é pequeno.
3. Auxiliares herdados de outro arquivo (por exemplo, `DocumentBase` no validate-docbr) não são mutados.
4. Testes parametrizados compartilham o mesmo contexto de cobertura.
5. Sem docstring, relevância e oráculo ficam pouco confiáveis.
6. O gabarito de redundância depende da ordem gulosa da suíte mínima (daí o limiar 0,5).
7. Relevância não tem gabarito objetivo; deve ser validada com anotação humana.
8. Juízes LLM medem a si mesmos tanto quanto o gerador; o viés de autojulgamento é mitigado, não eliminado.
9. Um prompt e um projeto-alvo pequeno limitam a generalização dos resultados.

## Licença

MIT — veja [LICENSE](LICENSE). O projeto-alvo [validate-docbr](https://github.com/alvarofpp/validate-docbr) tem
licença própria e não é redistribuído aqui (é clonado em `work/`).
