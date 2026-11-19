PY ?= python
export PYTHONPATH := src

setup:           ## instala dependências
	$(PY) -m pip install -r requirements.txt

test:            ## testes unitários (LLM e executor falsos, sem rede)
	$(PY) -m pytest tests -q -p no:cacheprovider

smoke:           ## pipeline completa SEM rede (3 modelos mock) — rode isto primeiro
	$(PY) -m testeval all -c config.smoke.yaml
	@echo "Veja work/smoke/: report.html, results.csv, hits_matrix.md, agreement.csv"

clone:           ## clona o alvo do piloto
	mkdir -p work && git clone --depth 1 https://github.com/alvarofpp/validate-docbr work/validate-docbr

run:             ## pipeline real (configure config.yaml e as chaves)
	$(PY) -m testeval all -c config.yaml

pilot:           ## piloto real via OpenRouter (exige OPENROUTER_API_KEY e `make clone`)
	$(PY) -m testeval all -c config.yaml
