# TRIBUNE developer targets. Everything here runs offline and free by default.

PY ?= .venv/bin/python

.PHONY: test lint demo eval quant-smoke quant-full quant-real freeze-seed

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check tribune tests

demo:
	$(PY) -m tribune.cli demo

eval:
	$(PY) -m tribune.cli eval

# CI smoke: two quant rungs of the deterministic mock backend over ~5 cases.
quant-smoke:
	$(PY) -m tribune.cli quant-eval --smoke

# Full ladder over the frozen 50-case seed set with the free mock backend.
quant-full:
	$(PY) -m tribune.cli quant-eval

# Full ladder against real served endpoints (llama.cpp / vLLM / SGLang).
# Needs a ladder config listing one endpoint per rung; see docs/quant_sensitivity.md
# for the JSON format and hardware notes. Example:
#   make quant-real CONFIG=my_ladder.json
quant-real:
	$(PY) -m tribune.cli quant-eval --config $(CONFIG)

# Deliberately redefine the frozen quant seed set (invalidates comparability!).
freeze-seed:
	$(PY) -m tribune.cli quant-eval --freeze-seed
