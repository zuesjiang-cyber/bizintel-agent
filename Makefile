PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
STREAMLIT ?= .venv/bin/streamlit
VERSION ?= v1

.PHONY: help install test lint compile demo ui eval eval-retrieval refresh-demo-corpus fetch-benchmark-sources normalize-benchmark-corpus freeze-benchmark-snapshot resolve-benchmark-evidence build-benchmark-kb run-benchmark check

help:
	@echo "Available targets:"
	@echo "  make install  Install project and dev dependencies into .venv"
	@echo "  make test     Run pytest suite"
	@echo "  make lint     Run ruff checks"
	@echo "  make compile  Run Python compile smoke test"
	@echo "  make demo     Generate offline demo artifacts into artifacts/demo"
	@echo "  make ui       Launch Streamlit UI"
	@echo "  make eval     Run same-corpus BizIntel vs plain-LLM evaluation"
	@echo "  make eval-retrieval  Run retrieval-only benchmark"
	@echo "  make refresh-demo-corpus  Rebuild processed Stripe demo corpus from source pack"
	@echo "  make fetch-benchmark-sources COMPANY=<name>  Download raw benchmark source pack files"
	@echo "  make normalize-benchmark-corpus COMPANY=<name>  Parse raw benchmark docs into normalized and processed corpora"
	@echo "  make freeze-benchmark-snapshot VERSION=<name>  Write hash-level corpus snapshot for a benchmark version"
	@echo "  make resolve-benchmark-evidence VERSION=<name>  Resolve anchor_text entries to real chunk ids"
	@echo "  make build-benchmark-kb VERSION=<name>  Freeze question set, rebuild normalized corpora, and resolve evidence"
	@echo "  make run-benchmark VERSION=<name>  Run the local benchmark suite"
	@echo "  make check    Run lint, tests, and compile smoke test"

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

compile:
	$(PYTHON) -m compileall agent app retrieval verification eval run.py

demo:
	$(PYTHON) run.py "Assess Stripe's revenue quality, valuation drivers, and key monitorables" --demo --artifact-dir artifacts/demo --trace

ui:
	$(STREAMLIT) run app/streamlit_app.py

eval:
	$(PYTHON) -m eval.evaluator

eval-retrieval:
	$(PYTHON) -m eval.retrieval_eval

refresh-demo-corpus:
	$(PYTHON) -m retrieval.ingest --company stripe

fetch-benchmark-sources:
ifndef COMPANY
	$(error Usage: make fetch-benchmark-sources COMPANY=cloudflare)
endif
	$(PYTHON) -m tools.fetch_benchmark_sources --company $(COMPANY) $(if $(DOC_ID),--doc-id $(DOC_ID),) $(ARGS)

normalize-benchmark-corpus:
ifndef COMPANY
	$(error Usage: make normalize-benchmark-corpus COMPANY=cloudflare)
endif
	$(PYTHON) -m tools.normalize_benchmark_corpus --company $(COMPANY) $(ARGS)

freeze-benchmark-snapshot:
	$(PYTHON) -m tools.freeze_benchmark_snapshot --version $(VERSION) --company cloudflare --company fastly $(ARGS)

resolve-benchmark-evidence:
	$(PYTHON) -m tools.resolve_benchmark_evidence --file data/benchmark/$(VERSION)/evidence.jsonl $(ARGS)

build-benchmark-kb:
	$(PYTHON) -m tools.normalize_benchmark_corpus --company cloudflare --company fastly
	$(PYTHON) -m tools.freeze_benchmark_snapshot --version $(VERSION) --company cloudflare --company fastly
	$(PYTHON) -m tools.resolve_benchmark_evidence --file data/benchmark/$(VERSION)/evidence.jsonl

run-benchmark:
	$(PYTHON) -m eval.benchmark_runner --version $(VERSION) $(ARGS)

check: lint test compile
