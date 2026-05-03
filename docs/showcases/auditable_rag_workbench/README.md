# FinTrust RAG Workbench Showcase

This showcase demonstrates the local audit layer for FinTrust RAG: benchmark forensics, standards-backed trust gates, ledger export, and document fidelity checks.

## Artifacts

- [benchmark_forensics_report.md](benchmark_forensics_report.md)
- [benchmark_ledger.csv](benchmark_ledger.csv)
- [document_fidelity_report.md](document_fidelity_report.md)

## Benchmark Forensics Snapshot

Candidate run:

- `eval/results/financebench_open150_all_live_20260410_parallel_merged.json`
- Rows: 150
- Answered: 39
- Partial: 100
- Abstained: 11

Trust metrics:

- `unsupported_claim_rate`: 0.0419
- `unsupported_numeric_claim_rate`: 0.0435
- `fabricated_citation_rate`: 0.0000
- `wrong_entity_rate`: 0.0000
- `wrong_period_rate`: 0.0000
- `numeric_exact_match_rate`: 0.9197
- `primary_source_claim_coverage`: 0.8809
- `required_fact_recall`: 0.1944
- `hard_fact_complete_but_not_published_rate`: 0.5800

## What This Proves

The project is no longer presenting a single polished memo as the proof point. The audit layer shows:

- which trust gates passed or failed;
- which failure tags dominate the benchmark cohort;
- whether baseline/candidate changes improved safety or completeness;
- which evidence and artifact paths back each benchmark row;
- whether source-pack metadata is good enough for financial RAG.

The core story is: financial RAG quality is judged through retrieval, grounding, finance hard gates, and publication diagnostics, not through model fluency alone.
