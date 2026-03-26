# Benchmark Collection Checklist

This checklist is for building a credible local benchmark pack around:

- `cloudflare`
- `fastly`

The goal is not to collect "a lot of documents". The goal is to collect a stable, memo-grade source pack that can support:

1. retrieval benchmarking
2. memo generation benchmarking
3. same-corpus BizIntel vs plain-LLM comparison

Without downloading the raw files locally, you do not have a reproducible benchmark corpus.

## Canonical Layout

```text
data/
  raw/
    <company>/
      manifest.json
      periods.json
      docs/
  normalized/
    <company>/
      documents.jsonl
      chunks.jsonl
  benchmark/
    v1/
      items.jsonl
      answers.jsonl
      evidence.jsonl
      scores.jsonl
```

## Required Source Types

Keep the `source_type` enum fixed:

- `ir_overview`
- `annual_report`
- `quarterly_results`
- `earnings_call_transcript`
- `investor_presentation`
- `investor_supplement`
- `quarterly_report`
- `proxy_statement`
- `event_page`

## Cloudflare Raw Files

Required:

- `data/raw/cloudflare/docs/cloudflare_ir_overview.html`
- `data/raw/cloudflare/docs/cloudflare_2024_10k.html`
- `data/raw/cloudflare/docs/cloudflare_q4_2025_results.html`
- `data/raw/cloudflare/docs/cloudflare_q4_2025_event_page.html`
- `data/raw/cloudflare/docs/cloudflare_q4_2025_transcript.pdf`
- `data/raw/cloudflare/docs/cloudflare_q4_2025_results_exhibit_99_1.pdf`
- `data/raw/cloudflare/docs/cloudflare_q3_2025_results.html`
- `data/raw/cloudflare/docs/cloudflare_q3_2025_transcript.pdf`
- `data/raw/cloudflare/docs/cloudflare_q2_2025_results.html`
- `data/raw/cloudflare/docs/cloudflare_q2_2025_transcript.pdf`

Strongly recommended:

- `data/raw/cloudflare/docs/cloudflare_2025_proxy.html`
- `data/raw/cloudflare/docs/cloudflare_2025_investor_day_presentation.pdf`

## Fastly Raw Files

Required:

- `data/raw/fastly/docs/fastly_ir_overview.html`
- `data/raw/fastly/docs/fastly_2025_10k.html`
- `data/raw/fastly/docs/fastly_2025_q3_10q.html`
- `data/raw/fastly/docs/fastly_q4_2025_results.html`
- `data/raw/fastly/docs/fastly_q4_2025_event_page.html`
- `data/raw/fastly/docs/fastly_q4_2025_investor_supplement.pdf`
- `data/raw/fastly/docs/fastly_q4_2025_investor_presentation.pdf`
- `data/raw/fastly/docs/fastly_q4_2025_transcript.pdf`
- `data/raw/fastly/docs/fastly_q3_2025_results.html`
- `data/raw/fastly/docs/fastly_q2_2025_transcript.pdf`

Strongly recommended:

- `data/raw/fastly/docs/fastly_q2_2025_results.html`
- `data/raw/fastly/docs/fastly_q2_2025_investor_supplement.pdf`
- `data/raw/fastly/docs/fastly_q3_2025_transcript.pdf`

## Required Metadata Files

Per company:

- `data/raw/<company>/manifest.json`
- `data/raw/<company>/periods.json`
- `data/normalized/<company>/documents.jsonl`
- `data/normalized/<company>/chunks.jsonl`

Global benchmark files:

- `data/benchmark/v1/items.jsonl`
- `data/benchmark/v1/answers.jsonl`
- `data/benchmark/v1/evidence.jsonl`
- `data/benchmark/v1/scores.jsonl`

## Practical Execution Order

1. Download raw files into `data/raw/<company>/docs/`
2. Fill `manifest.json`
3. Fill `periods.json`
4. Convert raw files into normalized markdown or text if needed
5. Generate `documents.jsonl`
6. Generate `chunks.jsonl`
7. Write `items.jsonl`
8. Write `answers.jsonl`
9. Write `evidence.jsonl`
10. Run retrieval / generation / verification evaluation

## Quality Rules

- Prefer primary IR / SEC materials over blogs.
- Preserve the original raw file in `data/raw/`.
- Track period and date explicitly.
- Keep source naming stable so benchmark IDs stay reproducible.
- Do not mix quarterly press releases and transcripts under one document ID.
- Do not invent missing investor decks. Cloudflare Q4 2025 uses `results_exhibit_99_1.pdf`, not an unverified Q4 investor presentation.
