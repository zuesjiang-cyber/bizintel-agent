# Corpus Audit

This audit records what the current source packs are good enough to support before running the `v2` benchmark.

## Cloudflare

Strong coverage:

- `company_overview`
- `time_sensitive`
- `source_priority`
- `risk_catalyst`

Why:

- [2024 10-K](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_2024_10k.html)
- [Q3 2025 transcript](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_q3_2025_transcript.pdf)
- [Q4 2025 transcript](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_q4_2025_transcript.pdf)
- [Q4 2025 results](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_q4_2025_results.html)
- [Q4 2025 Exhibit 99.1](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_q4_2025_results_exhibit_99_1.pdf)
- [Investor Day presentation](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs/cloudflare_2025_investor_day_presentation.pdf)

Weak coverage:

- period-matched investor-presentation questions for Q3/Q4
- hard filing-to-filing quarterly-report questions

Note:

- `company_overview` is strong only when the item allows `2024FY + 2025Q4 + strategy_context` style evidence pooling.

## Fastly

Strong coverage:

- `company_overview`
- `numeric_grounding`
- `time_sensitive`
- `evidence_conflict`
- `risk_catalyst`

Why:

- [2025 10-K](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs/fastly_2025_10k.html)
- [Q2/Q3/Q4 results](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs/fastly_q2_2025_results.html)
- [Q2/Q3/Q4 supplements](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs/fastly_q2_2025_investor_supplement.pdf)
- [Q2/Q3/Q4 transcripts](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs/fastly_q2_2025_transcript.pdf)
- [Q4 investor presentation](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs/fastly_q4_2025_investor_presentation.pdf)

Weak coverage:

- questions that hard-require [Q3 2025 10-Q](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/manifest.json) before it is downloaded
- event-page-only questions

## Version-Specific Notes

`v2` intentionally fixes the item-shape problems found in `v1`:

- `CMP-001` now requires only the source types actually used by its gold evidence.
- `TS-002` no longer pretends to require `investor_supplement`.
- `TS-003` no longer pretends to require `quarterly_results`, and its gold anchors now use exact substring matches instead of all-token-overlap bindings.

## Benchmark Shape

The current corpus supports a `12-item` mini benchmark.

It does not yet support a clean `120-item` benchmark without increasing annotation noise or forcing unsupported question types.
