# Selection Protocol

This file freezes the `v1` mini benchmark before looking at benchmark scores.

## Goal

Build a small benchmark that is diagnostic enough to test:

- retrieval quality
- period-aware comparison
- source prioritization
- evidence conflict handling
- citation-aware synthesis

## Included Corpus

The corpus is fixed to the locally downloaded source packs under:

- [cloudflare docs](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/cloudflare/docs)
- [fastly docs](/Users/jiang/Documents/cv project/bizintel-agent/data/raw/fastly/docs)

The corpus snapshot is tied to:

- `manifest.json`
- `periods.json`
- `download_receipts.jsonl`

for each company.

## Snapshot Freeze

`v1` is defined against this local snapshot:

- `cloudflare`: `12` raw docs, `12` receipt lines, `631` normalized chunks
- `fastly`: `12` raw docs, `12` receipt lines, `484` normalized chunks

If any of those counts or their underlying manifests/receipts change, the benchmark version should change as well.

## Inclusion Rules

- Questions are included only if the current raw corpus clearly supports them.
- Each item can require at most `3` source types.
- Cross-company items are capped at `2`.
- Cloudflare items should rely on `10-K / quarterly results / transcript / Exhibit 99.1 / Investor Day`.
- Fastly items should rely on `10-K / results / supplement / transcript`.

## Exclusion Rules

The following are explicitly excluded from `v1`:

- Cloudflare questions that require a period-matched Q3/Q4 investor deck.
- Fastly questions that require the missing Q3 2025 10-Q as a hard dependency.
- Broad "write a full investment memo comparing both companies" prompts.
- Any question added because it made a system score look better after an earlier run.

## Anti P-Hacking Rules

- Freeze the question set before running results.
- Freeze the dev/test split before running results.
- Freeze anchor evidence before looking at output quality.
- If any item wording, gold outline, or evidence set changes after a run, bump the version.
- Do not drop hard questions after seeing failures.
