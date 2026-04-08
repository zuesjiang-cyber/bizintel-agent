# Selection Protocol

This file freezes the `v2` mini benchmark after the `v1` diagnostic run exposed methodology holes.

## Why `v2` Exists

`v1` was useful as a failure-revealing dry run, but it had three problems that made the version unsafe to keep extending in place:

- `CMP-001`, `TS-002`, and `TS-003` had schema-to-gold mismatches.
- the runner used a shared split-level index, which allowed irrelevant-company contamination for single-company items.
- the freeze only recorded counts, not content hashes.

Per the benchmark rules, those are version-bump changes, not silent edits.

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
- the raw document file hashes recorded in `corpus_snapshot.json`

for each company.

## Snapshot Freeze

`v2` is defined against this local snapshot:

- `cloudflare`: `12` raw docs, `12` receipt lines, `631` normalized chunks
- `fastly`: `12` raw docs, `12` receipt lines, `484` normalized chunks

The canonical freeze artifact is:

- [corpus_snapshot.json](/Users/jiang/Documents/cv project/bizintel-agent/data/benchmark/v2/corpus_snapshot.json)

If the snapshot id or the underlying manifest/receipt/doc hashes change, the benchmark version must change too.

## Inclusion Rules

- Questions are included only if the current raw corpus clearly supports them.
- Each item can require at most `3` source types.
- Cross-company items are capped at `2`.
- Cloudflare items should rely on `10-K / quarterly results / transcript / Exhibit 99.1 / Investor Day`.
- Fastly items should rely on `10-K / results / supplement / transcript`.
- Single-company items are executed against a same-company index only.
- Cross-company items remain in the frozen benchmark as diagnostic rows, but the current mainline controller may refuse them when scope is unsupported.

## Exclusion Rules

The following are explicitly excluded from `v2`:

- Cloudflare questions that require a period-matched Q3/Q4 investor deck.
- Fastly questions that require the missing Q3 2025 10-Q as a hard dependency.
- Broad "write a full investment memo comparing both companies" prompts.
- Any question added because it made a system score look better after an earlier run.

## Split Rules

The `dev` split must cover all major benchmark categories so controller or prompt tuning does not force test leakage from unseen categories.

`v2` therefore keeps:

- at least one `company_overview` item in `dev`
- at least one `risk_catalyst` item in `dev`
- at least one `comparison` item in `dev`
- at least one `time_sensitive` item in `dev`
- at least one `evidence_conflict` item in `dev`

## Anti P-Hacking Rules

- Freeze the question set before running results.
- Freeze the dev/test split before running results.
- Freeze anchor evidence before looking at output quality.
- If any item wording, gold outline, or evidence set changes after a run, bump the version.
- Do not drop hard questions after seeing failures.

## Profile Rules

`v2` may define additional named profiles in [profiles.json](/Users/jiang/Documents/cv project/bizintel-agent/data/benchmark/v2/profiles.json), but they do not replace the frozen `dev` and `test` splits.

- `diagnostic` profiles exist to reveal failures.
- `showcase` profiles exist to demonstrate conservative, evidence-bound behavior on narrower questions.
- A showcase profile must be named and frozen before it is used in reporting.
- A showcase profile must never be reported as if it were the full benchmark.

## Current Product-Boundary Note

The frozen `v2` question set still contains comparison items because they are useful diagnostics.
Current mainline product behavior remains single-company-first, so comparison rows should be interpreted as stress tests unless and until multi-company research becomes a supported feature.
