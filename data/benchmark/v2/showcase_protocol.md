# Showcase Protocol

This protocol defines a separate `showcase` track for `v2`. It does not replace the full diagnostic benchmark.

## Why a Separate Showcase Track Exists

The diagnostic suite is intentionally broad and failure-revealing. It includes open-ended synthesis prompts that are useful for finding hallucinations, but they are a poor fit for a public trust demo when the product goal is:

- high verified citation binding
- low unsupported-claim rate
- conservative, evidence-first answers

Those are different jobs.

## Target Envelope

The `trust_showcase_v1` profile is designed around this envelope:

- `verified_claim_coverage >= 0.828`
- `unsupported_claim_rate <= 0.1225`

These are target gates, not guaranteed outputs. If the system misses them, the run should report that directly.

## Question Design Rules

Showcase questions should:

- ask for at most `3` must-cover facts
- lock the answer to named periods
- name the allowed source pack implicitly or explicitly
- prefer source-priority or period-diff tasks over broad investment judgments
- allow the answer to distinguish stronger evidence from weaker management framing

Showcase questions should not:

- ask for broad catalyst/risk narratives
- ask which company is "better" without a bounded comparison frame
- require long multi-paragraph investment theses
- encourage unstated causal explanations

## Current Showcase Portfolio

`trust_showcase_v1` includes:

- `BO-002`
- `NUM-001`
- `TS-002`
- `TS-003`
- `SRC-001`

The common pattern is that each item is:

- medium difficulty
- period-locked
- answerable from `2` minimum distinct sources
- narrow enough to support slot-like, evidence-bound output

## Anti P-Hacking Rule

The showcase track is allowed to be narrower than the diagnostic track, but it must be frozen before release reporting:

- do not swap items after seeing the latest score
- if the showcase item list changes, record a new profile name
- if item wording or evidence changes, bump the benchmark version
