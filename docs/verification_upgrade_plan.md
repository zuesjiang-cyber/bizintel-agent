# Verification Upgrade Plan

## Goal

Move the verification layer from a single loose check:

`extract claim -> score NLI -> check numbers`

to a staged pipeline that is easier to trust, debug, and extend:

`extract coarse claim -> normalize into atomic claim -> select evidence -> run hard rules -> run semantic check -> gate final writing`

This upgrade keeps the current trust-first philosophy:

- no citation, no factual sentence
- no numeric alignment, no financial number
- no primary source, no strong financial conclusion

## What Changed

### 1. Richer verification data model

`agent/schemas.py` now carries more structure on both sides of the verifier.

`Claim` now records:

- `claim_type`
- `risk_level`
- `subject`
- `metric`
- `value`
- `unit`
- `period`
- `comparison_basis`
- `directionality`
- `is_inference`
- `requires_primary_source`
- `atomicity`

`VerificationResult` now records:

- `failure_stage`
- `period_verified`
- `currency_verified`
- `unit_verified`
- `directionality_verified`
- `contradiction_detected`
- `verdict_trace`
- `review_notes`

`ConfidenceLevel` also now supports `CONTRADICTED`.

### 2. New verification modules

Three new modules were added under `verification/`:

- `claim_normalizer.py`
- `evidence_selector.py`
- `rule_engine.py`

#### `claim_normalizer.py`

This turns coarse extracted sentences into more atomic, finance-aware claims.

Current behavior includes:

- splitting simple multi-fact growth statements such as `grew 18% to $12.3 billion`
- tagging claim type: `numeric`, `descriptive`, `comparative`, `causal`, `management_commentary`
- extracting metric, value, unit, period, comparison basis, directionality
- marking high-risk claims as requiring primary-source support

#### `evidence_selector.py`

This pulls evidence selection out of the verifier and ranks candidates before semantic scoring.

Current selection behavior includes:

- source-first selection from cited sources
- explicit `chunk_id` preference
- period-aware scoring
- metric and numeric overlap scoring
- neighbor window candidates for nearby chunks
- multi-chunk candidates when a claim explicitly cites multiple chunks

#### `rule_engine.py`

This is now the hard-rule layer for financial verification.

It currently enforces:

- citation/source existence checks
- numeric alignment
- period alignment
- currency alignment
- unit alignment
- directionality alignment
- contradiction detection
- primary-source requirements for high-risk claims

It also keeps the existing derived-numeric support for:

- quick ratio
- working capital
- line-item facts such as capex
- highest/lowest comparison claims backed by grouped numeric evidence

### 3. `evidence_verifier.py` is now the orchestrator

`verification/evidence_verifier.py` still owns NLI loading and scoring, but its role changed.

The verifier now does this:

1. normalize coarse claims
2. run structure checks
3. select evidence candidates
4. score candidates with NLI
5. run hard rules on the best candidate
6. emit a structured `VerificationResult`

Important behavior:

- hard-rule failures are no longer mixed into a single vague unsupported result
- contradiction can now surface separately from missing support
- `verdict_trace` keeps candidate scores and rule-check results for later debugging

### 4. Writer gating now uses verification stages

`agent/report_writer.py` now treats verification output more deliberately:

- hard failures such as numeric mismatch, period mismatch, unit mismatch, contradiction, or missing primary source are deleted from final text
- soft semantic failures are downgraded to insufficiency language
- if a numeric sentence would be deleted only because of a weak semantic score, the writer can keep the original sentence instead of wiping the whole section

This keeps the system conservative without over-pruning simple well-grounded numeric statements.

### 5. Evaluator and benchmark output now surface richer truth signals

`eval/evaluator.py` now reads the richer verifier output and exposes it in claim diagnostics.

In addition to the earlier trust metrics, it now computes:

- `atomic_claim_rate`
- `claim_extract_success_rate`
- `period_match_rate`
- `currency_match_rate`
- `directionality_match_rate`
- `unsupported_numeric_claim_rate`
- `primary_source_missing_rate`
- `unsafe_publish_rate`

`eval/benchmark_runner.py` now carries these metrics into benchmark rows and aggregate summaries.

## Current Flow

```text
claim_extractor
  -> claim_normalizer
  -> evidence_selector
  -> rule_engine
  -> evidence_verifier (NLI + aggregation)
  -> report_writer gating
  -> evaluator / benchmark_runner diagnostics
```

## Design Choices

### Rule-first, LLM-second

This upgrade keeps all hard finance checks deterministic.

LLM review remains optional and limited to gray areas such as:

- borderline semantic support
- answer-status adjudication
- weak vs unsupported support judgments

The system does **not** let an LLM override:

- numeric mismatch
- unit mismatch
- period mismatch
- contradiction
- source absence

### Backward compatibility

The upgrade was implemented to preserve current tests and interfaces where possible.

- `claim_extractor.py` remains the coarse front door
- `evidence_verifier.py` still exposes `verify_memo()` and `summary_stats()`
- older evaluator tests using lightweight fake verifier results continue to work

## Remaining Work

This upgrade establishes the structure, but there are still good next steps:

1. make causal/reason claims split even more cleanly into fact vs explanation parts
2. expand rule coverage for more finance-specific comparison cases
3. add question-aware report rewrites rather than only sentence deletion/downgrade
4. extend benchmark reports to render the new diagnostics more visibly
5. validate the new verifier on a larger live slice after the current code path stabilizes

## Files Touched

- `/Users/jiang/Documents/cv project/bizintel-agent/agent/schemas.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/verification/claim_normalizer.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/verification/evidence_selector.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/verification/rule_engine.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/verification/evidence_verifier.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/agent/report_writer.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/eval/evaluator.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/eval/benchmark_runner.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/tests/test_verifier.py`
- `/Users/jiang/Documents/cv project/bizintel-agent/tests/test_report_writer.py`
