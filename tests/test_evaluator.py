import importlib.util
from pathlib import Path


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "eval" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("bizintel_eval_evaluator", EVALUATOR_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

required_fact_recall = MODULE.required_fact_recall
extract_scorable_markdown = MODULE.extract_scorable_markdown
score_markdown = MODULE.score_markdown


def test_required_fact_recall_counts_present_facts():
    markdown = "Stripe crossed 1 trillion in TPV and was valued at 50 billion."

    recall = required_fact_recall(markdown, ["1 trillion", "50 billion", "enterprise"])

    assert recall == 2 / 3


def test_score_markdown_reports_unsupported_rate():
    markdown = (
        "Stripe raised $8.7 billion [Source: stripe_profile]. "
        "Stripe has 50,000 employees [Source: stripe_profile]."
    )
    evidence_store = {
        "stripe_profile": [
            "Stripe has raised $8.7 billion in total funding and has roughly 8,000 employees."
        ]
    }

    metrics = score_markdown(markdown, ["8.7"], evidence_store)

    assert metrics["total_claims"] >= 1
    assert 0.0 <= metrics["unsupported_claim_rate"] <= 1.0
    assert metrics["required_fact_recall"] == 1.0
    assert metrics["claim_diagnostics"]
    assert "scoreable_markdown" in metrics
    assert "citation_coverage" not in metrics
    assert "citation_marker_coverage" in metrics
    assert "verified_claim_coverage" in metrics


def test_score_markdown_can_reuse_extractor_and_verifier():
    markdown = "Stripe raised $8.7 billion [Source: stripe_profile]."
    evidence_store = {
        "stripe_profile": ["Stripe has raised $8.7 billion in total funding."]
    }
    extractor = MODULE.ClaimExtractor()
    verifier = MODULE.EvidenceVerifier(use_dummy_model=True)

    metrics = score_markdown(
        markdown,
        ["8.7"],
        evidence_store,
        extractor=extractor,
        verifier=verifier,
    )

    assert metrics["total_claims"] >= 1


def test_extract_scorable_markdown_strips_sources_and_verification_summary():
    markdown = """# Memo

Generated: 2026-03-25 | Mode: company_deep_dive | Confidence: 81%

## Executive Summary
Fastly revenue grew [Source: fastly_q4].

## Financial Analysis
Adjusted EBITDA was positive [Source: fastly_q4].
*Section confidence: 80% claims supported*

---
## Sources
- **[fastly_q4]** Fastly Q4

## Verification Summary
- Total claims analyzed: 20
- Unsupported: 10
"""

    scorable = extract_scorable_markdown(markdown)

    assert "## Sources" not in scorable
    assert "## Verification Summary" not in scorable
    assert "Section confidence" not in scorable
    assert "Fastly revenue grew" in scorable


def test_extract_scorable_markdown_strips_coverage_contract():
    markdown = """# Memo

## Executive Summary
Fastly revenue grew [Source: fastly_q4].

## Coverage Contract
- Q3 vs Q4 comparison
- management commentary shift

## Business Model
Fastly sells edge cloud services [Source: fastly_10k].
"""

    scorable = extract_scorable_markdown(markdown)

    assert "Coverage Contract" not in scorable
    assert "Q3 vs Q4 comparison" not in scorable
    assert "Fastly sells edge cloud services" in scorable


def test_score_markdown_ignores_verification_appendix_claims():
    markdown = """# Memo

## Executive Summary
Fastly revenue was $100 million [Source: fastly_q4].

---
## Sources
- **[fastly_q4]** Fastly Q4

## Verification Summary
- Unsupported claim: Fastly revenue was $999 million.
"""
    evidence_store = {
        "fastly_q4": ["Fastly revenue was $100 million."]
    }

    metrics = score_markdown(markdown, ["$100 million"], evidence_store)

    assert metrics["required_fact_recall"] == 1.0
    assert metrics["unsupported_claim_rate"] < 1.0


def test_score_markdown_flags_unscorable_cited_content_as_unsupported():
    markdown = """# Memo

## Executive Summary
- noise only [Source: fastly_10k]
"""
    evidence_store = {
        "fastly_10k": ["Fastly discusses third-party log management and debugging challenges."]
    }

    metrics = score_markdown(markdown, ["debugging challenges"], evidence_store)

    assert metrics["total_claims"] == 0
    assert metrics["unsupported_claim_rate"] == 1.0
    assert metrics["claim_diagnostics"][0]["failure_reason"] == "no_extractable_claims"


def test_score_markdown_uses_fallback_extraction_for_cited_bullet_fragments():
    markdown = """# Memo

## Executive Summary
- third-party log management and debugging challenges. [Source: fastly_10k]
"""
    evidence_store = {
        "fastly_10k": ["Fastly discusses third-party log management and debugging challenges."]
    }

    metrics = score_markdown(markdown, ["debugging challenges"], evidence_store)

    assert metrics["total_claims"] == 1
    assert metrics["unsupported_claim_rate"] < 1.0
