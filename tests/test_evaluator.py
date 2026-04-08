import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "eval" / "evaluator.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location("bizintel_eval_evaluator", EVALUATOR_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

required_fact_recall = MODULE.required_fact_recall
extract_scorable_markdown = MODULE.extract_scorable_markdown
score_markdown = MODULE.score_markdown
score_research_trace = MODULE.score_research_trace


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


def test_extract_scorable_markdown_strips_evidence_gaps_and_verified_support_lines():
    markdown = """# Memo

## Executive Summary
Fastly revenue grew [Source: fastly_q4].

## Hard Fact Findings
Revenue was $100 million [Source: fastly_q4].
*Verified section support: 100% claims supported*

## Evidence Gaps
- Profitability could not be supported confidently.

## Semantic Findings
Management highlighted enterprise demand [Source: fastly_q4_transcript].
"""

    scorable = extract_scorable_markdown(markdown)

    assert "Verified section support" not in scorable
    assert "Evidence Gaps" not in scorable
    assert "Profitability could not be supported confidently" not in scorable
    assert "Management highlighted enterprise demand" in scorable


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


def test_score_markdown_does_not_penalize_refusal_only_summary_as_overclaim():
    markdown = """# Memo

## Executive Summary
Insufficient evidence in the source pack to produce a confident research summary.
"""
    metrics = score_markdown(markdown, ["4.2%"], {})

    assert metrics["total_claims"] == 0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["claim_diagnostics"] == []


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


def test_score_markdown_can_upgrade_gray_claim_with_llm_judge():
    claim = MODULE.Claim(
        claim_id="c1",
        text="Revenue growth was driven by data center demand.",
        section="memo",
        cited_sources=["amd_2022_10k"],
    )

    class FakeExtractor:
        def extract_claims(self, markdown, section):
            return [claim]

    class FakeVerifier:
        def verify_memo(self, claims, evidence_store):
            return [
                SimpleNamespace(
                    confidence=SimpleNamespace(value="weak"),
                    nli_score=0.35,
                    numeric_verified=None,
                    supporting_evidence=["Data Center segment revenue increased due to EPYC server processor sales."],
                    explanation="borderline entailment",
                    failure_reason="low_entailment",
                    supporting_source_ids=["amd_2022_10k"],
                    supporting_chunk_ids=["chunk-1"],
                    primary_source_supported=True,
                )
            ]

        def summary_stats(self, verification_results):
            return {
                "total_claims": 1,
                "citation_marker_coverage": 1.0,
                "verified_claim_coverage": 0.0,
                "avg_nli_score": 0.35,
            }

    class FakeJudge:
        def adjudicate_claim_support(self, **kwargs):
            return {"support_label": "strong_support", "reason": "Direct support from the cited filing excerpt."}

    metrics = score_markdown(
        "Revenue growth was driven by data center demand. [Source: amd_2022_10k]",
        ["data center"],
        {"amd_2022_10k": [{"text": "Data Center segment revenue increased due to EPYC server processor sales.", "source_type": "annual_report", "is_primary": True}]},
        question_text="What drove revenue change as of FY22 for AMD?",
        extractor=FakeExtractor(),
        verifier=FakeVerifier(),
        judge=FakeJudge(),
    )

    assert metrics["strong_support_rate"] == 1.0
    assert metrics["claim_diagnostics"][0]["support_label"] == "strong_support"
    assert metrics["claim_diagnostics"][0]["llm_adjudicated"] is True


def test_score_markdown_does_not_override_numeric_mismatch_with_llm_judge():
    claim = MODULE.Claim(
        claim_id="c1",
        text="Revenue was $5 billion.",
        section="memo",
        cited_sources=["amd_2022_10k"],
        contains_numbers=True,
        extracted_numbers=["$5 billion"],
    )

    class FakeExtractor:
        def extract_claims(self, markdown, section):
            return [claim]

    class FakeVerifier:
        def verify_memo(self, claims, evidence_store):
            return [
                SimpleNamespace(
                    confidence=SimpleNamespace(value="unsupported"),
                    nli_score=0.2,
                    numeric_verified=False,
                    supporting_evidence=["Revenue was $4 billion."],
                    explanation="numeric mismatch",
                    failure_reason="numeric_mismatch",
                    supporting_source_ids=["amd_2022_10k"],
                    supporting_chunk_ids=["chunk-1"],
                    primary_source_supported=True,
                )
            ]

        def summary_stats(self, verification_results):
            return {
                "total_claims": 1,
                "citation_marker_coverage": 1.0,
                "verified_claim_coverage": 0.0,
                "avg_nli_score": 0.2,
            }

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        def adjudicate_claim_support(self, **kwargs):
            self.calls += 1
            return {"support_label": "strong_support", "reason": "Should not be used."}

    judge = FakeJudge()
    metrics = score_markdown(
        "Revenue was $5 billion. [Source: amd_2022_10k]",
        ["$5 billion"],
        {"amd_2022_10k": [{"text": "Revenue was $4 billion.", "source_type": "annual_report", "is_primary": True}]},
        question_text="What was AMD revenue?",
        extractor=FakeExtractor(),
        verifier=FakeVerifier(),
        judge=judge,
    )

    assert judge.calls == 0
    assert metrics["strong_support_rate"] == 0.0
    assert metrics["claim_diagnostics"][0]["support_label"] == "unsupported"


def test_score_research_trace_reports_completion_and_slot_coverage():
    result = {
        "subquestion_results": [
            SimpleNamespace(
                status="completed",
                refusal_reason=None,
                answer_text="Revenue was $10 million.",
                supported_content="Revenue was $10 million.",
                evidence=[
                    SimpleNamespace(company="fastly", period="2025Q4"),
                    SimpleNamespace(company="fastly", period="2025Q4"),
                ],
                subquestion=SimpleNamespace(
                    question_id="q1",
                    required=True,
                    lane="hard_fact",
                    fact_slot="reported revenue",
                    metric_family="revenue",
                    text="What was reported revenue?",
                ),
            ),
            SimpleNamespace(
                status="refused",
                refusal_reason="No high-trust semantic evidence.",
                answer_text="",
                supported_content="",
                evidence=[SimpleNamespace(company="fastly", period="2025Q4")],
                subquestion=SimpleNamespace(
                    question_id="q2",
                    required=True,
                    lane="semantic",
                    fact_slot="management demand commentary",
                    metric_family="management",
                    text="What demand commentary did management provide?",
                ),
            ),
        ],
        "research_trace": SimpleNamespace(
            subquestions=[
                SimpleNamespace(question_id="q1", status="completed"),
                SimpleNamespace(question_id="q2", status="refused"),
            ],
            decisions=[
                SimpleNamespace(question_id="q1"),
                SimpleNamespace(question_id="q2"),
            ],
        ),
    }

    metrics = score_research_trace(
        result,
        ["reported revenue", "management demand commentary"],
        expected_companies=["fastly"],
        target_periods=["2025Q4"],
    )

    assert metrics["subquestion_completion_rate"] == 0.5
    assert metrics["required_subquestion_coverage"] == 0.5
    assert metrics["required_slot_coverage"] == 0.5
    assert metrics["decision_replay_consistency"] == 1.0
    assert metrics["controller_failure_reasons"] == ["No high-trust semantic evidence."]
