import pytest

from agent.config import settings
from agent import report_writer as report_writer_module
from agent.report_writer import ReportWriter
from agent.llm_utils import build_stub_section_analysis
from agent.schemas import AnalysisMode, AnalysisPlan, AnalysisStep, Claim, ConfidenceLevel, RetrievedChunk, VerificationResult


@pytest.fixture
def force_stub_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "stub")
    monkeypatch.setattr(settings, "openai_api_key", "")


def test_stub_executive_summary_is_persisted(force_stub_llm):
    writer = ReportWriter()
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "company_overview": {
            "content": "- Stripe is a payments company founded in 2010 [Source: stripe_profile]",
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe is a payments company founded in 2010.",
                }
            ],
        }
    }

    memo = writer.generate_memo(plan, step_outputs, [])
    assert memo.executive_summary
    assert memo.sections[0].writing_trace["final_answer"]


def test_render_verification_summary_uses_nli_score(force_stub_llm):
    writer = ReportWriter()
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "financial_analysis": {
            "content": "Stripe raised $8.7 billion in funding [Source: stripe_profile].",
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe raised $8.7 billion in funding.",
                }
            ],
        }
    }

    memo = writer.generate_memo(plan, step_outputs, [])
    rendered = writer.render_memo(memo)
    assert "Average NLI score" in rendered


def test_stub_section_analysis_prefers_financial_facts():
    step = AnalysisStep(
        name="financial_analysis",
        description="Financial metrics, funding history, valuation changes, and revenue data.",
        required=True,
    )
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Company: Stripe Founded: 2010 Total Funding: $8.7B Latest Valuation: $50B Key Products: Stripe Payments, Stripe Connect",
            source_id="stripe_profile",
            page=None,
            score=1.0,
            bm25_rank=0,
            dense_rank=0,
            rerank_score=1.0,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Stripe is a financial infrastructure platform for the internet.",
            source_id="stripe_about",
            page=None,
            score=0.9,
            bm25_rank=1,
            dense_rank=1,
            rerank_score=0.9,
        ),
    ]

    rendered = build_stub_section_analysis(step, "Assess Stripe valuation and funding", chunks)

    assert "raised $8.7B in total funding" in rendered
    assert "latest valuation is $50B" in rendered


def test_demo_rendering_labels_heuristic_support():
    writer = ReportWriter(demo_mode=True)
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "company_overview": {
            "content": "Stripe was founded in 2010 [Source: stripe_profile].",
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe was founded in 2010.",
                }
            ],
        }
    }

    memo = writer.generate_memo(plan, step_outputs, [])
    rendered = writer.render_memo(memo)

    assert "Demo Heuristic Support" in rendered
    assert "walkthrough aids" in rendered


def test_render_without_internal_verification_omits_verification_sections(force_stub_llm):
    writer = ReportWriter(enable_report_verification=False)
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "company_overview": {
            "content": "Stripe was founded in 2010 [Source: stripe_profile].",
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe was founded in 2010.",
                }
            ],
        }
    }

    memo = writer.generate_memo(plan, step_outputs, [])
    rendered = writer.render_memo(memo)

    assert "## Verification Summary" not in rendered
    assert "Confidence:" not in rendered


def test_best_supported_sentence_prefers_complete_evidence_sentence(force_stub_llm):
    writer = ReportWriter()

    sentence = writer._best_supported_sentence(
        {
            "claim": "third-party log management and debugging challenges.",
            "evidence_text": (
                "third-party log management and debugging challenges. "
                "For customers building apps with Compute, Fastly tags individual requests "
                "with unique identifiers and maintains request tracing parameters."
            ),
        }
    )

    assert sentence.startswith("For customers building apps with Compute")


def test_generate_memo_prunes_unsupported_claims(force_stub_llm):
    writer = ReportWriter()
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "financial_analysis": {
            "content": (
                "Stripe raised $8.7 billion in funding [Source: stripe_profile].\n"
                "Stripe has 50,000 employees [Source: stripe_profile]."
            ),
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe raised $8.7 billion in funding.",
                }
            ],
        }
    }

    def fake_verify(claims, evidence_store):
        return [
            VerificationResult(
                claim=Claim(
                    claim_id="c1",
                    text="Stripe raised $8.7 billion in funding",
                    section="financial_analysis",
                    cited_sources=["stripe_profile"],
                    contains_numbers=True,
                    extracted_numbers=["$8.7 billion"],
                ),
                confidence=ConfidenceLevel.STRONG,
                nli_score=0.9,
                numeric_verified=True,
                supporting_evidence=["Stripe raised $8.7 billion in funding."],
                explanation="Supported.",
            ),
            VerificationResult(
                claim=Claim(
                    claim_id="c2",
                    text="Stripe has 50,000 employees",
                    section="financial_analysis",
                    cited_sources=["stripe_profile"],
                    contains_numbers=True,
                    extracted_numbers=["50,000"],
                ),
                confidence=ConfidenceLevel.UNSUPPORTED,
                nli_score=0.0,
                numeric_verified=False,
                supporting_evidence=[],
                explanation="Unsupported.",
                failure_reason="numeric_mismatch",
            ),
        ]

    writer.verifier.verify_memo = fake_verify
    memo = writer.generate_memo(plan, step_outputs, [])

    assert "raised $8.7 billion" in memo.sections[0].content
    assert "50,000 employees" not in memo.sections[0].content


def test_generate_memo_rewrites_unsupported_nonnumeric_claims_to_insufficient_evidence(force_stub_llm):
    writer = ReportWriter()
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "competitive_landscape": {
            "content": "Stripe appears stronger than Visa in enterprise payments [Source: stripe_profile].",
            "sources_used": [{"source_id": "stripe_profile"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_profile",
                    "text": "Stripe serves businesses globally.",
                }
            ],
        }
    }

    call_count = {"value": 0}

    def fake_verify(claims, evidence_store):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return [
                VerificationResult(
                    claim=Claim(
                        claim_id="c1",
                        text="Stripe appears stronger than Visa in enterprise payments",
                        section="competitive_landscape",
                        cited_sources=["stripe_profile"],
                        contains_numbers=False,
                    ),
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.1,
                    numeric_verified=None,
                    supporting_evidence=[],
                    explanation="Unsupported.",
                    failure_reason="low_entailment",
                )
            ]
        return []

    writer.verifier.verify_memo = fake_verify
    memo = writer.generate_memo(plan, step_outputs, [])

    assert "Stripe appears stronger than Visa" not in memo.sections[0].content
    assert "Insufficient evidence" in memo.sections[0].content
    assert memo.sections[0].verification_results == []


def test_generate_memo_prunes_primary_source_missing_high_risk_claim(force_stub_llm):
    writer = ReportWriter()
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Stripe")
    step_outputs = {
        "financial_analysis": {
            "content": "Stripe revenue was $1 billion [Source: stripe_blog].",
            "sources_used": [{"source_id": "stripe_blog"}],
            "raw_evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "stripe_blog",
                    "text": "Stripe revenue was $1 billion.",
                }
            ],
        }
    }

    def fake_verify(claims, evidence_store):
        return [
            VerificationResult(
                claim=Claim(
                    claim_id="c1",
                    text="Stripe revenue was $1 billion",
                    section="financial_analysis",
                    cited_sources=["stripe_blog"],
                    contains_numbers=True,
                    extracted_numbers=["$1 billion"],
                    claim_type="numeric",
                    risk_level="high",
                    requires_primary_source=True,
                ),
                confidence=ConfidenceLevel.UNSUPPORTED,
                nli_score=0.9,
                numeric_verified=True,
                supporting_evidence=["Stripe revenue was $1 billion."],
                explanation="Missing primary source support.",
                failure_reason="primary_source_missing",
                failure_stage="rules",
                supporting_source_ids=["stripe_blog"],
                supporting_chunk_ids=["c1"],
                primary_source_supported=False,
            )
        ]

    writer.verifier.verify_memo = fake_verify
    memo = writer.generate_memo(plan, step_outputs, [])

    assert "revenue was $1 billion" not in memo.sections[0].content.lower()


def test_live_writer_uses_small_llm_pass_when_evidence_notes_exist(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    writer = ReportWriter(enable_report_verification=False)

    captured = {}

    def fake_generate_text_response(*args, **kwargs):
        captured.update(kwargs)
        return '{"draft":"draft","critique":["tighten wording"],"final_answer":"Cloudflare sells network security subscriptions [Chunk: c1] [Source: cloudflare_profile]."}'

    monkeypatch.setattr(
        report_writer_module,
        "generate_text_response",
        fake_generate_text_response,
    )

    content = writer._compose_section_content(
        "business_model",
        {
            "content": "Cloudflare sells network security subscriptions [Source: cloudflare_profile].",
            "evidence_notes": [
                {
                    "claim": "Cloudflare sells network security subscriptions",
                    "chunk_id": "c1",
                    "source_id": "cloudflare_profile",
                    "evidence_text": "Cloudflare sells network security subscriptions to enterprise customers.",
                    "evidence_type": "fact",
                }
            ],
        },
        {"fact_first": True},
    )

    assert "Cloudflare sells network security subscriptions" in content
    assert captured["max_tokens"] == 350
    assert captured["max_retries"] == 1


def test_live_writer_falls_back_to_stub_when_llm_write_fails(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    writer = ReportWriter(enable_report_verification=False)

    monkeypatch.setattr(
        report_writer_module,
        "generate_text_response",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("gateway timeout")),
    )

    content = writer._compose_section_content(
        "business_model",
        {
            "content": "- draft",
            "missing_facts": ["monetization"],
            "evidence_notes": [
                {
                    "claim": "Cloudflare sells network security subscriptions",
                    "chunk_id": "c1",
                    "source_id": "cloudflare_profile",
                    "evidence_text": "Cloudflare sells network security subscriptions to enterprise customers.",
                    "evidence_type": "fact",
                }
            ],
        },
        {"fact_first": True},
    )

    assert "Cloudflare sells network security subscriptions" in content
    assert "Insufficient evidence" in content


def test_live_writer_renders_structured_final_answer_dict(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    writer = ReportWriter(enable_report_verification=False)

    monkeypatch.setattr(
        report_writer_module,
        "generate_text_response",
        lambda *args, **kwargs: (
            '{"draft":{"facts":["draft fact"]},"critique":["tighten"],'
            '"final_answer":{"facts":["Cloudflare is a connectivity cloud company [Chunk: c1] [Source: cloudflare_10k]"],'
            '"management_commentary":["Management said AI agents are a new class of users [Chunk: c2] [Source: cloudflare_q4_call]"]}}'
        ),
    )

    content = writer._compose_section_content(
        "business_model",
        {
            "content": "",
            "evidence_notes": [
                {
                    "claim": "Cloudflare is a connectivity cloud company",
                    "chunk_id": "c1",
                    "source_id": "cloudflare_10k",
                    "evidence_text": "Cloudflare is a connectivity cloud company.",
                    "evidence_type": "fact",
                }
            ],
        },
        {"fact_first": True},
    )

    assert "**Facts**" in content
    assert "**Management Commentary**" in content
    assert "[Source: cloudflare_10k]" in content
    assert not content.lstrip().startswith("{")


def test_live_writer_falls_back_when_final_answer_has_no_citations(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    writer = ReportWriter(enable_report_verification=False)

    monkeypatch.setattr(
        report_writer_module,
        "generate_text_response",
        lambda *args, **kwargs: '{"draft":"draft","critique":["remove hedging"],"final_answer":"Cloudflare sells network security subscriptions to enterprises."}',
    )

    content = writer._compose_section_content(
        "business_model",
        {
            "content": "",
            "missing_facts": ["monetization"],
            "evidence_notes": [
                {
                    "claim": "Cloudflare sells network security subscriptions",
                    "chunk_id": "c1",
                    "source_id": "cloudflare_profile",
                    "evidence_text": "Cloudflare sells network security subscriptions to enterprise customers.",
                    "evidence_type": "fact",
                }
            ],
        },
        {"fact_first": True},
    )

    assert "[Source: cloudflare_profile]" in content
    assert "Insufficient evidence" in content


def test_live_writer_handles_non_json_response_without_crashing(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    writer = ReportWriter(enable_report_verification=False)

    monkeypatch.setattr(
        report_writer_module,
        "generate_text_response",
        lambda *args, **kwargs: "Cloudflare sells network security subscriptions [Chunk: c1] [Source: cloudflare_profile].",
    )

    output = {
        "content": "",
        "missing_facts": ["monetization"],
        "evidence_notes": [
            {
                "claim": "Cloudflare sells network security subscriptions",
                "chunk_id": "c1",
                "source_id": "cloudflare_profile",
                "evidence_text": "Cloudflare sells network security subscriptions to enterprise customers.",
                "evidence_type": "fact",
            }
        ],
    }

    content = writer._compose_section_content(
        "business_model",
        output,
        {"fact_first": True},
    )

    assert content == "Cloudflare sells network security subscriptions [Chunk: c1] [Source: cloudflare_profile]."
    assert output["writing_trace"]["final_answer"] == content
    assert output["writing_trace"]["draft"] == "Cloudflare sells network security subscriptions [Chunk: c1] [Source: cloudflare_profile]."
