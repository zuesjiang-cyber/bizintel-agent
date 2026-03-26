from agent.prompts.research import (
    RESEARCH_PLANNING_SYSTEM_PROMPT,
    QUERY_GENERATION_SYSTEM_PROMPT,
    EVIDENCE_ORGANIZATION_SYSTEM_PROMPT,
    GAP_REFLECTION_SYSTEM_PROMPT,
    EVIDENCE_WRITING_SYSTEM_PROMPT,
    STRICT_VERIFICATION_SYSTEM_PROMPT,
    build_evidence_writing_prompt,
    build_query_generation_prompt,
    build_research_planning_prompt,
    build_strict_verification_prompt,
)


def test_prompt_system_templates_encode_core_guardrails():
    joined = "\n".join(
        [
            RESEARCH_PLANNING_SYSTEM_PROMPT,
            QUERY_GENERATION_SYSTEM_PROMPT,
            EVIDENCE_ORGANIZATION_SYSTEM_PROMPT,
            GAP_REFLECTION_SYSTEM_PROMPT,
            EVIDENCE_WRITING_SYSTEM_PROMPT,
            STRICT_VERIFICATION_SYSTEM_PROMPT,
        ]
    ).lower()

    assert "evidence" in joined
    assert "non-goals" in joined
    assert "insufficient evidence" in joined
    assert "numbers" in joined or "numeric" in joined


def test_research_planning_prompt_requests_structured_section_contract():
    prompt = build_research_planning_prompt(
        user_query="Assess Cloudflare revenue quality",
        mode="company_deep_dive",
        draft_contract={"required_slots": ["revenue quality"]},
        steps=[{"name": "financial_quality", "description": "Revenue quality and margins"}],
    )

    assert "normalized_question" in prompt
    assert "answer_type" in prompt
    assert "estimate" in prompt
    assert "required_facts" in prompt
    assert "allowed_source_types" in prompt
    assert "writing_order" in prompt


def test_query_generation_prompt_requests_metric_specific_queries():
    prompt = build_query_generation_prompt(
        user_query="Assess Cloudflare revenue quality",
        step_name="financial_quality",
        step_description="Revenue quality and margins",
        contract={"query_type": "standard"},
        evidence_requirements={"required_facts": ["revenue quality", "margin"]},
    )

    assert "Generate 2-4 precise retrieval queries" in prompt
    assert "required_facts" in prompt
    assert '"fact_slot"' in prompt
    assert '"query_text"' in prompt
    assert '"expected_evidence_type"' in prompt


def test_evidence_writing_prompt_keeps_uncertainty_and_citations_explicit():
    prompt = build_evidence_writing_prompt(
        mode="memo_section",
        step_name="financial_quality",
        section_contract={"fact_first": True},
        draft_notes="Draft notes",
        evidence_notes=[{"claim": "Revenue was $100M", "chunk_id": "c1", "source_id": "src1"}],
        previous_findings="Previous findings",
    )

    assert "draft" in prompt.lower()
    assert "critique" in prompt.lower()
    assert "final_answer" in prompt
    assert "Every factual sentence must cite" in prompt
    assert "insufficient evidence" in prompt
    assert "uncertainty" in prompt.lower()


def test_strict_verification_prompt_preserves_supported_claims_and_removes_bad_ones():
    prompt = build_strict_verification_prompt(
        section_title="financial_quality",
        content="Revenue was $100M [Source: src1].",
        verification_findings=[{"claim_text": "Revenue was $100M", "failure_reason": "numeric_mismatch"}],
    )

    assert "claim_reviews" in prompt
    assert "issue_type" in prompt
    assert "rewritten_section" in prompt
    assert "unsupported claims are removed or downgraded" in prompt
    assert "verification findings" in prompt.lower()
