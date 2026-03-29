from agent.orchestrator import BizIntelAgent
from agent.research_controller import ResearchController
from agent.schemas import AnalysisMode, ResearchLane, ResearchReplayRecord, ResearchQuestionResult, ResearchQuestionStatus, ResearchSubquestion, ResearchTask
from retrieval.hybrid_retriever import HybridRetriever
from agent.schemas import RetrievedChunk


def test_mixed_query_decomposes_into_hard_fact_and_semantic_subquestions():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller._build_task(
        query="Assess Stripe revenue quality and management outlook",
        mode=AnalysisMode.COMPANY,
    )

    subquestions = controller._decompose_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    lanes = {item.lane for item in subquestions}

    assert ResearchLane.HARD_FACT in lanes
    assert ResearchLane.SEMANTIC in lanes


def test_hard_fact_strategy_prefers_exact_period_numeric_chunk():
    retriever = HybridRetriever(load_models=False)
    retriever.index(
        [
            {
                "chunk_id": "c1",
                "text": "Cloudflare total revenue was $400 million in Q4 2024.",
                "source_id": "cloudflare_q4",
                "company": "cloudflare",
                "period": "2024Q4",
                "source_type": "quarterly_results",
                "is_primary": True,
            },
            {
                "chunk_id": "c2",
                "text": "Cloudflare total revenue was $350 million in Q3 2024.",
                "source_id": "cloudflare_q3",
                "company": "cloudflare",
                "period": "2024Q3",
                "source_type": "quarterly_results",
                "is_primary": True,
            },
            {
                "chunk_id": "c3",
                "text": "Cloudflare discussed revenue quality and enterprise demand trends.",
                "source_id": "cloudflare_commentary",
                "company": "cloudflare",
                "period": "2024Q4",
                "source_type": "webpage",
                "is_primary": False,
            },
        ]
    )

    results, trace = retriever.retrieve_with_trace(
        "Cloudflare Q4 2024 revenue",
        top_k=2,
        filters={"companies": ["cloudflare"], "periods": ["2024Q4"]},
        strategy="hard_fact",
    )

    assert results[0].chunk_id == "c1"
    assert trace["strategy"] == "hard_fact"


def test_deep_research_run_records_research_tree_and_decisions():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research(
        "Assess Stripe's revenue quality and management outlook",
        mode=AnalysisMode.COMPANY,
    )

    assert result["subquestion_results"]
    assert result["research_trace"].decisions
    assert all(item.status.value != "pending" for item in result["subquestion_results"])
    assert result["memo_object"].contract["research_tree"]


def test_multi_company_scope_is_refused_explicitly():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research(
        "Compare Cloudflare and Fastly on monetization style",
        mode=AnalysisMode.COMPETITIVE,
    )

    assert result["memo_object"].contract["scope_supported"] is False
    assert result["subquestion_results"][0].status.value == "refused"
    assert "Cross-company comparison is out of scope" in result["memo_object"].executive_summary


def test_build_task_collects_multiple_target_periods():
    controller = ResearchController(load_models=False, demo_mode=True)

    task = controller.build_task(
        query="How did Fastly's growth narrative change from Q2 2025 to Q4 2025?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
    )

    assert task.company_id == "fastly"
    assert task.target_periods == ["2025Q2", "2025Q4"]
    assert task.period is None


def test_required_slots_are_injected_into_subquestion_plan():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="How does Fastly make money, and what does management emphasize as the most important revenue mix components in Q4 2025?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025FY", "2025Q4"],
        required_slots=["revenue mix", "business model", "management KPI focus"],
        required_source_types=["annual_report", "quarterly_results", "investor_supplement"],
    )

    subquestions = controller.plan_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    fact_slots = {item.fact_slot for item in subquestions}

    assert "revenue mix" in fact_slots
    assert "business model" in fact_slots
    assert any(
        item.lane == ResearchLane.SEMANTIC
        and ("management" in item.text.lower() or "kpi" in item.text.lower())
        for item in subquestions
    )


def test_management_commentary_period_diff_slots_stay_semantic():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="What changed from Q3 2025 to Q4 2025 in how management described revenue mix and growth drivers?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q3", "2025Q4"],
        required_slots=["Q3 vs Q4", "revenue mix", "growth drivers"],
        required_source_types=["investor_supplement", "earnings_call_transcript"],
        query_types=["period_diff", "management_commentary"],
    )

    subquestions = controller.plan_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    slot_lanes = {item.fact_slot: item.lane for item in subquestions}

    assert slot_lanes["Q3 vs Q4"] == ResearchLane.SEMANTIC
    assert any(
        item.lane == ResearchLane.SEMANTIC
        and "revenue mix" in item.text.lower()
        and "growth drivers" in item.text.lower()
        for item in subquestions
    )


def test_source_priority_queries_generate_semantic_claim_weighting_questions():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="Using Q4 2025 results, transcript, and Exhibit 99.1, identify one high-confidence claim and one lower-confidence management claim about Cloudflare's growth outlook.",
        mode=AnalysisMode.COMPANY,
        company_id="cloudflare",
        target_periods=["2025Q4"],
        required_slots=["high-confidence claim", "lower-confidence management claim", "evidence distinction"],
        required_source_types=["quarterly_results", "earnings_call_transcript"],
        query_types=["source_priority", "evidence_weighting"],
    )

    subquestions = controller.plan_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    slot_lanes = {item.fact_slot: item.lane for item in subquestions}

    assert slot_lanes["high-confidence claim"] == ResearchLane.SEMANTIC
    assert slot_lanes["lower-confidence management claim"] == ResearchLane.SEMANTIC
    assert any("evidence distinction" in item.text.lower() for item in subquestions)


def test_hard_fact_conflict_detection_ignores_complementary_numeric_chunks():
    controller = ResearchController(load_models=False, demo_mode=True)
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What revenue facts are disclosed for Fastly?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue",
        metric_family="revenue",
        needs_numeric_verification=True,
    )
    first = RetrievedChunk(
        chunk_id="c1",
        text="Record fourth quarter revenue was $172.6 million and gross margin improved.",
        source_id="s1",
        page=None,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="fastly",
        period="2025Q4",
        source_type="quarterly_results",
        is_primary=True,
        metric_signals=["revenue"],
        content_type="quantitative",
    )
    second = RetrievedChunk(
        chunk_id="c2",
        text="Network services revenue was $130.8 million in the same quarter.",
        source_id="s2",
        page=None,
        score=0.9,
        bm25_rank=1,
        dense_rank=1,
        rerank_score=0.9,
        company="fastly",
        period="2025Q4",
        source_type="investor_supplement",
        is_primary=True,
        metric_signals=["revenue"],
        content_type="quantitative",
    )

    conflict = controller._detect_conflict(
        subquestion,
        [
            {"chunk_id": "c1", "text": first.text, "entailment": 0.9, "contradiction": 0.1},
            {"chunk_id": "c2", "text": second.text, "entailment": 0.85, "contradiction": 0.1},
        ],
        {"c1": first, "c2": second},
    )

    assert conflict is False


def test_assess_evidence_uses_declarative_support_hypotheses_for_hard_facts(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="How does Fastly make money, and what does management emphasize as the most important revenue mix components in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="For 2025 Q4 specifically, what is Fastly’s revenue breakdown (or mix) by major product line (e.g., Delivery vs Security vs Other/Compute), as reported in filings/earnings materials?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue mix",
        metric_family="revenue",
        needs_numeric_verification=True,
    )
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Network services revenue of $130.8 million, Security revenue of $35.4 million, and Other revenue of $6.4 million were reported in Q4 2025.",
            source_id="fastly_q4_2025_results",
            page=None,
            score=1.0,
            bm25_rank=0,
            dense_rank=0,
            rerank_score=1.0,
            company="fastly",
            period="2025Q4",
            source_type="quarterly_results",
            is_primary=True,
            trust_level=5,
            metric_signals=["revenue"],
            content_type="quantitative",
        )
    ]

    seen_hypotheses = []

    def fake_score(hypothesis, chunk_payload):
        seen_hypotheses.append(hypothesis)
        entailment = 0.72 if "reports revenue mix" in hypothesis.lower() else 0.01
        chunk = chunk_payload[0]
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
                "source_type": chunk["source_type"],
                "is_primary": chunk["is_primary"],
                "contradiction": 0.02,
                "entailment": entailment,
                "neutral": 0.26,
            }
        ]

    monkeypatch.setattr(controller.verifier, "score_hypothesis_against_chunks", fake_score)

    assessment = controller._assess_evidence(task, subquestion, chunks)

    assert any("reports revenue mix" in hypothesis.lower() for hypothesis in seen_hypotheses)
    assert assessment.sufficient is True
    assert assessment.matched_chunk_ids == ["c1"]
    assert assessment.best_support >= 0.78


def test_assess_evidence_uses_support_score_for_semantic_commentary(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What changed from Q3 2025 to Q4 2025 in how management described revenue mix and growth drivers?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q2",
        text="For 2025 Q4, what revenue mix components does management explicitly emphasize as most important, based on prepared remarks and Q&A?",
        lane=ResearchLane.SEMANTIC,
        priority=1,
        fact_slot="management emphasis",
        metric_family="management_tone",
        needs_numeric_verification=False,
    )
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Management highlighted stronger balanced traffic mix and better contribution from security alongside delivery in Q4 2025.",
            source_id="fastly_q4_2025_transcript",
            page=None,
            score=1.0,
            bm25_rank=0,
            dense_rank=0,
            rerank_score=1.0,
            company="fastly",
            period="2025Q4",
            source_type="earnings_call_transcript",
            is_primary=True,
            trust_level=5,
            content_type="qualitative",
            metric_signals=["management_tone"],
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Prepared remarks framed security adoption and traffic mix as central drivers of the quarter's mix quality.",
            source_id="fastly_q4_2025_transcript",
            page=None,
            score=0.9,
            bm25_rank=1,
            dense_rank=1,
            rerank_score=0.9,
            company="fastly",
            period="2025Q4",
            source_type="earnings_call_transcript",
            is_primary=True,
            trust_level=5,
            content_type="qualitative",
            metric_signals=["management_tone"],
        ),
    ]

    def fake_score(hypothesis, chunk_payload):
        base = 0.46 if "management commentary" in hypothesis.lower() else 0.04
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
                "source_type": chunk["source_type"],
                "is_primary": chunk["is_primary"],
                "contradiction": 0.02,
                "entailment": base,
                "neutral": 0.30,
            }
            for chunk in chunk_payload
        ]

    monkeypatch.setattr(controller.verifier, "score_hypothesis_against_chunks", fake_score)

    assessment = controller._assess_evidence(task, subquestion, chunks)

    assert assessment.sufficient is True
    assert assessment.mean_support >= 0.58


def test_batch_answer_generation_falls_back_to_stub_on_live_failure(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    controller.use_stub_llm = False
    task = controller.build_task(
        query="Summarize Fastly revenue quality",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What revenue facts are disclosed for Fastly in 2025Q4?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue",
        metric_family="revenue",
        needs_numeric_verification=True,
        status=ResearchQuestionStatus.COMPLETED,
    )
    result = ResearchQuestionResult(
        subquestion=subquestion,
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Fastly reported Q4 2025 revenue of $172.6 million.",
                source_id="fastly_q4_2025_results",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="quarterly_results",
                is_primary=True,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    def _boom(_batch):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(controller, "_generate_batch_answers", _boom)
    replay = ResearchReplayRecord(task=task)
    controller._generate_subquestion_answers(task, [result], {"used": 0, "max": task.llm_call_budget}, replay)

    assert result.answer_text


def test_render_structured_answer_keeps_only_valid_cited_claims():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What did Fastly report about revenue in Q4 2025?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="revenue",
            metric_family="revenue",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Fastly reported Q4 2025 revenue of $172.6 million.",
                source_id="fastly_q4_2025_results",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="quarterly_results",
                is_primary=True,
                trust_level=5,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    rendered = controller._render_structured_answer(
        result,
        {
            "question_id": "q1",
            "status": "answered",
            "claims": [
                {
                    "statement": "Fastly reported Q4 2025 revenue of $172.6 million",
                    "chunk_id": "c1",
                    "source_id": "fastly_q4_2025_results",
                },
                {
                    "statement": "This second claim cites a bad chunk",
                    "chunk_id": "missing",
                    "source_id": "fastly_q4_2025_results",
                },
            ],
        },
    )

    assert rendered == "- Fastly reported Q4 2025 revenue of $172.6 million. [Chunk: c1] [Source: fastly_q4_2025_results]"


def test_render_structured_answer_drops_secondary_numeric_hard_fact_claims():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What security revenue growth did the company highlight?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="security revenue growth",
            metric_family="revenue",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Investor deck: YoY Security Revenue Growth was 32%.",
                source_id="fastly_q4_2025_investor_presentation",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="investor_presentation",
                is_primary=False,
                trust_level=3,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    rendered = controller._render_structured_answer(
        result,
        {
            "question_id": "q1",
            "status": "answered",
            "claims": [
                {
                    "statement": "YoY Security Revenue Growth was 32%",
                    "chunk_id": "c1",
                    "source_id": "fastly_q4_2025_investor_presentation",
                }
            ],
            "insufficiency_reason": "Only the investor presentation disclosed this metric.",
        },
    )

    assert "Insufficient evidence" in rendered


def test_verify_answer_refuses_uncited_freeform_output():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What revenue facts are disclosed for Fastly in 2025Q4?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="revenue",
            metric_family="revenue",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[],
        answer_text="Fastly had a strong quarter without citing evidence.",
    )

    controller._verify_answer(result)

    assert result.status == ResearchQuestionStatus.REFUSED
    assert "did not produce any verifiable cited claims" in result.refusal_reason


def test_render_structured_answer_derives_atomic_hard_fact_from_chunk_blob():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is Fastly's disclosed revenue and growth for 2025Q4?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="revenue",
            metric_family="revenue",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "All News Fastly Announces Both Record Fourth Quarter and Full Year 2025 Financial Results "
                    "SAN FRANCISCO--(BUSINESS WIRE)-- Fastly, Inc. Fourth Quarter 2025 Financial Summary "
                    "Total revenue of $172.6 million, representing 23% year-over-year growth. "
                    "Network services revenue of $130.8 million, representing 19% year-over-year growth."
                ),
                source_id="fastly_q4_2025_results",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="quarterly_results",
                is_primary=True,
                trust_level=5,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    rendered = controller._render_structured_answer(
        result,
        {
            "question_id": "q1",
            "status": "answered",
            "claims": [
                {
                    "statement": "All News Fastly Announces Both Record Fourth Quarter and Full Year 2025 Financial Results SAN FRANCISCO--(BUSINESS WIRE)-- Fastly, Inc.",
                    "chunk_id": "c1",
                    "source_id": "fastly_q4_2025_results",
                }
            ],
        },
    )

    assert "Total revenue of $172.6 million, representing 23% year-over-year growth." in rendered
    assert "All News Fastly Announces" not in rendered


def test_rewrite_legacy_answer_derives_atomic_hard_fact_from_cited_lines():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is Fastly's disclosed revenue and growth for 2025Q4?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="revenue",
            metric_family="revenue",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "All News Fastly Announces Both Record Fourth Quarter and Full Year 2025 Financial Results "
                    "SAN FRANCISCO--(BUSINESS WIRE)-- Fastly, Inc. Fourth Quarter 2025 Financial Summary "
                    "Total revenue of $172.6 million, representing 23% year-over-year growth. "
                    "Network services revenue of $130.8 million, representing 19% year-over-year growth."
                ),
                source_id="fastly_q4_2025_results",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="quarterly_results",
                is_primary=True,
                trust_level=5,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    rewritten = controller._rewrite_legacy_answer(
        result,
        "- All News Fastly Announces Both Record Fourth Quarter and Full Year 2025 Financial Results SAN FRANCISCO--(BUSINESS WIRE)-- Fastly, Inc. [Chunk: c1] [Source: fastly_q4_2025_results]",
    )

    assert "Total revenue of $172.6 million, representing 23% year-over-year growth." in rewritten
    assert "All News Fastly Announces" not in rewritten


def test_batch_answer_generation_raises_in_strict_live_mode(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    controller.use_stub_llm = False
    monkeypatch.setattr("agent.research_controller.settings.strict_live_mode", True)
    task = controller.build_task(
        query="Summarize Fastly revenue quality",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What revenue facts are disclosed for Fastly in 2025Q4?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue",
        metric_family="revenue",
        needs_numeric_verification=True,
        status=ResearchQuestionStatus.COMPLETED,
    )
    result = ResearchQuestionResult(
        subquestion=subquestion,
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Fastly reported Q4 2025 revenue of $172.6 million.",
                source_id="fastly_q4_2025_results",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="fastly",
                period="2025Q4",
                source_type="quarterly_results",
                is_primary=True,
                metric_signals=["revenue"],
                content_type="quantitative",
            )
        ],
    )

    def _boom(_batch):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(controller, "_generate_batch_answers", _boom)
    replay = ResearchReplayRecord(task=task)

    import pytest

    with pytest.raises(RuntimeError, match="provider unavailable"):
        controller._generate_subquestion_answers(task, [result], {"used": 0, "max": task.llm_call_budget}, replay)


def test_priority_labels_from_live_decomposition_are_coerced():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="What changed from Q3 2025 to Q4 2025 in how management described revenue mix and growth drivers?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q3", "2025Q4"],
    )

    subquestions = controller._normalize_subquestions(
        task,
        [
            {
                "text": "Compare Q3 and Q4 management commentary on revenue mix.",
                "lane": "semantic",
                "priority": "high",
                "fact_slot": "Q3 vs Q4",
                "metric_family": "management_tone",
                "needs_numeric_verification": False,
            }
        ],
    )

    assert subquestions[0].priority == 1
