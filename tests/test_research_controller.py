from agent.orchestrator import BizIntelAgent
from agent.research_controller import ResearchController
from agent.schemas import AnalysisMode, EvidenceAssessment, ResearchLane, ResearchReplayRecord, ResearchQuestionResult, ResearchQuestionStatus, ResearchSubquestion, ResearchTask
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


def test_enforce_lane_routes_business_model_questions_to_semantic():
    controller = ResearchController(load_models=False, demo_mode=True)

    lane = controller._enforce_lane(
        "What are Fastly's main revenue sources and how does the company make money according to its business description?",
        "hard_fact",
    )

    assert lane == ResearchLane.SEMANTIC


def test_enforce_lane_does_not_treat_generic_what_is_business_model_as_numeric():
    controller = ResearchController(load_models=False, demo_mode=True)

    lane = controller._enforce_lane(
        "How does Fastly generate its revenue and what is its primary business model for making money?",
        "hard_fact",
    )

    assert lane == ResearchLane.SEMANTIC


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


def test_numeric_only_required_slots_are_not_injected_as_subquestions():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="Does Paypal have positive working capital based on FY2022 data?",
        mode=AnalysisMode.COMPANY,
        company_id="paypal",
        target_periods=["2022FY"],
        required_slots=["1.6", "2022", "Yes. Paypal has a positive working capital of $ 1.6Bn as of FY2022 end."],
    )

    subquestions = controller.plan_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    fact_slots = {item.fact_slot for item in subquestions}

    assert "1.6" not in fact_slots
    assert "2022" not in fact_slots
    assert "Yes. Paypal has a positive working capital of $ 1.6Bn as of FY2022 end." not in fact_slots


def test_executive_summary_is_built_from_verified_supported_content():
    controller = ResearchController(load_models=False, demo_mode=False)
    task = controller.build_task(
        query="Summarize Fastly revenue quality",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    completed = [
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q1",
                text="What revenue facts are disclosed for Fastly in 2025Q4?",
                lane=ResearchLane.HARD_FACT,
                priority=1,
                fact_slot="revenue",
                metric_family="revenue",
                needs_numeric_verification=True,
                status=ResearchQuestionStatus.COMPLETED,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            supported_content=(
                "- Fastly reported Q4 2025 revenue of $172.6 million. "
                "[Chunk: c1] [Source: fastly_q4_2025_results]"
            ),
        )
    ]

    summary = controller._build_executive_summary(task, completed, {"used": 0, "max": task.llm_call_budget})

    assert "[Chunk: c1]" in summary
    assert "[Source: fastly_q4_2025_results]" in summary
    assert "Fastly reported Q4 2025 revenue of $172.6 million." in summary


def test_executive_summary_prefers_query_focused_supported_answer_lines():
    controller = ResearchController(load_models=False, demo_mode=False)
    task = controller.build_task(
        query="Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
        mode=AnalysisMode.COMPANY,
        company_id="financebench_amd",
        target_periods=["2022FY"],
    )
    completed = [
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q1",
                text="What are the line items needed to assess AMD's quick ratio for FY 2022?",
                lane=ResearchLane.HARD_FACT,
                priority=1,
                fact_slot="quick_ratio_inputs",
                metric_family="financials",
                needs_numeric_verification=True,
                status=ResearchQuestionStatus.COMPLETED,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            supported_content="- Accounts receivable, net, were $3,353 million. [Chunk: c1] [Source: amd_2022_10k]",
        ),
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q2",
                text="What is AMD's quick ratio for FY 2022?",
                lane=ResearchLane.HARD_FACT,
                priority=2,
                fact_slot="quick_ratio_fy22",
                metric_family="financials",
                needs_numeric_verification=True,
                status=ResearchQuestionStatus.COMPLETED,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            supported_content="- AMD's quick ratio for FY2022 was 1.57, which indicates reasonably healthy short-term liquidity. [Chunk: c2] [Source: amd_2022_10k]",
        ),
    ]

    summary = controller._build_executive_summary(task, completed, {"used": 0, "max": task.llm_call_budget})
    first_line = next(line for line in summary.splitlines() if line.strip())

    assert "AMD's quick ratio for FY2022 was 1.57" in first_line


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


def test_management_commentary_questions_prefer_transcript_first_sources():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="What did management emphasize about revenue mix in Q4 2025?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q4"],
    )

    subquestions = controller.plan_subquestions(task, {"used": 0, "max": task.llm_call_budget})
    management_items = [
        item for item in subquestions
        if item.lane == ResearchLane.SEMANTIC and "management" in item.text.lower()
    ]

    assert management_items
    assert management_items[0].allowed_source_types[:3] == [
        "earnings_call_transcript",
        "shareholder_letter",
        "investor_presentation",
    ]


def test_overlapping_business_model_questions_are_collapsed():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="How does Fastly make money and what is its business model?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
    )

    subquestions = controller._normalize_subquestions(
        task,
        [
            {
                "text": "How does Fastly make money and what is its business model?",
                "lane": "semantic",
                "priority": 1,
                "fact_slot": "business model",
                "metric_family": "business_model",
            },
            {
                "text": "What are Fastly's primary revenue sources and overall business model for generating money?",
                "lane": "semantic",
                "priority": 2,
                "fact_slot": "fastly_revenue_model",
                "metric_family": "business_model",
            },
        ],
    )

    assert len(subquestions) == 1


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


def test_assess_evidence_accepts_compositional_liquidity_hard_fact(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
        required_source_types=["annual_report"],
        query_types=["numeric_grounding", "reasoned_analysis"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's quick ratio for FY 2022?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="quick_ratio_fy22",
        metric_family="financials",
        needs_numeric_verification=True,
    )
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Cash and cash equivalents were $5,912 million. Short-term investments were $1,879 million. Accounts receivable, net, were $3,353 million.",
            source_id="amd_2022_10k",
            page=None,
            score=1.0,
            bm25_rank=0,
            dense_rank=0,
            rerank_score=1.0,
            company="financebench_amd",
            period="2022FY",
            source_type="annual_report",
            is_primary=True,
            trust_level=5,
            metric_signals=["financials"],
            content_type="quantitative",
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Total current liabilities were $7,106 million as of December 31, 2022.",
            source_id="amd_2022_10k",
            page=None,
            score=0.9,
            bm25_rank=1,
            dense_rank=1,
            rerank_score=0.9,
            company="financebench_amd",
            period="2022FY",
            source_type="annual_report",
            is_primary=True,
            trust_level=5,
            metric_signals=["financials"],
            content_type="quantitative",
        ),
    ]

    def fake_score(hypothesis, chunk_payload):
        lowered = hypothesis.lower()
        entailment = 0.34 if any(
            token in lowered
            for token in (
                "quick ratio",
                "cash and cash equivalents",
                "accounts receivable",
                "current liabilities",
            )
        ) else 0.06
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
                "source_type": chunk["source_type"],
                "is_primary": chunk["is_primary"],
                "contradiction": 0.02,
                "entailment": entailment,
                "neutral": 0.30,
            }
            for chunk in chunk_payload
        ]

    monkeypatch.setattr(controller.verifier, "score_hypothesis_against_chunks", fake_score)

    assessment = controller._assess_evidence(task, subquestion, chunks)

    assert assessment.sufficient is True
    assert assessment.best_support >= 0.60
    assert assessment.matched_chunk_ids == ["c1", "c2"]


def test_build_assessment_hypotheses_adds_liquidity_component_phrasing():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's quick ratio for FY 2022?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="quick_ratio_fy22",
        metric_family="financials",
        needs_numeric_verification=True,
    )

    hypotheses = controller._build_assessment_hypotheses(task, subquestion)

    assert any("balance sheet line items" in hypothesis.lower() for hypothesis in hypotheses)
    assert any(
        "cash and cash equivalents" in hypothesis.lower() and "current liabilities" in hypothesis.lower()
        for hypothesis in hypotheses
    )


def test_liquidity_queries_add_balance_sheet_line_item_probes():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's quick ratio for FY 2022?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="quick_ratio_fy22",
        metric_family="financials",
        needs_numeric_verification=True,
    )

    initial_queries = controller._build_initial_queries(task, subquestion)
    followup_queries = controller._rule_followup_queries(
        task,
        subquestion,
        EvidenceAssessment(sufficient=False, reasons=["below threshold"]),
    )

    assert any(
        "cash and cash equivalents short-term investments accounts receivable current liabilities" in query.lower()
        for query in initial_queries
    )
    assert any("liquidity and capital resources" in query.lower() for query in followup_queries)


def test_formula_operand_queries_do_not_add_change_driver_noise():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query=(
            "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
            "Fixed asset turnover ratio is defined as: FY2019 revenue / "
            "(average PP&E between FY2018 and FY2019)."
        ),
        company_id="financebench_activision_blizzard",
        target_periods=["2019FY"],
    )
    subquestion = ResearchSubquestion(
        question_id="q2",
        text=(
            "What is Activision Blizzard's net property, plant, and equipment (PP&E) balance "
            "as of the end of FY2018 (December 31, 2018)?"
        ),
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="pp_and_e_fy2018",
        metric_family="pp_and_e",
        needs_numeric_verification=True,
    )

    queries = controller._build_initial_queries(task, subquestion)

    assert not any("change drivers" in query.lower() for query in queries)
    assert not any("increased decreased due to" in query.lower() for query in queries)
    assert any("pp_and_e" in query.lower() or "property, plant" in query.lower() for query in queries)


def test_numeric_value_subquestions_inside_change_tasks_stay_value_focused():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What drove operating margin change as of the FY22 for AMD?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What was AMD's operating margin in FY2022?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="operating_margin_fy2022",
        metric_family="operating_margin",
        needs_numeric_verification=True,
    )

    queries = controller._build_initial_queries(task, subquestion)

    assert not any("change drivers" in query.lower() for query in queries)
    assert not any("increased decreased due to" in query.lower() for query in queries)


def test_build_assessment_hypotheses_adds_segment_table_phrasing_for_ranking_questions():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="In 2022 Q2, which of JPM's business segments had the highest net income?",
        company_id="financebench_jpmorgan",
        target_periods=["2022Q2"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="In 2022 Q2, which of JPM's business segments had the highest net income?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="jpm_2022q2_highest_net_income_segment",
        metric_family="segment_net_income",
        needs_numeric_verification=True,
    )

    hypotheses = controller._build_assessment_hypotheses(task, subquestion)

    assert any("net income by segment" in hypothesis.lower() for hypothesis in hypotheses)
    assert any("table comparing segment-level net income values" in hypothesis.lower() for hypothesis in hypotheses)


def test_assess_evidence_accepts_segment_ranking_table_support(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="In 2022 Q2, which of JPM's business segments had the highest net income?",
        company_id="financebench_jpmorgan",
        target_periods=["2022Q2"],
        required_source_types=["quarterly_report"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="In 2022 Q2, which of JPM's business segments had the highest net income?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="jpm_2022q2_highest_net_income_segment",
        metric_family="segment_net_income",
        needs_numeric_verification=True,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "Segment results - managed basis. Three months ended June 30, 2022. "
            "Consumer & Community Banking net income was $3,100 million. "
            "Corporate & Investment Bank net income was $3,725 million. "
            "Commercial Banking net income was $994 million. "
            "Asset & Wealth Management net income was $1,004 million."
        ),
        source_id="jpmorgan_2022q2_10q",
        page=21,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="financebench_jpmorgan",
        period="2022Q2",
        source_type="quarterly_report",
        is_primary=True,
        trust_level=5,
        metric_signals=["segment_net_income"],
        content_type="quantitative",
    )

    def fake_score(hypothesis, chunk_payload):
        lowered = hypothesis.lower()
        entailment = 0.26 if any(
            phrase in lowered
            for phrase in (
                "net income by segment",
                "table comparing segment-level net income values",
                "multiple segment net income values",
            )
        ) else 0.04
        return [
            {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "text": item["text"],
                "source_type": item["source_type"],
                "is_primary": item["is_primary"],
                "contradiction": 0.01,
                "entailment": entailment,
                "neutral": 0.85,
            }
            for item in chunk_payload
        ]

    monkeypatch.setattr(controller.verifier, "score_hypothesis_against_chunks", fake_score)

    assessment = controller._assess_evidence(task, subquestion, [chunk])

    assert assessment.sufficient is True
    assert assessment.best_support >= 0.58
    assert assessment.matched_chunk_ids == ["c1"]


def test_build_assessment_hypotheses_uses_revenue_mix_specific_phrasing():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What was Fastly's revenue mix or breakdown by product, service, or segment in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q2",
        text="What was Fastly's revenue mix or breakdown by product, service, or segment in Q4 2025?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue_mix",
        metric_family="revenue",
        needs_numeric_verification=True,
    )

    hypotheses = controller._build_assessment_hypotheses(task, subquestion)

    assert any("revenue mix or breakdown" in hypothesis.lower() for hypothesis in hypotheses)
    assert any("network services revenue" in hypothesis.lower() for hypothesis in hypotheses)


def test_derive_atomic_claim_from_chunk_ignores_page_header_fragments():
    controller = ResearchController(load_models=False, demo_mode=True)
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What was AMD depreciation and amortization in 2015?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="depreciation and amortization",
        metric_family="depreciation",
        needs_numeric_verification=True,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "stock issued under stock-based compensation plans, net of tax withholding 16 — 5 (4) — — 1 "
            "Stock-based compensation — — 63 — — — 63 December 26, 2015 792 $ 8 $ 7,017 $ (123) "
            "$ (7,306) $ (8) $ (412) See accompanying notes to consolidated financial statements. "
            "57 [Page 60] Advanced Micro Devices, Inc. Consolidated Statements of Cash Flows Year Ended "
            "December 26, December 27, December 28, 2015 2014 2013 (In millions) Cash flows from "
            "operating activities: Net loss $ (660) $ (403) $ (83) Adjustments to reconcile net loss "
            "to net cash used in operating activities: Depreciation and amortization 167 203 236 "
            "Net loss on disposal of property, plant and equipment — — 31 Stock-based compensation expense 63 81 91"
        ),
        source_id="amd_2015_10k",
        page=60,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="advanced micro devices",
        period="2015FY",
        source_type="annual_report",
        is_primary=True,
        trust_level=5,
        metric_signals=["depreciation"],
        content_type="quantitative",
    )

    statement = controller._derive_atomic_claim_from_chunk(subquestion, chunk)

    assert statement == "Depreciation and amortization 167 203 236."
    assert "Advanced Micro Devices, Inc." not in statement


def test_income_statement_revenue_questions_include_net_revenue_alias_queries():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What is AMD's total revenue/net sales for FY2015 as reported in the P&L (income) statement?",
        company_id="financebench_amd",
        target_periods=["2015FY"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's total revenue/net sales for FY2015 as reported in the P&L (income) statement?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue_fy2015",
        metric_family="revenue",
        needs_numeric_verification=True,
    )

    initial_queries = controller._build_initial_queries(task, subquestion)
    followup_queries = controller._rule_followup_queries(
        task,
        subquestion,
        EvidenceAssessment(sufficient=False, reasons=["below threshold"]),
    )

    assert any("net revenue statement of operations" in query.lower() for query in initial_queries)
    assert any("total revenue income statement" in query.lower() for query in followup_queries)


def test_build_assessment_hypotheses_adds_management_revenue_mix_phrasing():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What did management emphasize about revenue mix in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q3",
        text="What did management emphasize as the most important revenue mix components in Q4 2025?",
        lane=ResearchLane.SEMANTIC,
        priority=1,
        fact_slot="management revenue mix emphasis",
        metric_family="management_tone",
        needs_numeric_verification=False,
    )

    hypotheses = controller._build_assessment_hypotheses(task, subquestion)

    assert any("management commentary" in hypothesis.lower() for hypothesis in hypotheses)
    assert any("cross-sell" in hypothesis.lower() or "network services" in hypothesis.lower() for hypothesis in hypotheses)
    assert any("balanced traffic mix" in hypothesis.lower() for hypothesis in hypotheses)


def test_initial_queries_humanize_fact_slot_and_add_revenue_mix_probe():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What was Fastly's revenue mix in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q2",
        text="What was Fastly's revenue mix in Q4 2025?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="q4_2025_revenue_mix",
        metric_family="revenue",
        needs_numeric_verification=True,
    )

    queries = controller._build_initial_queries(task, subquestion)

    assert any("q4 2025 revenue mix" in query.lower() for query in queries)
    assert any("network services revenue security revenue other revenue" in query.lower() for query in queries)


def test_query_coverage_proposals_restore_business_model_mix_and_management_slots():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="How does Fastly make money, and what does management emphasize as the most important revenue mix components in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )

    proposals = controller._ensure_query_coverage_proposals(
        task,
        [
            {
                "text": "What are Fastly's primary revenue sources and overall business model for generating money?",
                "lane": "semantic",
                "priority": 1,
                "fact_slot": "fastly_revenue_model",
                "metric_family": "business_model",
                "needs_numeric_verification": False,
            }
        ],
    )

    assert any(item["lane"] == "hard_fact" and "revenue_mix" in str(item["fact_slot"]) for item in proposals)


def test_query_coverage_proposals_add_product_service_catalog_question():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What are the major products and services that AMD sells as of FY22?",
        company_id="amd",
        target_periods=["FY22"],
    )

    proposals = controller._ensure_query_coverage_proposals(task, [])

    assert any(
        item["lane"] == "semantic"
        and "products, platforms, and services" in item["text"].lower()
        for item in proposals
    )


def test_build_initial_queries_add_product_service_catalog_searches():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What are the major products and services that AMD sells as of FY22?",
        company_id="amd",
        target_periods=["FY22"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What major products, platforms, and services does amd explicitly say it sells in FY22?",
        lane=ResearchLane.SEMANTIC,
        priority=1,
        fact_slot="amd_products_and_services",
        metric_family="business_model",
        needs_numeric_verification=False,
    )

    queries = controller._build_initial_queries(task, subquestion)

    assert any("products services offerings annual report" in query.lower() for query in queries)
    assert any("what does amd sell" in query.lower() for query in queries)


def test_required_slot_products_and_services_stay_semantic():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What are the major products and services that AMD sells as of FY22?",
        company_id="amd",
        target_periods=["FY22"],
    )

    proposal = controller._proposal_for_required_slot(
        task,
        "major products and services",
        priority=1,
    )

    assert proposal["lane"] == "semantic"
    assert proposal["metric_family"] == "business_model"
    assert proposal["needs_numeric_verification"] is False


def test_enforce_lane_converts_product_service_listing_to_semantic():
    controller = ResearchController(load_models=False, demo_mode=True)

    lane = controller._enforce_lane("What were AMD's major products sold during FY2022?", "hard_fact")

    assert lane == ResearchLane.SEMANTIC


def test_query_coverage_proposals_add_customer_concentration_question():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="Did AMD report customer concentration in FY22?",
        company_id="amd",
        target_periods=["FY22"],
    )

    proposals = controller._ensure_query_coverage_proposals(task, [])

    assert any(
        item["lane"] == "hard_fact"
        and "customer concentration" in item["text"].lower()
        for item in proposals
    )


def test_build_initial_queries_add_customer_concentration_searches():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="Did AMD report customer concentration in FY22?",
        company_id="amd",
        target_periods=["FY22"],
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="Did amd disclose customer concentration in FY22, and if so what percentage of revenue did the customer represent?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="amd_customer_concentration",
        metric_family="revenue",
        needs_numeric_verification=True,
    )

    queries = controller._build_initial_queries(task, subquestion)

    assert any("customer concentration major customer percentage revenue" in query.lower() for query in queries)
    assert any("significant customer 10 percent revenue" in query.lower() for query in queries)


def test_revenue_mix_bonus_lifts_explicit_component_breakout_to_hard_fact_support(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query="What was Fastly's revenue mix in Q4 2025?",
        company_id="fastly",
        target_periods=["2025Q4"],
    )
    subquestion = ResearchSubquestion(
        question_id="q4",
        text="What was Fastly's revenue mix in Q4 2025?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="2025q4_revenue_mix",
        metric_family="revenue",
        needs_numeric_verification=True,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "Network services revenue of $130.8 million. "
            "Security revenue of $35.4 million. "
            "Other revenue of $6.4 million."
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
        content_type="quantitative",
        metric_signals=["revenue"],
    )

    def fake_score(_hypothesis, chunk_payload):
        return [
            {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "text": item["text"],
                "source_type": item["source_type"],
                "is_primary": item["is_primary"],
                "contradiction": 0.01,
                "entailment": 0.04,
                "neutral": 0.90,
            }
            for item in chunk_payload
        ]

    monkeypatch.setattr(controller.verifier, "score_hypothesis_against_chunks", fake_score)

    assessment = controller._assess_evidence(task, subquestion, [chunk])

    assert assessment.sufficient is True
    assert assessment.best_support >= 0.78


def test_derive_atomic_claim_from_chunk_keeps_full_revenue_mix_amount():
    controller = ResearchController(load_models=False, demo_mode=True)
    subquestion = ResearchSubquestion(
        question_id="q2",
        text="What was Fastly's revenue mix in Q4 2025?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="q4_2025_revenue_mix",
        metric_family="revenue",
        needs_numeric_verification=True,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="Network services revenue of $130.8 million, representing 19% year-over-year growth. Security revenue of $35.4 million.",
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
        content_type="quantitative",
        metric_signals=["revenue"],
    )

    statement = controller._derive_atomic_claim_from_chunk(subquestion, chunk)

    assert "$130.8 million" in statement
    assert "19% year-over-year growth" in statement


def test_render_structured_answer_falls_back_to_evidence_when_model_marks_insufficient():
    controller = ResearchController(load_models=False, demo_mode=True)
    subquestion = ResearchSubquestion(
        question_id="q3",
        text="What did management emphasize about revenue mix in Q4 2025?",
        lane=ResearchLane.SEMANTIC,
        priority=1,
        fact_slot="q4_2025_management_revenue_emphasis",
        metric_family="management_tone",
        needs_numeric_verification=False,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "These efforts were reflected in balanced revenue growth across product lines, geographic regions, "
            "and customer segments in 2025, positioning us to drive continued growth in 2026."
        ),
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
    )
    result = ResearchQuestionResult(
        subquestion=subquestion,
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[chunk],
        assessments=[
            EvidenceAssessment(
                valid_chunk_count=1,
                best_entailment=0.8,
                mean_entailment=0.8,
                best_support=0.8,
                mean_support=0.8,
                metric_match=True,
                high_trust_hit=True,
                sufficient=True,
                matched_chunk_ids=["c1"],
            )
        ],
    )

    rendered = controller._render_structured_answer(
        result,
        {"question_id": "q3", "status": "insufficient", "claims": [], "insufficiency_reason": "model said insufficient"},
    )

    assert "balanced revenue growth across product lines" in rendered
    assert "[Chunk: c1]" in rendered


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


def test_batch_answer_missing_structured_payload_is_not_silently_stubbed(monkeypatch):
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

    def _empty(_task, _batch):
        return None

    monkeypatch.setattr(controller, "_generate_batch_answers", _empty)
    replay = ResearchReplayRecord(task=task)
    controller._generate_subquestion_answers(task, [result], {"used": 0, "max": task.llm_call_budget}, replay)

    assert result.answer_text == (
        "- Fastly reported Q4 2025 revenue of $172.6 million. "
        "[Chunk: c1] [Source: fastly_q4_2025_results]"
    )
    assert any(item["status"] == "missing_structured_answer" for item in result.trace)
    assert any(item["status"] == "fallback_answer_used" for item in result.trace)
    assert any(
        decision.decision_type == "answer_generation_contract_failure"
        for decision in replay.decisions
    )
    assert any(
        decision.decision_type == "answer_generated"
        and decision.payload.get("recovered_from_evidence") is True
        for decision in replay.decisions
    )


def test_batch_answer_prompt_requires_direct_question_shaped_hard_fact_claims():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = controller.build_task(
        query="Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
        mode=AnalysisMode.COMPANY,
        company_id="financebench_amd",
        target_periods=["2022FY"],
    )
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is AMD's quick ratio for FY 2022?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="quick_ratio_fy22",
            metric_family="financials",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Cash and cash equivalents were $4,835 million. Short-term investments were $1,020 million. Accounts receivable, net, were $4,126 million. Total current liabilities were $6,369 million.",
                source_id="amd_2022_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_amd",
                period="2022FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["financials"],
                content_type="quantitative",
            )
        ],
    )

    prompt = controller._build_batch_answer_prompt(task, [result])

    assert "claim 1 must answer the question directly" in prompt
    assert "quick ratio was 1.57" in prompt
    assert "chunk_ids" in prompt
    assert "Do not copy raw table fragments" in prompt


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


def test_render_structured_answer_replaces_off_topic_numeric_fragment_with_line_item_answer():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="capital_expenditures",
            metric_family="capital_expenditures",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "Foreign exchange had a positive impact of $102 million on revenue. "
                    "Capital expenditures (1,577)."
                ),
                source_id="three_m_2018_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_3m",
                period="2018FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["capital_expenditures"],
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
                    "statement": "Foreign exchange had a positive impact of $102 million on revenue.",
                    "chunk_id": "c1",
                    "source_id": "three_m_2018_10k",
                }
            ],
        },
    )

    assert "Capital expenditures were $1,577 million." in rendered
    assert "positive impact of $102 million on revenue" not in rendered


def test_render_structured_answer_keeps_direct_computed_hard_fact_statement():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is AMD's quick ratio for FY 2022?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="quick_ratio_fy22",
            metric_family="financials",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Cash and cash equivalents were $5,912 million. Short-term investments were $1,879 million. Accounts receivable, net, were $3,353 million. Total current liabilities were $7,106 million.",
                source_id="amd_2022_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_amd",
                period="2022FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["financials"],
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
                    "statement": "AMD's quick ratio for FY2022 was 1.57, which indicates reasonably healthy short-term liquidity.",
                    "chunk_id": "c1",
                    "source_id": "amd_2022_10k",
                }
            ],
        },
    )

    assert "AMD's quick ratio for FY2022 was 1.57" in rendered
    assert "Cash and cash equivalents were $5,912 million" not in rendered


def test_render_structured_answer_keeps_multi_chunk_cited_direct_statement():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="In 2022 Q2, which of JPM's business segments had the highest net income?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="jpm_2022q2_highest_net_income_segment",
            metric_family="segment_net_income",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Consumer & Community Banking net income was $3,100 million. Corporate & Investment Bank net income was $3,725 million.",
                source_id="jpmorgan_2022q2_10q",
                page=21,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_jpmorgan",
                period="2022Q2",
                source_type="quarterly_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["segment_net_income"],
                content_type="quantitative",
            ),
            RetrievedChunk(
                chunk_id="c2",
                text="Commercial Banking net income was $994 million. Asset & Wealth Management net income was $1,004 million.",
                source_id="jpmorgan_2022q2_10q",
                page=21,
                score=0.9,
                bm25_rank=1,
                dense_rank=1,
                rerank_score=0.9,
                company="financebench_jpmorgan",
                period="2022Q2",
                source_type="quarterly_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["segment_net_income"],
                content_type="quantitative",
            ),
        ],
    )

    rendered = controller._render_structured_answer(
        result,
        {
            "question_id": "q1",
            "status": "answered",
            "claims": [
                {
                    "statement": "Corporate & Investment Bank had the highest net income at $3,725 million.",
                    "chunk_ids": ["c1", "c2"],
                    "source_ids": ["jpmorgan_2022q2_10q", "jpmorgan_2022q2_10q"],
                }
            ],
        },
    )

    assert "Corporate & Investment Bank had the highest net income at $3,725 million." in rendered
    assert "[Chunk: c1] [Source: jpmorgan_2022q2_10q]" in rendered
    assert "[Chunk: c2] [Source: jpmorgan_2022q2_10q]" in rendered


def test_render_structured_answer_falls_back_to_direct_quick_ratio_when_structured_claims_are_invalid():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is AMD's quick ratio for FY 2022?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="quick_ratio_fy22",
            metric_family="financials",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Cash and cash equivalents were $5,912 million. Short-term investments were $1,879 million. Accounts receivable, net, were $3,353 million. Total current liabilities were $7,106 million.",
                source_id="amd_2022_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_amd",
                period="2022FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["financials"],
                content_type="quantitative",
            )
        ],
        assessments=[
            EvidenceAssessment(
                sufficient=True,
                matched_chunk_ids=["c1"],
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
                    "statement": "AMD's quick ratio for FY2022 was 1.57.",
                    "chunk_id": "missing",
                    "source_id": "amd_2022_10k",
                }
            ],
        },
    )

    assert "quick ratio was 1.57" in rendered
    assert "[Chunk: c1]" in rendered


def test_render_structured_answer_falls_back_to_direct_ranking_answer_when_structured_claims_are_invalid():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="In 2022 Q2, which of JPM's business segments had the highest net income?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="jpm_2022q2_highest_net_income_segment",
            metric_family="segment_net_income",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "Segment results - managed basis. Three months ended June 30, 2022. "
                    "Consumer & Community Banking net income was $3,100 million. "
                    "Corporate & Investment Bank net income was $3,725 million. "
                    "Commercial Banking net income was $994 million. "
                    "Asset & Wealth Management net income was $1,004 million."
                ),
                source_id="jpmorgan_2022q2_10q",
                page=21,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_jpmorgan",
                period="2022Q2",
                source_type="quarterly_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["segment_net_income"],
                content_type="quantitative",
            )
        ],
        assessments=[
            EvidenceAssessment(
                sufficient=True,
                matched_chunk_ids=["c1"],
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
                    "statement": "Corporate & Investment Bank had the highest net income.",
                    "chunk_id": "bad",
                    "source_id": "jpmorgan_2022q2_10q",
                }
            ],
        },
    )

    assert "Corporate & Investment Bank had the highest net income at $3,725 million." in rendered
    assert "[Chunk: c1]" in rendered


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


def test_verify_answer_recovers_direct_line_item_answer_after_verification_failure():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="capital_expenditures",
            metric_family="capital_expenditures",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Capital expenditures (1,577).",
                source_id="three_m_2018_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_3m",
                period="2018FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["capital_expenditures"],
                content_type="quantitative",
            )
        ],
        answer_text="- Revenue growth was strong. [Chunk: c1] [Source: three_m_2018_10k]",
    )

    controller._verify_answer(result)

    assert result.status == ResearchQuestionStatus.COMPLETED
    assert "Capital expenditures were $1,577 million." in result.supported_content
    assert any(event["stage"] == "verification_recovery" for event in result.trace)


def test_execute_subquestion_llm_gray_zone_review_can_promote_near_threshold_hard_fact(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    controller.use_stub_llm = False
    controller.client = object()
    task = ResearchTask(
        query="What is AMD's quick ratio for FY 2022?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
        max_followup_rounds=0,
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's quick ratio for FY 2022?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="quick_ratio_fy22",
        metric_family="financials",
        needs_numeric_verification=True,
        max_rounds=0,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="Cash and cash equivalents were $5,912 million. Short-term investments were $1,879 million. Accounts receivable, net, were $3,353 million. Total current liabilities were $7,106 million.",
        source_id="amd_2022_10k",
        page=None,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="financebench_amd",
        period="2022FY",
        source_type="annual_report",
        is_primary=True,
        trust_level=5,
        metric_signals=["financials"],
        content_type="quantitative",
    )

    monkeypatch.setattr(controller, "_retrieve_queries", lambda *args, **kwargs: [chunk])
    monkeypatch.setattr(
        controller,
        "_assess_evidence",
        lambda *args, **kwargs: EvidenceAssessment(
            valid_chunk_count=1,
            best_entailment=0.52,
            mean_entailment=0.52,
            best_support=0.52,
            mean_support=0.52,
            numeric_match=True,
            metric_match=True,
            high_trust_hit=True,
            sufficient=False,
            insufficient=True,
            reasons=["Best support 0.52 below hard-fact threshold."],
            matched_chunk_ids=["c1"],
        ),
    )
    monkeypatch.setattr(
        "agent.research_controller.generate_text_response",
        lambda *args, **kwargs: '{"verdict":"sufficient","rationale":"The filing contains the line items needed to compute the quick ratio directly."}',
    )

    replay = ResearchReplayRecord(task=task)
    llm_budget = {"used": 0, "max": 1}
    result = controller._execute_subquestion(
        task=task,
        subquestion=subquestion,
        retriever=object(),
        llm_budget=llm_budget,
        replay=replay,
    )

    assert result.status == ResearchQuestionStatus.COMPLETED
    assert result.assessments[-1].sufficient is True
    assert llm_budget["used"] == 1
    assert any(decision.decision_type == "llm_gray_zone_review" for decision in replay.decisions)
    assert any(
        decision.decision_type == "complete"
        and decision.reason == "gray-zone evidence review accepted the evidence as sufficient"
        for decision in replay.decisions
    )


def test_execute_subquestion_llm_gray_zone_review_does_not_override_hard_blockers(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    controller.use_stub_llm = False
    controller.client = object()
    task = ResearchTask(
        query="What is AMD's total revenue for FY 2015?",
        company_id="financebench_amd",
        target_periods=["2015FY"],
        max_followup_rounds=0,
    )
    subquestion = ResearchSubquestion(
        question_id="q1",
        text="What is AMD's total revenue for FY 2015?",
        lane=ResearchLane.HARD_FACT,
        priority=1,
        fact_slot="revenue_fy2015",
        metric_family="revenue",
        needs_numeric_verification=True,
        max_rounds=0,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="AMD discussed demand trends.",
        source_id="amd_2015_10k",
        page=None,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="financebench_amd",
        period="2015FY",
        source_type="annual_report",
        is_primary=True,
        trust_level=5,
        metric_signals=["narrative"],
        content_type="qualitative",
    )

    monkeypatch.setattr(controller, "_retrieve_queries", lambda *args, **kwargs: [chunk])
    monkeypatch.setattr(
        controller,
        "_assess_evidence",
        lambda *args, **kwargs: EvidenceAssessment(
            valid_chunk_count=1,
            best_entailment=0.60,
            mean_entailment=0.60,
            best_support=0.60,
            mean_support=0.60,
            numeric_match=False,
            metric_match=False,
            high_trust_hit=True,
            sufficient=False,
            insufficient=True,
            reasons=[
                "Target metric or numeric evidence was not found in retrieved chunks.",
                "Best support 0.60 below hard-fact threshold.",
            ],
            matched_chunk_ids=["c1"],
        ),
    )
    calls = {"count": 0}

    def _should_not_run(*args, **kwargs):
        calls["count"] += 1
        return '{"verdict":"sufficient","rationale":"Should never be used."}'

    monkeypatch.setattr("agent.research_controller.generate_text_response", _should_not_run)

    replay = ResearchReplayRecord(task=task)
    llm_budget = {"used": 0, "max": 1}
    result = controller._execute_subquestion(
        task=task,
        subquestion=subquestion,
        retriever=object(),
        llm_budget=llm_budget,
        replay=replay,
    )

    assert result.status == ResearchQuestionStatus.REFUSED
    assert calls["count"] == 0
    assert llm_budget["used"] == 0


def test_execute_subquestion_llm_gray_zone_review_can_promote_semantic_reasoning(monkeypatch):
    controller = ResearchController(load_models=False, demo_mode=True)
    controller.use_stub_llm = False
    controller.client = object()
    task = ResearchTask(
        query="What drove revenue change as of the FY22 for AMD?",
        company_id="financebench_amd",
        target_periods=["2022FY"],
        max_followup_rounds=0,
    )
    subquestion = ResearchSubquestion(
        question_id="q4",
        text="What were the primary drivers of the revenue change for AMD in FY 2022?",
        lane=ResearchLane.SEMANTIC,
        priority=1,
        fact_slot="revenue_change_drivers",
        metric_family="revenue",
        needs_numeric_verification=False,
        max_rounds=0,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="Revenue growth was primarily driven by the inclusion of Xilinx embedded product revenue following the acquisition in February 2022.",
        source_id="amd_2022_10k",
        page=None,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
        company="financebench_amd",
        period="2022FY",
        source_type="annual_report",
        is_primary=True,
        trust_level=5,
        metric_signals=["revenue"],
        content_type="mixed",
    )

    monkeypatch.setattr(controller, "_retrieve_queries", lambda *args, **kwargs: [chunk])
    monkeypatch.setattr(
        controller,
        "_assess_evidence",
        lambda *args, **kwargs: EvidenceAssessment(
            valid_chunk_count=1,
            best_entailment=0.52,
            mean_entailment=0.52,
            best_support=0.52,
            mean_support=0.52,
            numeric_match=False,
            metric_match=True,
            high_trust_hit=True,
            sufficient=False,
            insufficient=True,
            reasons=["Mean support 0.52 below semantic threshold."],
            matched_chunk_ids=["c1"],
        ),
    )
    monkeypatch.setattr(
        "agent.research_controller.generate_text_response",
        lambda *args, **kwargs: '{"verdict":"sufficient","rationale":"The filing directly states the main revenue driver for FY2022."}',
    )

    replay = ResearchReplayRecord(task=task)
    llm_budget = {"used": 0, "max": 1}
    result = controller._execute_subquestion(
        task=task,
        subquestion=subquestion,
        retriever=object(),
        llm_budget=llm_budget,
        replay=replay,
    )

    assert result.status == ResearchQuestionStatus.COMPLETED
    assert result.assessments[-1].sufficient is True
    assert any(decision.decision_type == "llm_gray_zone_review" for decision in replay.decisions)


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


def test_render_structured_answer_derives_atomic_semantic_financial_claim_from_chunk():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What are Fastly's primary revenue sources?",
            lane=ResearchLane.SEMANTIC,
            priority=1,
            fact_slot="revenue sources",
            metric_family="business_model",
            needs_numeric_verification=False,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "Our revenue model is primarily based on customer consumption, which can lead to variability in our quarterly results. "
                    "Network services revenue of $130.8 million, representing 19% year-over-year growth."
                ),
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
                metric_signals=["revenue"],
                content_type="mixed",
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
                    "statement": "Fastly's revenue model is primarily based on customer consumption.",
                    "chunk_id": "c1",
                    "source_id": "fastly_q4_2025_transcript",
                }
            ],
        },
    )

    assert "Our revenue model is primarily based on customer consumption" in rendered
    assert "Fastly's revenue model is primarily based on customer consumption" not in rendered


def test_build_executive_summary_synthesizes_fixed_asset_turnover_from_completed_operands():
    controller = ResearchController(load_models=False, demo_mode=True)
    task = ResearchTask(
        query=(
            "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? "
            "Fixed asset turnover ratio is defined as: FY2019 revenue / "
            "(average PP&E between FY2018 and FY2019)."
        ),
        company_id="financebench_activision_blizzard",
        target_periods=["2019FY"],
    )
    completed = [
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q1",
                text="What is Activision Blizzard's total revenue for FY2019?",
                lane=ResearchLane.HARD_FACT,
                priority=1,
                fact_slot="revenue_fy2019",
                metric_family="revenue",
                needs_numeric_verification=True,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            evidence=[
                RetrievedChunk(
                    chunk_id="c1",
                    text="Total revenue was $7,500 million.",
                    source_id="activisionblizzard_2019_10k",
                    page=None,
                    score=1.0,
                    bm25_rank=0,
                    dense_rank=0,
                    rerank_score=1.0,
                    company="financebench_activision_blizzard",
                    period="2019FY",
                    source_type="annual_report",
                    is_primary=True,
                    trust_level=5,
                    metric_signals=["revenue"],
                    content_type="quantitative",
                )
            ],
            supported_content="- Revenue was $7,500 million. [Chunk: c1] [Source: activisionblizzard_2019_10k]",
        ),
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q2",
                text="What is Activision Blizzard's net property, plant, and equipment (PP&E) balance as of the end of FY2018?",
                lane=ResearchLane.HARD_FACT,
                priority=1,
                fact_slot="pp_and_e_fy2018",
                metric_family="pp_and_e",
                needs_numeric_verification=True,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            evidence=[
                RetrievedChunk(
                    chunk_id="c2",
                    text="Property, plant and equipment, net was $180 million.",
                    source_id="activisionblizzard_2019_10k",
                    page=None,
                    score=1.0,
                    bm25_rank=0,
                    dense_rank=0,
                    rerank_score=1.0,
                    company="financebench_activision_blizzard",
                    period="2019FY",
                    source_type="annual_report",
                    is_primary=True,
                    trust_level=5,
                    metric_signals=["pp_and_e"],
                    content_type="quantitative",
                )
            ],
            supported_content="- Net property, plant and equipment was $180 million. [Chunk: c2] [Source: activisionblizzard_2019_10k]",
        ),
        ResearchQuestionResult(
            subquestion=ResearchSubquestion(
                question_id="q3",
                text="What is Activision Blizzard's net property, plant, and equipment (PP&E) balance as of the end of FY2019?",
                lane=ResearchLane.HARD_FACT,
                priority=1,
                fact_slot="pp_and_e_fy2019",
                metric_family="pp_and_e",
                needs_numeric_verification=True,
            ),
            status=ResearchQuestionStatus.COMPLETED,
            evidence=[
                RetrievedChunk(
                    chunk_id="c3",
                    text="Property, plant and equipment, net was $220 million.",
                    source_id="activisionblizzard_2019_10k",
                    page=None,
                    score=1.0,
                    bm25_rank=0,
                    dense_rank=0,
                    rerank_score=1.0,
                    company="financebench_activision_blizzard",
                    period="2019FY",
                    source_type="annual_report",
                    is_primary=True,
                    trust_level=5,
                    metric_signals=["pp_and_e"],
                    content_type="quantitative",
                )
            ],
            supported_content="- Net property, plant and equipment was $220 million. [Chunk: c3] [Source: activisionblizzard_2019_10k]",
        ),
    ]

    summary = controller._build_executive_summary(task, completed, {"used": 0, "max": task.llm_call_budget})

    assert "fixed asset turnover ratio was 37.5" in summary.lower()
    assert "[Chunk: c1] [Source: activisionblizzard_2019_10k]" in summary


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


def test_rewrite_legacy_answer_keeps_direct_hard_fact_statement():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="What is AMD's quick ratio for FY 2022?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="quick_ratio_fy22",
            metric_family="financials",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Cash and cash equivalents were $5,912 million. Short-term investments were $1,879 million. Accounts receivable, net, were $3,353 million. Total current liabilities were $7,106 million.",
                source_id="amd_2022_10k",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_amd",
                period="2022FY",
                source_type="annual_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["financials"],
                content_type="quantitative",
            )
        ],
    )

    rewritten = controller._rewrite_legacy_answer(
        result,
        "- AMD's quick ratio for FY2022 was 1.57, which indicates reasonably healthy short-term liquidity. [Chunk: c1] [Source: amd_2022_10k]",
    )

    assert "AMD's quick ratio for FY2022 was 1.57" in rewritten
    assert "Cash and cash equivalents were $5,912 million" not in rewritten


def test_rewrite_legacy_answer_keeps_multi_chunk_direct_statement():
    controller = ResearchController(load_models=False, demo_mode=True)
    result = ResearchQuestionResult(
        subquestion=ResearchSubquestion(
            question_id="q1",
            text="In 2022 Q2, which of JPM's business segments had the highest net income?",
            lane=ResearchLane.HARD_FACT,
            priority=1,
            fact_slot="jpm_2022q2_highest_net_income_segment",
            metric_family="segment_net_income",
            needs_numeric_verification=True,
        ),
        status=ResearchQuestionStatus.COMPLETED,
        evidence=[
            RetrievedChunk(
                chunk_id="c1",
                text="Consumer & Community Banking net income was $3,100 million. Corporate & Investment Bank net income was $3,725 million.",
                source_id="jpmorgan_2022q2_10q",
                page=21,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
                company="financebench_jpmorgan",
                period="2022Q2",
                source_type="quarterly_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["segment_net_income"],
                content_type="quantitative",
            ),
            RetrievedChunk(
                chunk_id="c2",
                text="Commercial Banking net income was $994 million. Asset & Wealth Management net income was $1,004 million.",
                source_id="jpmorgan_2022q2_10q",
                page=21,
                score=0.9,
                bm25_rank=1,
                dense_rank=1,
                rerank_score=0.9,
                company="financebench_jpmorgan",
                period="2022Q2",
                source_type="quarterly_report",
                is_primary=True,
                trust_level=5,
                metric_signals=["segment_net_income"],
                content_type="quantitative",
            ),
        ],
    )

    rewritten = controller._rewrite_legacy_answer(
        result,
        "- Corporate & Investment Bank had the highest net income at $3,725 million. [Chunk: c1] [Source: jpmorgan_2022q2_10q] [Chunk: c2] [Source: jpmorgan_2022q2_10q]",
    )

    assert "Corporate & Investment Bank had the highest net income at $3,725 million." in rewritten
    assert "[Chunk: c1] [Source: jpmorgan_2022q2_10q]" in rewritten
    assert "[Chunk: c2] [Source: jpmorgan_2022q2_10q]" in rewritten


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

    def _boom(_task, _batch):
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
