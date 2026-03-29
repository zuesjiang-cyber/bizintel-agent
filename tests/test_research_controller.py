from agent.orchestrator import BizIntelAgent
from agent.research_controller import ResearchController
from agent.schemas import AnalysisMode, ResearchLane
from retrieval.hybrid_retriever import HybridRetriever


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
