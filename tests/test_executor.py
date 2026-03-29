from agent.config import settings
from agent.executor import AnalysisExecutor
from agent.schemas import AnalysisMode, AnalysisPlan, AnalysisStep, RetrievedChunk


class FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def retrieve_with_trace(self, query, top_k=10, mode="full_hybrid", filters=None, strategy=None):
        self.calls.append({"query": query, "filters": filters, "strategy": strategy})
        chunks = self.mapping.get(query, [])
        return chunks, {"query": query, "returned": len(chunks), "filters": filters, "strategy": strategy}


def _chunk(chunk_id: str, text: str, source_id: str = "src1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_id=source_id,
        page=None,
        score=1.0,
        bm25_rank=0,
        dense_rank=0,
        rerank_score=1.0,
    )


def test_executor_runs_gap_filling_when_required_fact_is_missing(monkeypatch):
    step = AnalysisStep(
        name="financial_analysis",
        description="Financial metrics.",
        required=True,
        search_queries=["base revenue query"],
        evidence_requirements={
            "required_facts": ["revenue", "profitability"],
            "fact_first": True,
            "writing_order": ["facts", "management_commentary", "inference"],
        },
    )
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Assess Fastly quarter", steps=[step], contract={"sections": {}})
    retriever = FakeRetriever(
        {
            "base revenue query": [_chunk("c1", "Fastly revenue was $100 million. [Source: fastly_q4]")],
            "Assess Fastly quarter financial analysis profitability": [
                _chunk("c2", "Fastly profitability improved and adjusted EBITDA was positive.", "fastly_q4")
            ],
        }
    )
    executor = AnalysisExecutor(retriever)
    monkeypatch.setattr(settings, "retrieval_gap_max_rounds", 3)
    monkeypatch.setattr(
        executor,
        "_analyze_with_llm",
        lambda **kwargs: "Draft from notes.",
    )

    result = executor.execute_plan(plan)["step_outputs"]["financial_analysis"]

    assert "revenue" in result["covered_facts"]
    assert "profitability" in result["covered_facts"]
    assert result["missing_facts"] == []
    assert len(result["gap_iterations"]) == 2
    assert result["evidence_ledger"]
    assert result["gap_reflection"]["confidence_assessments"]
    assert result["query_contracts"] == []


def test_executor_stops_gap_loop_when_no_new_evidence(monkeypatch):
    step = AnalysisStep(
        name="financial_analysis",
        description="Financial metrics.",
        required=True,
        search_queries=["base revenue query"],
        evidence_requirements={
            "required_facts": ["revenue", "profitability"],
            "fact_first": True,
        },
    )
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Assess Fastly quarter", steps=[step], contract={"sections": {}})
    retriever = FakeRetriever(
        {
            "base revenue query": [_chunk("c1", "Fastly revenue was $100 million.", "fastly_q4")],
            "Assess Fastly quarter financial analysis profitability": [_chunk("c1", "Fastly revenue was $100 million.", "fastly_q4")],
        }
    )
    executor = AnalysisExecutor(retriever)
    monkeypatch.setattr(settings, "retrieval_gap_max_rounds", 3)
    monkeypatch.setattr(
        executor,
        "_analyze_with_llm",
        lambda **kwargs: "Draft from notes.",
    )

    result = executor.execute_plan(plan)["step_outputs"]["financial_analysis"]

    assert "revenue" in result["covered_facts"]
    assert "profitability" in result["missing_facts"]
    assert result["gap_iterations"][-1]["new_chunks"] == 0
    assert len(result["gap_iterations"]) == 2
    assert result["gap_reflection"]["should_continue"] is True


def test_executor_uses_deterministic_gap_reflection_without_llm(monkeypatch):
    step = AnalysisStep(
        name="business_model",
        description="Business model.",
        required=True,
        search_queries=["base query"],
        evidence_requirements={
            "required_facts": ["core products", "monetization"],
            "allowed_source_types": ["company_profile"],
        },
    )
    executor = AnalysisExecutor(FakeRetriever({"base query": []}))
    monkeypatch.setattr(settings, "llm_mode", "live")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
    executor.use_stub_llm = False

    reflection = executor._build_gap_reflection(
        user_query="Analyze Cloudflare",
        step=step,
        missing_facts=["monetization"],
        evidence_ledger=[],
    )

    assert reflection["should_continue"] is True
    assert reflection["additional_queries"]


def test_executor_evidence_notes_include_chunk_excerpt(monkeypatch):
    step = AnalysisStep(
        name="business_model",
        description="Business model.",
        required=True,
        search_queries=["base query"],
        evidence_requirements={
            "required_facts": ["core products"],
        },
    )
    plan = AnalysisPlan(mode=AnalysisMode.COMPANY, user_query="Analyze Cloudflare", steps=[step], contract={"sections": {}})
    retriever = FakeRetriever(
        {
            "base query": [
                _chunk(
                    "c1",
                    "Cloudflare sells application security, network security, and developer platform products to enterprise customers.",
                    "cloudflare_profile",
                )
            ]
        }
    )
    executor = AnalysisExecutor(retriever)
    monkeypatch.setattr(
        executor,
        "_analyze_with_llm",
        lambda **kwargs: "Draft from notes.",
    )

    result = executor.execute_plan(plan)["step_outputs"]["business_model"]

    assert result["evidence_notes"]
    assert "evidence_text" in result["evidence_notes"][0]
    assert "Cloudflare sells application security" in result["evidence_notes"][0]["evidence_text"]
