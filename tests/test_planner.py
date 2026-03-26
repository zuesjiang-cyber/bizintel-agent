import pytest

from agent.config import settings
from agent import planner as planner_module
from agent.planner import QueryClassifier, Planner
from agent.schemas import AnalysisMode


@pytest.fixture
def force_stub_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "stub")
    monkeypatch.setattr(settings, "openai_api_key", "")


class TestQueryClassifier:
    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_single_company(self):
        assert self.classifier.classify("Tell me about Stripe") == AnalysisMode.COMPANY

    def test_comparison(self):
        assert self.classifier.classify("Stripe vs Adyen") == AnalysisMode.COMPETITIVE

    def test_industry(self):
        assert self.classifier.classify("fintech industry landscape") == AnalysisMode.INDUSTRY

    def test_multiple_companies(self):
        assert self.classifier.classify("Compare Stripe and Notion") == AnalysisMode.COMPETITIVE

    def test_compare_keyword(self):
        assert self.classifier.classify("comparison of Stripe Adyen PayPal") == AnalysisMode.COMPETITIVE

    def test_company_name_does_not_trigger_industry_keyword_substring(self):
        query = "What is Cloudflare's core business model and long-term strategy?"
        assert self.classifier.classify(query) == AnalysisMode.COMPANY


class TestPlanner:
    def test_company_plan_has_all_required_steps(self, force_stub_llm):
        planner = Planner()
        plan = planner.create_plan("Analyze Stripe", mode=AnalysisMode.COMPANY)

        assert plan.mode == AnalysisMode.COMPANY
        step_names = [s.name for s in plan.steps]
        assert "company_overview" in step_names
        assert "business_model" in step_names
        assert "financial_quality" in step_names
        assert "competitive_landscape" in step_names
        assert "investment_takeaway" in step_names

    def test_each_step_has_search_queries(self, force_stub_llm):
        planner = Planner()
        plan = planner.create_plan("Analyze Stripe", mode=AnalysisMode.COMPANY)

        for step in plan.steps:
            assert len(step.search_queries) >= 2, f"{step.name} has too few queries"

    def test_stub_queries_infer_unknown_company_name(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan("Assess Revolut revenue quality", mode=AnalysisMode.COMPANY)

        overview_queries = plan.steps[0].search_queries
        assert any("Revolut" in query for query in overview_queries)
        assert all(not query.startswith("Assess ") for query in overview_queries)

    def test_stub_competitive_queries_infer_both_unknown_entities(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan("Compare Revolut vs Nubank", mode=AnalysisMode.COMPETITIVE)

        product_queries = next(step.search_queries for step in plan.steps if step.name == "product_comparison")
        assert any("Revolut vs Nubank" in query for query in product_queries)

    def test_plan_includes_execution_contract(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan("Compare Revolut vs Nubank", mode=AnalysisMode.COMPETITIVE)

        assert plan.contract["query_type"] == "comparison"
        assert plan.contract["entities"][:2] == ["Revolut", "Nubank"]
        assert "product breadth" in plan.contract["required_slots"]
        assert plan.contract["normalized_question"]
        assert plan.contract["answer_type"]
        assert plan.contract["estimate"]
        assert plan.contract["non_goals"]
        assert plan.contract["preferred_source_order"]

    def test_plan_includes_section_level_evidence_requirements(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan(
            "Compare Cloudflare and Fastly on product breadth, monetization style, and growth drivers",
            mode=AnalysisMode.COMPETITIVE,
        )

        product_step = next(step for step in plan.steps if step.name == "product_comparison")

        assert product_step.evidence_requirements["fact_first"] is True
        assert "product breadth" in " ".join(product_step.evidence_requirements["required_facts"]).lower()
        assert product_step.evidence_requirements["downgrade_rule"]
        assert product_step.query_contracts
        assert product_step.query_contracts[0]["query_text"]

    def test_live_planner_uses_deterministic_contract_without_llm(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_mode", "live")
        monkeypatch.setattr(settings, "openai_api_key", "sk-test-ascii-key")
        monkeypatch.setattr(settings, "openai_api_base", "https://callflow.top/v1")
        monkeypatch.setattr(settings, "openai_model", "gpt-5.2")

        call_count = {"value": 0}

        def fake_build_client(api_key: str, base_url: str):
            return object()

        def fake_generate_text_response(*args, **kwargs):
            call_count["value"] += 1
            return "{}"

        monkeypatch.setattr(planner_module, "build_openai_client", fake_build_client)
        monkeypatch.setattr(planner_module, "generate_text_response", fake_generate_text_response)

        planner = Planner()
        plan = planner.create_plan("Analyze Cloudflare", mode=AnalysisMode.COMPANY)

        assert plan.steps
        assert call_count["value"] == 0

    def test_quarter_reference_alone_does_not_force_time_sensitive_contract(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan(
            "How does Fastly make money, and what does management emphasize as the most important revenue mix components in Q4 2025?",
            mode=AnalysisMode.COMPANY,
        )

        assert plan.contract["query_type"] == "standard"
        assert "Q3 vs Q4 comparison" not in plan.contract["required_slots"]

    def test_temporal_change_query_uses_time_sensitive_contract(self, force_stub_llm):
        planner = Planner()

        plan = planner.create_plan(
            "How did Fastly's growth and profitability narrative change from Q2 2025 to Q4 2025?",
            mode=AnalysisMode.COMPANY,
        )

        assert plan.contract["query_type"] == "time_sensitive"
        assert "Q3 vs Q4 comparison" in plan.contract["required_slots"]
