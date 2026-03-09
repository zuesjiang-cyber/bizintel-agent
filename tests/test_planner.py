import pytest
from unittest.mock import MagicMock, patch
from agent.planner import QueryClassifier, Planner
from agent.schemas import AnalysisMode

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


class TestPlanner:
    @patch('agent.planner.OpenAI')
    def test_company_plan_has_all_required_steps(self, mock_openai):
        # Setup mock behavior
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["mock query 1", "mock query 2"]'
        mock_client.chat.completions.create.return_value = mock_response

        planner = Planner()
        plan = planner.create_plan("Analyze Stripe", mode=AnalysisMode.COMPANY)

        assert plan.mode == AnalysisMode.COMPANY
        step_names = [s.name for s in plan.steps]
        assert "company_profile" in step_names
        assert "business_model" in step_names
        assert "market_position" in step_names

    @patch('agent.planner.OpenAI')
    def test_each_step_has_search_queries(self, mock_openai):
        # Setup mock behavior
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["mock query 1", "mock query 2"]'
        mock_client.chat.completions.create.return_value = mock_response

        planner = Planner()
        plan = planner.create_plan("Analyze Stripe", mode=AnalysisMode.COMPANY)

        for step in plan.steps:
            assert len(step.search_queries) >= 2, f"{step.name} has too few queries"
