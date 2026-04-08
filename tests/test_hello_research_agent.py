from agent.hello_research_agent import HelloDeepResearchAgent
from agent.research_controller import ResearchController
from agent.schemas import AnalysisMode


def test_hello_agent_records_planner_and_tool_calls():
    controller = ResearchController(load_models=False, demo_mode=True)
    agent = HelloDeepResearchAgent(controller)

    result = agent.research(
        "Assess Stripe revenue quality and management outlook",
        mode=AnalysisMode.COMPANY,
    )

    decision_types = [item.decision_type for item in result["research_trace"].decisions]
    assert "todo_plan" in decision_types
    assert "tool_call" in decision_types
    assert any(event["node_name"] == "todo_planner" for event in result["workflow_events"])


def test_hello_agent_preserves_explicit_scope_refusal():
    controller = ResearchController(load_models=False, demo_mode=True)
    agent = HelloDeepResearchAgent(controller)

    result = agent.research(
        "Compare Cloudflare and Fastly on product breadth",
        mode=AnalysisMode.COMPETITIVE,
    )

    assert result["memo_object"].contract["scope_supported"] is False
    assert result["subquestion_results"][0].status.value == "refused"


def test_hello_agent_respects_explicit_company_and_period_scope():
    controller = ResearchController(load_models=False, demo_mode=True)
    agent = HelloDeepResearchAgent(controller)

    result = agent.research(
        "What changed from Q3 2025 to Q4 2025 in how management described revenue mix and growth drivers?",
        mode=AnalysisMode.COMPANY,
        company_id="fastly",
        target_periods=["2025Q3", "2025Q4"],
    )

    assert result["research_task"].company_id == "fastly"
    assert result["research_task"].target_periods == ["2025Q3", "2025Q4"]
