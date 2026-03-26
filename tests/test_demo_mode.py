from agent.orchestrator import BizIntelAgent
from agent.schemas import AnalysisMode


def test_demo_mode_runs_without_live_llm():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research("Analyze Stripe", mode=AnalysisMode.COMPANY)

    memo = result["memo_object"]
    assert memo.sections
    assert memo.executive_summary
    assert "workflow_events" in result


def test_demo_mode_returns_traceable_plan():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research("Compare Stripe vs PayPal", mode=AnalysisMode.COMPETITIVE)

    assert result["plan"].steps
    assert any(event["event_type"] == "started" for event in result["workflow_events"])


def test_demo_mode_uses_dummy_verifier():
    agent = BizIntelAgent(load_models=False, demo_mode=True)

    assert agent.report_writer.verifier._get_model().__class__.__name__ == "_DummyNLIModel"
