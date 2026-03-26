import json

from agent.artifacts import (
    build_trace_payload,
    build_verification_rows,
    write_artifact_bundle,
)
from agent.orchestrator import BizIntelAgent
from agent.schemas import AnalysisMode


def test_artifact_bundle_exports_demo_run(tmp_path):
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research("Analyze Stripe", mode=AnalysisMode.COMPANY)

    bundle = write_artifact_bundle(result, tmp_path)

    assert bundle["memo"].exists()
    assert bundle["trace"].exists()
    assert bundle["summary"].exists()
    assert bundle["verification"].exists()
    assert "Company Research Memo" in bundle["memo"].read_text(encoding="utf-8")


def test_trace_payload_includes_query_and_steps():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research("Compare Stripe vs PayPal", mode=AnalysisMode.COMPETITIVE)

    payload = build_trace_payload(result)

    assert payload["query"] == "Compare Stripe vs PayPal"
    assert payload["steps"]
    assert payload["contract"]["query_type"] == "comparison"
    assert payload["step_traces"]
    assert "final_answer" in payload
    assert any(event["event_type"] == "started" for event in payload["workflow_events"])
    assert "query_contracts" in payload["step_traces"][0]
    assert "evidence_ledger" in payload["step_traces"][0]
    assert "writing_trace" in payload["step_traces"][0]


def test_verification_rows_are_serializable():
    agent = BizIntelAgent(load_models=False, demo_mode=True)
    result = agent.research("Analyze Stripe", mode=AnalysisMode.COMPANY)

    rows = build_verification_rows(result["memo_object"])

    assert rows
    assert "failure_reason" in rows[0]
    json.dumps(rows, ensure_ascii=False)
