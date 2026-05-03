from eval.trust_standards import (
    finance_hard_gates,
    mapping_for_metric,
    metric_family,
    render_trust_standards_markdown,
)


def test_trust_standards_map_core_metrics_to_gates():
    unsupported = mapping_for_metric("unsupported_claim_rate")
    numeric = mapping_for_metric("unsupported_numeric_claim_rate")

    assert unsupported is not None
    assert unsupported.fintrust_gate == "Grounding Gate"
    assert unsupported.judge_type == "hybrid"
    assert numeric is not None
    assert numeric.fintrust_gate == "Finance Hard Gate"
    assert numeric.judge_type == "deterministic"
    assert metric_family("required_fact_recall") == "retrieval_quality"
    assert metric_family("numeric_exact_match_rate") == "finance_hard_gate"
    assert metric_family("hard_fact_complete_but_not_published_rate") == "publication_gate"


def test_finance_hard_gates_include_numeric_and_citation_checks():
    gates = {gate["gate_name"]: gate for gate in finance_hard_gates()}

    assert gates["numeric_alignment"]["failure_action"] == "delete"
    assert gates["citation_validity"]["check_type"] == "metadata_check"
    assert "primary_source_required" in gates


def test_render_trust_standards_markdown_includes_external_patterns():
    markdown = render_trust_standards_markdown()

    assert "RAGAS/DeepEval faithfulness" in markdown
    assert "Finance Hard Gates" in markdown
