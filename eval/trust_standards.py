from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TrustMetricMapping:
    metric_name: str
    metric_family: str
    external_pattern: str
    fintrust_gate: str
    judge_type: str
    required_fields: tuple[str, ...]
    failure_reason: str
    failure_action: str
    artifact_fields: tuple[str, ...]
    display_priority: int


@dataclass(frozen=True)
class FinanceHardGate:
    gate_name: str
    claim_type: str
    required_evidence_fields: tuple[str, ...]
    check_type: str
    pass_condition: str
    failure_reason: str
    failure_action: str
    severity: str
    example_artifact: str


TRUST_METRIC_MAPPINGS: tuple[TrustMetricMapping, ...] = (
    TrustMetricMapping(
        metric_name="unsupported_claim_rate",
        metric_family="grounding_quality",
        external_pattern="RAGAS/DeepEval faithfulness; TruLens groundedness; Phoenix hallucination eval",
        fintrust_gate="Grounding Gate",
        judge_type="hybrid",
        required_fields=("claim_text", "supporting_chunk_ids", "nli_score", "failure_reason"),
        failure_reason="unsupported_claim",
        failure_action="delete_or_downgrade",
        artifact_fields=("verification.csv", "trace.json", "claim_results"),
        display_priority=1,
    ),
    TrustMetricMapping(
        metric_name="unsupported_numeric_claim_rate",
        metric_family="finance_hard_gate",
        external_pattern="OpenAI grader pattern plus deterministic numeric checks",
        fintrust_gate="Finance Hard Gate",
        judge_type="deterministic",
        required_fields=("claim_text", "supporting_evidence", "period", "unit", "currency"),
        failure_reason="numeric_mismatch",
        failure_action="delete",
        artifact_fields=("verification.csv", "claim_results"),
        display_priority=2,
    ),
    TrustMetricMapping(
        metric_name="fabricated_citation_rate",
        metric_family="citation_quality",
        external_pattern="Anthropic source-grounded citations",
        fintrust_gate="Citation Validity Gate",
        judge_type="metadata_check",
        required_fields=("cited_sources", "cited_chunks", "source_id", "chunk_id"),
        failure_reason="bad_source_id_or_chunk_id",
        failure_action="delete",
        artifact_fields=("verification.csv", "trace.json"),
        display_priority=3,
    ),
    TrustMetricMapping(
        metric_name="primary_source_claim_coverage",
        metric_family="source_quality",
        external_pattern="Source-grounded citation discipline; finance primary-source hierarchy",
        fintrust_gate="Primary Source Gate",
        judge_type="metadata_check",
        required_fields=("source_type", "primary_source", "supporting_source_ids"),
        failure_reason="missing_primary_source",
        failure_action="downgrade_or_refuse",
        artifact_fields=("verification.csv", "sources.json"),
        display_priority=4,
    ),
    TrustMetricMapping(
        metric_name="numeric_exact_match_rate",
        metric_family="finance_hard_gate",
        external_pattern="Deterministic numeric exact-match grading",
        fintrust_gate="Finance Hard Gate",
        judge_type="deterministic",
        required_fields=("claim_text", "supporting_evidence", "expected_numeric_answer"),
        failure_reason="numeric_answer_miss",
        failure_action="delete_or_mark_miss",
        artifact_fields=("benchmark.json", "verification.csv", "claim_results"),
        display_priority=5,
    ),
    TrustMetricMapping(
        metric_name="required_fact_recall",
        metric_family="retrieval_quality",
        external_pattern="RAGAS/DeepEval context recall",
        fintrust_gate="Retrieval Gate",
        judge_type="hybrid",
        required_fields=("required_facts", "retrieval_candidates", "final_answer"),
        failure_reason="required_fact_missing",
        failure_action="needs_followup_or_mark_gap",
        artifact_fields=("benchmark.json", "trace.json"),
        display_priority=6,
    ),
    TrustMetricMapping(
        metric_name="required_slot_coverage",
        metric_family="retrieval_quality",
        external_pattern="RAGAS/DeepEval context recall plus task-slot coverage",
        fintrust_gate="Retrieval Gate",
        judge_type="hybrid",
        required_fields=("required_slots", "covered_facts", "final_answer"),
        failure_reason="required_slot_missing",
        failure_action="needs_followup_or_mark_gap",
        artifact_fields=("benchmark.json", "trace.json"),
        display_priority=7,
    ),
    TrustMetricMapping(
        metric_name="retrieval_hit",
        metric_family="retrieval_quality",
        external_pattern="Context relevance / retrieval relevance",
        fintrust_gate="Retrieval Gate",
        judge_type="metadata_check",
        required_fields=("retrieval_candidates", "source_id", "chunk_id"),
        failure_reason="no_relevant_candidate",
        failure_action="needs_followup_or_refuse",
        artifact_fields=("benchmark.json", "trace.json"),
        display_priority=8,
    ),
    TrustMetricMapping(
        metric_name="hard_fact_complete_but_not_published_rate",
        metric_family="publication_gate",
        external_pattern="Trace-level error localization / publication gate regression",
        fintrust_gate="Publication Gate",
        judge_type="metadata_check",
        required_fields=("covered_facts", "final_answer", "verification_rows"),
        failure_reason="complete_fact_not_published",
        failure_action="recover_or_report_publication_gap",
        artifact_fields=("benchmark.json", "trace.json", "verification.csv"),
        display_priority=9,
    ),
    TrustMetricMapping(
        metric_name="wrong_entity_rate",
        metric_family="finance_hard_gate",
        external_pattern="Deterministic entity consistency check",
        fintrust_gate="Entity Gate",
        judge_type="deterministic",
        required_fields=("company_id", "item_companies", "source_company"),
        failure_reason="wrong_entity",
        failure_action="delete_or_refuse",
        artifact_fields=("benchmark.json", "sources.json"),
        display_priority=10,
    ),
    TrustMetricMapping(
        metric_name="wrong_period_rate",
        metric_family="finance_hard_gate",
        external_pattern="Deterministic period consistency check",
        fintrust_gate="Period Gate",
        judge_type="deterministic",
        required_fields=("target_period", "period", "claim_text"),
        failure_reason="wrong_period",
        failure_action="delete_or_refuse",
        artifact_fields=("benchmark.json", "verification.csv"),
        display_priority=11,
    ),
    TrustMetricMapping(
        metric_name="gold_benchmark_pass",
        metric_family="benchmark_quality",
        external_pattern="Task-level exactness / QA correctness grading",
        fintrust_gate="Benchmark Gate",
        judge_type="hybrid",
        required_fields=("gold_answer", "final_answer", "gold_evidence"),
        failure_reason="gold_answer_miss",
        failure_action="report_regression",
        artifact_fields=("benchmark.json",),
        display_priority=19,
    ),
    TrustMetricMapping(
        metric_name="answer_quality",
        metric_family="presentation_quality",
        external_pattern="Answer relevancy / QA correctness evaluator",
        fintrust_gate="Presentation Gate",
        judge_type="llm_judge",
        required_fields=("query", "final_answer", "required_facts"),
        failure_reason="low_answer_quality",
        failure_action="warn",
        artifact_fields=("benchmark.json",),
        display_priority=20,
    ),
)


FINANCE_HARD_GATES: tuple[FinanceHardGate, ...] = (
    FinanceHardGate(
        gate_name="entity_match",
        claim_type="all_factual_claims",
        required_evidence_fields=("company_id", "source_id", "source_company"),
        check_type="deterministic",
        pass_condition="Claim entity matches the benchmark item company and cited source company.",
        failure_reason="wrong_entity",
        failure_action="delete_or_refuse",
        severity="critical",
        example_artifact="benchmark.json:item_companies + sources.json:company",
    ),
    FinanceHardGate(
        gate_name="period_match",
        claim_type="period_sensitive_claims",
        required_evidence_fields=("period", "source_date", "claim_text"),
        check_type="deterministic",
        pass_condition="Claim period is supported by the cited evidence period or source date.",
        failure_reason="wrong_period",
        failure_action="delete_or_refuse",
        severity="critical",
        example_artifact="verification.csv:claim_text + trace.json:evidence_notes",
    ),
    FinanceHardGate(
        gate_name="numeric_alignment",
        claim_type="financial_numeric_claims",
        required_evidence_fields=("claim_text", "supporting_evidence", "unit", "currency"),
        check_type="deterministic",
        pass_condition="Number, unit, currency, period, and directionality align with supporting evidence.",
        failure_reason="numeric_mismatch",
        failure_action="delete",
        severity="critical",
        example_artifact="verification.csv:numeric_verified",
    ),
    FinanceHardGate(
        gate_name="citation_validity",
        claim_type="cited_claims",
        required_evidence_fields=("cited_sources", "cited_chunks", "source_id", "chunk_id"),
        check_type="metadata_check",
        pass_condition="Every cited source and chunk exists in the local source pack or trace evidence.",
        failure_reason="fabricated_citation",
        failure_action="delete",
        severity="critical",
        example_artifact="verification.csv:cited_sources,cited_chunks",
    ),
    FinanceHardGate(
        gate_name="primary_source_required",
        claim_type="high_risk_financial_claims",
        required_evidence_fields=("source_type", "primary_source", "supporting_source_ids"),
        check_type="metadata_check",
        pass_condition="Strong financial conclusion is supported by a primary or high-trust company source.",
        failure_reason="missing_primary_source",
        failure_action="downgrade_or_refuse",
        severity="high",
        example_artifact="sources.json:source_type + verification.csv:primary_source_supported",
    ),
)


def metric_mappings() -> list[dict[str, object]]:
    return [asdict(item) for item in sorted(TRUST_METRIC_MAPPINGS, key=lambda item: item.display_priority)]


def finance_hard_gates() -> list[dict[str, object]]:
    return [asdict(item) for item in FINANCE_HARD_GATES]


def mapping_for_metric(metric_name: str) -> TrustMetricMapping | None:
    return next((item for item in TRUST_METRIC_MAPPINGS if item.metric_name == metric_name), None)


def metric_family(metric_name: str) -> str:
    mapping = mapping_for_metric(metric_name)
    return mapping.metric_family if mapping else "uncategorized"


def metrics_by_family(metrics: Iterable[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for metric in metrics:
        families.setdefault(metric_family(metric), []).append(metric)
    return families


def render_trust_standards_markdown() -> str:
    lines = [
        "# Trust Standards Mapping",
        "",
        "FinTrust RAG maps general RAG evaluation patterns to finance-specific hard gates.",
        "",
        "## Metric Mapping",
        "",
        "| Metric | Family | External Pattern | FinTrust Gate | Judge Type | Failure Action |",
        "|---|---|---|---|---|---|",
    ]
    for item in sorted(TRUST_METRIC_MAPPINGS, key=lambda row: row.display_priority):
        lines.append(
            "| "
            f"{item.metric_name} | {item.metric_family} | {item.external_pattern} | "
            f"{item.fintrust_gate} | {item.judge_type} | {item.failure_action} |"
        )
    lines.extend(
        [
            "",
            "## Finance Hard Gates",
            "",
            "| Gate | Claim Type | Check Type | Failure Reason | Failure Action | Severity |",
            "|---|---|---|---|---|---|",
        ]
    )
    for gate in FINANCE_HARD_GATES:
        lines.append(
            "| "
            f"{gate.gate_name} | {gate.claim_type} | {gate.check_type} | "
            f"{gate.failure_reason} | {gate.failure_action} | {gate.severity} |"
        )
    return "\n".join(lines) + "\n"
