"""
Run the local benchmark suite against the current indexed corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.artifacts import write_artifact_bundle
from agent.config import settings
from agent.orchestrator import BizIntelAgent
from eval.evaluator import extract_scorable_markdown, score_markdown, score_research_trace
from eval.financebench_mapping import score_gold_answer_mapping
from eval.llm_judge import LLMJudge
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier
from retrieval.hybrid_retriever import HybridRetriever

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError
except ImportError:  # pragma: no cover
    APIConnectionError = None
    APIStatusError = None
    APITimeoutError = None


logger = logging.getLogger(__name__)

PRIMARY_METRICS = [
    "contradiction_rate",
    "fabricated_citation_rate",
    "unsupported_claim_rate",
    "unsafe_publish_rate",
    "numeric_exact_match_rate",
    "unsupported_numeric_claim_rate",
    "primary_source_support_rate",
    "primary_source_missing_rate",
    "strong_support_rate",
    "weak_support_rate",
    "abstention_precision",
    "abstention_recall",
]
SECONDARY_METRICS = [
    "atomic_claim_rate",
    "claim_extract_success_rate",
    "period_match_rate",
    "currency_match_rate",
    "directionality_match_rate",
    "required_slot_coverage",
    "required_fact_recall",
    "retrieval_hit",
    "gold_answer_hit",
    "gold_numeric_hit",
    "answer_quality",
]


def format_metric(value: Any, decimals: int = 4, default: str = "n/a") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def contains_query_phrase(query: str, phrase: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
    return re.search(pattern, query) is not None


def contains_any_query_phrase(query: str, phrases: List[str]) -> bool:
    return any(contains_query_phrase(query, phrase) for phrase in phrases)


def classify_question_family(query: str, query_types: List[str]) -> str:
    lowered = query.lower()
    if contains_any_query_phrase(lowered, ["customer concentration", "concentration", "major customer"]):
        return "customer_concentration"
    if contains_query_phrase(lowered, "ratio"):
        return "ratio"
    if contains_any_query_phrase(lowered, ["year-over-year", "quarter-over-quarter", "yoy", "qoq"]):
        return "yoy_qoq"
    if contains_any_query_phrase(lowered, ["highest", "lowest", "largest", "smallest", "most", "least"]):
        if contains_query_phrase(lowered, "segment"):
            return "segment_ranking"
        return "max_min"
    if contains_any_query_phrase(lowered, ["what drove", "why did", "reason", "drivers"]):
        return "reason_attribution"
    if contains_any_query_phrase(lowered, ["from fy", "from q", "between", "vs", "versus", "compared"]):
        return "two_period_comparison"
    if contains_any_query_phrase(lowered, ["what is", "what was", "how much", "amount", "did", "reported"]):
        return "single_value"
    if "numeric_grounding" in query_types:
        return "single_value"
    return "general"


def derive_answer_status_decision(
    metrics: dict,
    scorable_markdown: str,
    *,
    question_text: str = "",
    judge: Optional[LLMJudge] = None,
) -> dict:
    lowered = scorable_markdown.lower()
    if metrics.get("total_claims", 0) == 0:
        rule_status = "abstained"
        rule_reason = "No verifiable claims were produced."
    elif "insufficient evidence" in lowered and metrics.get("strong_support_rate", 0.0) == 0.0 and metrics.get("required_slot_coverage", 0.0) < 0.5:
        rule_status = "abstained"
        rule_reason = "The answer text is mostly an insufficiency statement."
    elif (
        metrics.get("strong_support_rate", 0.0) > 0.0
        and metrics.get("fabricated_citation_rate", 0.0) == 0.0
        and metrics.get("contradiction_rate", 0.0) == 0.0
        and metrics.get("required_slot_coverage", 0.0) >= 0.75
    ):
        rule_status = "answered"
        rule_reason = "The answer contains supported claims and covers the required slots."
    else:
        rule_status = "partial"
        rule_reason = "The answer contains some usable content but still has important gaps."

    judge = judge or LLMJudge()
    decision = judge.adjudicate_answer_status(
        question_text=question_text,
        scorable_markdown=scorable_markdown,
        metrics=metrics,
        rule_status=rule_status,
    )
    if decision:
        return {
            "status": decision["answer_status"],
            "reason": decision.get("reason") or rule_reason,
            "rule_status": rule_status,
            "rule_reason": rule_reason,
            "llm_reviewed": True,
            "llm_adjudicated": decision["answer_status"] != rule_status,
        }
    return {
        "status": rule_status,
        "reason": rule_reason,
        "rule_status": rule_status,
        "rule_reason": rule_reason,
        "llm_reviewed": False,
        "llm_adjudicated": False,
    }


def derive_answer_status(metrics: dict, scorable_markdown: str) -> str:
    return derive_answer_status_decision(metrics, scorable_markdown)["status"]


def should_abstain(metrics: dict, research_metrics: dict) -> bool:
    return (
        metrics.get("strong_support_rate", 0.0) == 0.0
        and metrics.get("numeric_exact_match_rate", 0.0) == 0.0
        and metrics.get("primary_source_support_rate", 0.0) == 0.0
        and metrics.get("required_fact_recall", 0.0) == 0.0
        and research_metrics.get("hard_fact_completion_rate", 0.0) == 0.0
    )


def summarize_retrieval_candidates(result: dict, limit_per_question: int = 3) -> List[dict]:
    candidates: List[dict] = []
    for subquestion_result in result.get("subquestion_results", []):
        subquestion = getattr(subquestion_result, "subquestion", None)
        question_id = getattr(subquestion, "question_id", "")
        question_text = getattr(subquestion, "text", "")
        for chunk in list(getattr(subquestion_result, "evidence", []))[:limit_per_question]:
            candidates.append(
                {
                    "question_id": question_id,
                    "question_text": question_text,
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "source_id": getattr(chunk, "source_id", ""),
                    "source_type": getattr(chunk, "source_type", None),
                    "score": getattr(chunk, "score", None),
                    "period": getattr(chunk, "period", None),
                }
            )
    return candidates


def summarize_final_cited_evidence(claim_diagnostics: List[dict]) -> List[dict]:
    rows: List[dict] = []
    seen = set()
    for claim in claim_diagnostics:
        support_texts = claim.get("supporting_evidence") or []
        chunk_ids = claim.get("supporting_chunk_ids") or []
        source_ids = claim.get("supporting_source_ids") or []
        max_len = max(len(chunk_ids), len(source_ids), len(support_texts), 1)
        for idx in range(max_len):
            chunk_id = chunk_ids[idx] if idx < len(chunk_ids) else ""
            source_id = source_ids[idx] if idx < len(source_ids) else ""
            evidence_text = support_texts[idx] if idx < len(support_texts) else (support_texts[0] if support_texts else "")
            key = (chunk_id, source_id, evidence_text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "claim_text": claim.get("claim_text", ""),
                    "support_label": claim.get("support_label", "unsupported"),
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "evidence_text": evidence_text,
                    "source_tier": claim.get("source_tier"),
                }
            )
    return rows


def summarize_question_errors(metrics: dict, research_metrics: dict) -> dict:
    claim_error_counts = dict(metrics.get("claim_error_counts", {}))
    controller_failures = list(research_metrics.get("controller_failure_reasons", []))
    return {
        "claim_error_counts": claim_error_counts,
        "claim_error_rates": dict(metrics.get("claim_error_rates", {})),
        "controller_failure_reasons": controller_failures,
        "most_severe_failure_type": metrics.get("most_severe_failure_type") or (controller_failures[0] if controller_failures else None),
    }


def benchmark_artifact_root(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_artifacts")


def render_benchmark_case_report(item: dict, row: dict, result: dict) -> str:
    question_errors = row.get("question_error_summary", {})
    lines = [
        f"# Benchmark Case {row['item_id']}",
        "",
        f"- Query: {item['query']}",
        f"- Category: {item.get('category', 'unknown')}",
        f"- Difficulty: {item.get('difficulty', 'unknown')}",
        f"- Query types: {', '.join(item.get('query_type', [])) or 'n/a'}",
        f"- Question family: {row.get('question_family', 'general')}",
        f"- Answer status: {row.get('answer_status', 'unknown')}",
        f"- Companies: {', '.join(row.get('item_companies', [])) or 'n/a'}",
        f"- Sources used: {', '.join(row.get('sources_used', [])) or 'n/a'}",
        f"- Failure tags: {', '.join(row.get('failure_tags', [])) or 'none'}",
        "",
        "## Trust Metrics",
        "",
        f"- strong_support_rate: {row.get('strong_support_rate', 0.0):.4f}",
        f"- weak_support_rate: {row.get('weak_support_rate', 0.0):.4f}",
        f"- contradiction_rate: {row.get('contradiction_rate', 0.0):.4f}",
        f"- fabricated_citation_rate: {row.get('fabricated_citation_rate', 0.0):.4f}",
        f"- unsupported_claim_rate: {row.get('unsupported_claim_rate', 0.0):.4f}",
        f"- numeric_exact_match_rate: {row.get('numeric_exact_match_rate', 0.0):.4f}",
        f"- primary_source_support_rate: {row.get('primary_source_support_rate', 0.0):.4f}",
        f"- abstention_precision: {format_metric(row.get('abstention_precision'))}",
        f"- abstention_recall: {format_metric(row.get('abstention_recall'))}",
        "",
        "## Completeness Metrics",
        "",
        f"- retrieval_hit: {row.get('retrieval_hit', 0.0)}",
        f"- required_subquestion_coverage: {row.get('required_subquestion_coverage', 0.0):.4f}",
        f"- required_slot_coverage: {row.get('required_slot_coverage', 0.0):.4f}",
        f"- required_fact_recall: {row.get('required_fact_recall', 0.0):.4f}",
        f"- gold_answer_hit: {row.get('gold_answer_hit', 0.0):.4f}",
        f"- gold_numeric_hit: {row.get('gold_numeric_hit', 0.0):.4f}",
        "",
        "## Safety Diagnostics",
        "",
        f"- wrong_entity_rate: {row.get('wrong_entity_rate', 0.0)}",
        f"- wrong_period_rate: {row.get('wrong_period_rate', 0.0)}",
        f"- decision_replay_consistency: {row.get('decision_replay_consistency', 0.0):.4f}",
        f"- most_severe_failure_type: {question_errors.get('most_severe_failure_type') or 'none'}",
        "",
        "## Claim Errors",
        "",
        json.dumps(question_errors.get("claim_error_counts", {}), ensure_ascii=False, indent=2),
        "",
        "## Natural Text Memo",
        "",
        result["memo_markdown"].strip(),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def render_benchmark_report(payload: dict) -> str:
    averages = payload.get("averages", {})
    run_summary = payload.get("run_summary", {})
    portfolio_summary = payload.get("portfolio_summary", {})
    delta = payload.get("delta_from_baseline", {})
    lines = [
        f"# Benchmark Report: {payload.get('profile') or payload.get('split')}",
        "",
        f"- Version: {payload.get('version')}",
        f"- Mode: {payload.get('mode')}",
        f"- Timestamp: {payload.get('timestamp')}",
        f"- Companies: {', '.join(payload.get('companies', [])) or 'n/a'}",
        f"- Baseline: {payload.get('baseline_path') or 'n/a'}",
        "",
        "## Run Summary",
        "",
        f"- total_items: {run_summary.get('total_items', len(payload.get('rows', [])))}",
        f"- answered_items: {run_summary.get('answered_items', 0)}",
        f"- partial_items: {run_summary.get('partial_items', 0)}",
        f"- abstained_items: {run_summary.get('abstained_items', 0)}",
        f"- question_type_counts: {json.dumps(portfolio_summary.get('question_type_counts', {}), ensure_ascii=False)}",
        "",
        "## Primary Metrics",
        "",
        f"- contradiction_rate: {averages.get('contradiction_rate', 0.0):.4f}",
        f"- fabricated_citation_rate: {averages.get('fabricated_citation_rate', 0.0):.4f}",
        f"- unsupported_claim_rate: {averages.get('unsupported_claim_rate', 0.0):.4f}",
        f"- numeric_exact_match_rate: {averages.get('numeric_exact_match_rate', 0.0):.4f}",
        f"- primary_source_support_rate: {averages.get('primary_source_support_rate', 0.0):.4f}",
        f"- strong_support_rate: {averages.get('strong_support_rate', 0.0):.4f}",
        f"- weak_support_rate: {averages.get('weak_support_rate', 0.0):.4f}",
        f"- abstention_precision: {averages.get('abstention_precision', 0.0):.4f}",
        f"- abstention_recall: {averages.get('abstention_recall', 0.0):.4f}",
        "",
        "## Secondary Metrics",
        "",
        f"- retrieval_hit: {averages.get('retrieval_hit', 0.0):.4f}",
        f"- wrong_entity_rate: {averages.get('wrong_entity_rate', 0.0):.4f}",
        f"- wrong_period_rate: {averages.get('wrong_period_rate', 0.0):.4f}",
        f"- required_subquestion_coverage: {averages.get('required_subquestion_coverage', 0.0):.4f}",
        f"- required_slot_coverage: {averages.get('required_slot_coverage', 0.0):.4f}",
        f"- required_fact_recall: {averages.get('required_fact_recall', 0.0):.4f}",
        f"- decision_replay_consistency: {averages.get('decision_replay_consistency', 0.0):.4f}",
        f"- gold_answer_hit: {averages.get('gold_answer_hit', 0.0):.4f}",
        f"- gold_numeric_hit: {averages.get('gold_numeric_hit', 0.0):.4f}",
        "",
        "## Delta Vs Baseline",
        "",
        json.dumps({key: round(value, 6) for key, value in delta.items()}, ensure_ascii=False, indent=2),
        "",
        "## Failure Distribution",
        "",
        json.dumps(run_summary.get("failure_type_distribution", {}), ensure_ascii=False, indent=2),
        "",
        "## Cases",
        "",
    ]
    for row in payload.get("rows", []):
        lines.extend(
            [
                f"### {row.get('item_id', 'unknown')}",
                "",
                f"- Query: {row.get('query', '')}",
                f"- Failure tags: {', '.join(row.get('failure_tags', [])) or 'none'}",
                f"- Memo: {row.get('memo_path', '') or 'n/a'}",
                f"- Trace: {row.get('trace_path', '') or 'n/a'}",
                f"- Verification: {row.get('verification_path', '') or 'n/a'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_case_artifacts(output_path: Path, item: dict, row: dict, result: dict) -> Dict[str, str]:
    artifact_root = benchmark_artifact_root(output_path)
    case_dir = artifact_root / row["item_id"]
    bundle = write_artifact_bundle(result, case_dir)
    report_path = case_dir / "benchmark_report.md"
    report_path.write_text(render_benchmark_case_report(item, row, result), encoding="utf-8")
    return {
        "artifact_dir": str(case_dir),
        "memo_path": str(bundle["memo"]),
        "trace_path": str(bundle["trace"]),
        "summary_path": str(bundle["summary"]),
        "verification_path": str(bundle["verification"]),
        "report_path": str(report_path),
    }


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_chunks_by_doc() -> Dict[str, List[dict]]:
    chunks_by_doc: Dict[str, List[dict]] = {}
    normalized_root = settings.data_dir / "normalized"
    for company_dir in sorted(normalized_root.iterdir()):
        chunks_path = company_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        for chunk in load_jsonl(chunks_path):
            chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    return chunks_by_doc


def load_evidence_store(allowed_doc_ids: set[str] | None = None) -> Dict[str, List[dict]]:
    evidence_store: Dict[str, List[dict]] = {}
    normalized_root = settings.data_dir / "normalized"
    for company_dir in sorted(normalized_root.iterdir()):
        chunks_path = company_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        for chunk in load_jsonl(chunks_path):
            if allowed_doc_ids is not None and chunk["doc_id"] not in allowed_doc_ids:
                continue
            evidence_store.setdefault(chunk["source_id"], []).append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_id": chunk.get("source_id"),
                    "text": chunk.get("text", ""),
                    "source_type": chunk.get("source_type"),
                    "is_primary": chunk.get("is_primary"),
                }
            )
    return evidence_store


def load_evidence_store_for_companies(companies: List[str], allowed_doc_ids: set[str] | None = None) -> Dict[str, List[dict]]:
    evidence_store: Dict[str, List[dict]] = {}
    normalized_root = settings.data_dir / "normalized"
    for company in sorted(companies):
        chunks_path = normalized_root / company / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        for chunk in load_jsonl(chunks_path):
            if allowed_doc_ids is not None and chunk["doc_id"] not in allowed_doc_ids:
                continue
            evidence_store.setdefault(chunk["source_id"], []).append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_id": chunk.get("source_id"),
                    "text": chunk.get("text", ""),
                    "source_type": chunk.get("source_type"),
                    "is_primary": chunk.get("is_primary"),
                }
            )
    return evidence_store


def load_processed_chunks(companies: List[str]) -> List[dict]:
    chunks: List[dict] = []
    for company in companies:
        chunks_path = settings.data_dir / "processed" / company / "chunks.json"
        if not chunks_path.exists():
            continue
        with open(chunks_path, encoding="utf-8") as handle:
            chunks.extend(json.load(handle))
    return chunks


def benchmark_index_key(companies: List[str]) -> str:
    return "__".join(sorted(companies))


def benchmark_index_profile(load_models: bool) -> str:
    mode = "real" if load_models else "dummy"
    model_slug = re.sub(r"[^a-z0-9]+", "-", settings.embedding_model.lower()).strip("-")
    return f"{mode}__{model_slug}"


def ensure_benchmark_index(version: str, companies: List[str], load_models: bool = False) -> Path:
    index_dir = (
        settings.data_dir
        / "benchmark"
        / version
        / "index"
        / benchmark_index_profile(load_models)
        / benchmark_index_key(companies)
    )
    if (index_dir / "chunks.json").exists() and (index_dir / "dense_embeddings.npy").exists():
        return index_dir

    chunks = load_processed_chunks(companies)
    if not chunks:
        raise RuntimeError("No processed benchmark chunks found. Build the benchmark KB first.")

    retriever = HybridRetriever(load_models=load_models)
    retriever.index(chunks)
    retriever.save_index(index_dir)
    return index_dir


def matched_gold_docs(gold_entries: List[dict], memo_markdown: str, source_ids: List[str]) -> set[str]:
    cited = {entry["doc_id"] for entry in gold_entries if f"[Source: {entry['doc_id']}]" in memo_markdown}
    cited.update(source_ids)
    gold_doc_ids = {entry["doc_id"] for entry in gold_entries}
    return cited & gold_doc_ids


def retrieval_hit(gold_entries: List[dict], memo_markdown: str, source_ids: List[str], min_distinct_sources: int) -> int:
    return 1 if len(matched_gold_docs(gold_entries, memo_markdown, source_ids)) >= min_distinct_sources else 0


def verified_claim_coverage(metrics: dict) -> float:
    return float(metrics.get("verified_claim_coverage", 0.0))


def answer_quality(item: dict, metrics: dict, matched_count: int) -> int:
    strong_support_rate = float(metrics.get("strong_support_rate", metrics.get("verified_claim_coverage", 0.0)))
    contradiction_rate = float(metrics.get("contradiction_rate", 0.0))
    fabricated_citation_rate = float(metrics.get("fabricated_citation_rate", 0.0))
    numeric_exact_match_rate = float(
        metrics.get(
            "numeric_exact_match_rate",
            1.0 if strong_support_rate >= 0.75 and float(metrics.get("unsupported_claim_rate", 1.0)) <= 0.2 else 0.0,
        )
    )
    score = 1
    if strong_support_rate >= 0.55:
        score += 1
    if contradiction_rate == 0.0 and fabricated_citation_rate == 0.0:
        score += 1
    if numeric_exact_match_rate >= 0.75:
        score += 1
    if metrics["required_fact_recall"] >= 0.5:
        score += 1
    if matched_count >= item["required_evidence"]["min_distinct_sources"]:
        score += 1
    if strong_support_rate >= 0.75 and metrics["unsupported_claim_rate"] <= 0.2:
        score = min(score + 1, 5)
    if contradiction_rate > 0.0 or fabricated_citation_rate > 0.0:
        score = min(score, 2)
    if metrics.get("required_subquestion_coverage", 1.0) < 0.5:
        score = min(score, 3)
    if metrics.get("required_slot_coverage", 1.0) < 0.5:
        score = min(score, 3)
    if metrics.get("decision_replay_consistency", 1.0) < 1.0:
        score = min(score, 4)
    if metrics.get("gold_answer_mode", "unmapped") == "numeric_exact" and metrics.get("gold_numeric_hit", 0.0) < 1.0:
        score = min(score, 3)
    if (
        metrics.get("gold_answer_mode", "unmapped") == "semantic_gold_answer"
        and metrics.get("gold_semantic_hit", 1.0) < 1.0
        and strong_support_rate < 0.6
    ):
        score = min(score, 4)
    return min(score, 5)


def failure_tags(item: dict, metrics: dict, memo_markdown: str, gold_entries: List[dict], source_ids: List[str], matched_count: int) -> List[str]:
    tags: List[str] = []
    if metrics.get("primary_source_support_rate", 0.0) < 0.4:
        tags.append("R1_missing_primary_source")
    if metrics.get("fabricated_citation_rate", 0.0) > 0.0:
        tags.append("S5_fabricated_citation")
    if metrics.get("contradiction_rate", 0.0) > 0.0:
        tags.append("S6_contradiction")
    if metrics["unsupported_claim_rate"] > 0.4:
        tags.append("S5_untrusted_answer")
    if metrics["required_fact_recall"] < 0.5:
        tags.append("S1_incomplete_answer")
    if item["category"] == "time_sensitive" and "Q3" not in memo_markdown and "Q4" not in memo_markdown and "Q2" not in memo_markdown:
        tags.append("S4_temporal_confusion")
    if item["category"] == "comparison" and "Cloudflare" not in memo_markdown and "Fastly" not in memo_markdown:
        tags.append("S3_wrong_comparison_dimension")
    if item["category"] == "evidence_conflict" and "risk" not in memo_markdown.lower():
        tags.append("V4_conflict_not_detected")
    if matched_count < item["required_evidence"]["min_distinct_sources"]:
        tags.append("R1_missing_primary_source")
    if metrics.get("required_subquestion_coverage") is not None and metrics["required_subquestion_coverage"] < 0.5:
        tags.append("S1_question_tree_incomplete")
    if metrics.get("required_slot_coverage") is not None and metrics["required_slot_coverage"] < 0.5:
        tags.append("S1_missing_required_slot")
    if metrics.get("decision_replay_consistency") is not None and metrics["decision_replay_consistency"] < 1.0:
        tags.append("A2_replay_inconsistent")
    if metrics.get("wrong_entity_rate", 0.0) > 0.0:
        tags.append("R2_entity_leakage")
    if metrics.get("wrong_period_rate", 0.0) > 0.2:
        tags.append("S4_temporal_confusion")
    if metrics.get("scope_supported") is False:
        tags.append("A1_unsupported_scope_multi_company")
    if metrics.get("gold_answer_mode", "unmapped") == "numeric_exact" and metrics.get("gold_numeric_hit", 0.0) < 1.0:
        tags.append("G1_gold_numeric_miss")
    if metrics.get("gold_answer_mode", "unmapped") != "unmapped" and metrics.get("gold_citation_hit", 0.0) < 1.0:
        tags.append("G1_gold_citation_miss")
    if metrics.get("gold_answer_mode", "unmapped") != "unmapped" and metrics.get("gold_semantic_hit", 0.0) < 1.0:
        tags.append("G1_gold_semantic_gap")
    return sorted(set(tags))


def build_required_facts(answer_row: dict) -> List[str]:
    return list(answer_row.get("must_cover", []))


def describe_mode(load_models: bool) -> str:
    if settings.llm_mode == "stub" or not settings.openai_api_key:
        llm_mode = "offline_stub_llm"
    elif settings.strict_live_mode:
        llm_mode = "strict_live_llm"
    else:
        llm_mode = "live_llm"
    retrieval_mode = "real_retrieval" if load_models else "dummy_retrieval"
    verifier_mode = "dummy_nli" if settings.llm_mode == "stub" or not settings.openai_api_key else "real_nli"
    return "_".join([llm_mode, retrieval_mode, verifier_mode])


def is_retryable_live_error(exc: Exception) -> bool:
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is None or int(status_code) >= 500 or int(status_code) == 429:
            return True
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return True
    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return True

    text = str(exc).lower()
    retryable_markers = (
        "unknown provider",
        "server_error",
        "timed out",
        "timeout",
        "connection error",
        "bad gateway",
        "gateway",
        "502",
        "503",
        "504",
        "rate limit",
    )
    return any(marker in text for marker in retryable_markers)


def requested_item_ids(splits: dict, split: str) -> List[str]:
    if split == "all":
        return list(splits["dev"]) + list(splits["test"])
    return list(splits[split])


def item_splits(splits: dict, split: str) -> Dict[str, str]:
    if split == "all":
        mapping = {}
        for split_name in ("dev", "test"):
            for item_id in splits[split_name]:
                mapping[item_id] = split_name
        return mapping
    return {item_id: split for item_id in splits[split]}


def resolve_item_split_label(
    item_id: str,
    item_split_map: Dict[str, str],
    requested_split: str,
    profile_name: str | None = None,
) -> str:
    if item_id in item_split_map:
        return item_split_map[item_id]
    if profile_name:
        return f"profile:{profile_name}"
    return requested_split


def load_profiles(benchmark_dir: Path) -> Dict[str, dict]:
    profiles_path = benchmark_dir / "profiles.json"
    if not profiles_path.exists():
        return {}
    with open(profiles_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("profiles", {})


def resolve_profile_item_ids(profile: dict, splits: dict) -> List[str]:
    if profile.get("item_ids"):
        return list(profile["item_ids"])
    if profile.get("split"):
        return requested_item_ids(splits, profile["split"])
    raise ValueError("Profile must define either item_ids or split.")


def validate_profile_items(profile_name: str, profile: dict, items: Dict[str, dict], splits: dict) -> Dict[str, List[str]]:
    constraints = profile.get("question_constraints", {})
    if not constraints:
        return {}

    violations: Dict[str, List[str]] = {}
    for item_id in resolve_profile_item_ids(profile, splits):
        item = items.get(item_id)
        if item is None:
            violations.setdefault(item_id, []).append("missing_item")
            continue

        item_violations: List[str] = []
        must_cover = item.get("required_evidence", {}).get("must_cover", [])
        min_sources = item.get("required_evidence", {}).get("min_distinct_sources", 0)
        query_types = set(item.get("query_type", []))
        target_periods = item.get("target_periods", [])

        allowed_categories = set(constraints.get("allowed_categories", []))
        if allowed_categories and item.get("category") not in allowed_categories:
            item_violations.append(f"category:{item.get('category')}")

        allowed_difficulties = set(constraints.get("allowed_difficulties", []))
        if allowed_difficulties and item.get("difficulty") not in allowed_difficulties:
            item_violations.append(f"difficulty:{item.get('difficulty')}")

        max_must_cover = constraints.get("max_must_cover")
        if max_must_cover is not None and len(must_cover) > max_must_cover:
            item_violations.append(f"must_cover>{max_must_cover}")

        max_distinct_sources = constraints.get("max_distinct_sources")
        if max_distinct_sources is not None and min_sources > max_distinct_sources:
            item_violations.append(f"min_distinct_sources>{max_distinct_sources}")

        allowed_query_types = set(constraints.get("allowed_query_types", []))
        if allowed_query_types and not query_types.issubset(allowed_query_types):
            item_violations.append("query_type_outside_allowed_set")

        if constraints.get("require_target_periods") and not target_periods:
            item_violations.append("missing_target_periods")

        if item_violations:
            violations[item_id] = item_violations

    return violations


def selection_item_ids(
    splits: dict,
    split: str,
    profiles: Dict[str, dict],
    profile_name: str | None,
) -> tuple[List[str], dict | None]:
    if not profile_name:
        return requested_item_ids(splits, split), None
    if profile_name not in profiles:
        raise ValueError(f"Unknown benchmark profile: {profile_name}")
    profile = profiles[profile_name]
    return resolve_profile_item_ids(profile, splits), profile


def find_missing_item_ids(item_ids: List[str], items: Dict[str, dict]) -> List[str]:
    return [item_id for item_id in item_ids if item_id not in items]


def compute_group_averages(rows: List[dict], field: str) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get(field, "unknown"), []).append(row)
    return {name: compute_averages(group_rows) for name, group_rows in sorted(grouped.items())}


def compute_optional_metric_average(rows: List[dict], field: str) -> float:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def build_portfolio_summary(rows: List[dict]) -> dict:
    category_counts = Counter(row.get("category", "unknown") for row in rows)
    difficulty_counts = Counter(row.get("difficulty", "unknown") for row in rows)
    query_type_counts = Counter()
    question_family_counts = Counter(row.get("question_family", "general") for row in rows)
    answer_status_counts = Counter(row.get("answer_status", "unknown") for row in rows)
    for row in rows:
        for query_type in row.get("query_type", []):
            query_type_counts[query_type] += 1
    return {
        "item_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "query_type_counts": dict(sorted(query_type_counts.items())),
        "question_family_counts": dict(sorted(question_family_counts.items())),
        "question_type_counts": dict(sorted(question_family_counts.items())),
        "answer_status_counts": dict(sorted(answer_status_counts.items())),
    }


def build_failure_type_distribution(rows: List[dict]) -> dict:
    counts = Counter()
    for row in rows:
        for tag, count in row.get("question_error_summary", {}).get("claim_error_counts", {}).items():
            counts[tag] += count
        for reason in row.get("question_error_summary", {}).get("controller_failure_reasons", []):
            counts[f"controller::{reason}"] += 1
        most_severe = row.get("question_error_summary", {}).get("most_severe_failure_type")
        if most_severe:
            counts[f"question::{most_severe}"] += 1
    return dict(sorted(counts.items()))


def build_run_summary(rows: List[dict], averages: dict) -> dict:
    status_counts = Counter(row.get("answer_status", "unknown") for row in rows)
    summary = {
        "total_items": len(rows),
        "answered_items": status_counts.get("answered", 0),
        "partial_items": status_counts.get("partial", 0),
        "abstained_items": status_counts.get("abstained", 0),
        "primary_metrics": {metric: averages.get(metric, 0.0) for metric in PRIMARY_METRICS},
        "secondary_metrics": {metric: averages.get(metric, 0.0) for metric in SECONDARY_METRICS},
        "failure_type_distribution": build_failure_type_distribution(rows),
    }
    return summary


def assess_target_metrics(averages: dict, target_metrics: Dict[str, dict]) -> dict:
    assessment: Dict[str, Any] = {"all_met": True, "metrics": {}}
    for metric_name, spec in target_metrics.items():
        target = float(spec["target"])
        direction = spec["direction"]
        actual = float(averages.get(metric_name, 0.0))
        if direction == "at_least":
            met = actual >= target
        elif direction == "at_most":
            met = actual <= target
        else:
            raise ValueError(f"Unsupported target direction: {direction}")
        assessment["metrics"][metric_name] = {
            "actual": actual,
            "target": target,
            "direction": direction,
            "met": met,
        }
        assessment["all_met"] = assessment["all_met"] and met
    return assessment


def compute_averages(rows: List[dict]) -> dict:
    if not rows:
        return {
            "strong_support_rate": 0.0,
            "weak_support_rate": 0.0,
            "fabricated_citation_rate": 0.0,
            "contradiction_rate": 0.0,
            "numeric_exact_match_rate": 0.0,
            "primary_source_support_rate": 0.0,
            "unsafe_publish_rate": 0.0,
            "unsupported_numeric_claim_rate": 0.0,
            "primary_source_missing_rate": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "atomic_claim_rate": 0.0,
            "claim_extract_success_rate": 0.0,
            "period_match_rate": 0.0,
            "currency_match_rate": 0.0,
            "directionality_match_rate": 0.0,
            "retrieval_hit": 0.0,
            "citation_marker_coverage": 0.0,
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
            "required_fact_recall": 0.0,
            "avg_nli_score": 0.0,
            "answer_quality": 0.0,
            "subquestion_completion_rate": 0.0,
            "required_subquestion_coverage": 0.0,
            "required_slot_coverage": 0.0,
            "decision_trace_coverage": 0.0,
            "decision_replay_consistency": 0.0,
            "refused_subquestion_rate": 0.0,
            "hard_fact_completion_rate": 0.0,
            "semantic_completion_rate": 0.0,
            "wrong_entity_rate": 0.0,
            "wrong_period_rate": 0.0,
            "gold_answer_hit": 0.0,
            "gold_numeric_hit": 0.0,
            "gold_citation_hit": 0.0,
            "gold_semantic_similarity": 0.0,
            "gold_semantic_hit": 0.0,
        }

    abstained_rows = [row for row in rows if row.get("answer_status") == "abstained"]
    should_abstain_rows = [row for row in rows if row.get("should_abstain") is True]
    correct_abstentions = [row for row in abstained_rows if row.get("should_abstain") is True]
    return {
        "strong_support_rate": sum(row.get("strong_support_rate", 0.0) for row in rows) / len(rows),
        "weak_support_rate": sum(row.get("weak_support_rate", 0.0) for row in rows) / len(rows),
        "fabricated_citation_rate": sum(row.get("fabricated_citation_rate", 0.0) for row in rows) / len(rows),
        "contradiction_rate": sum(row.get("contradiction_rate", 0.0) for row in rows) / len(rows),
        "numeric_exact_match_rate": sum(row.get("numeric_exact_match_rate", 0.0) for row in rows) / len(rows),
        "primary_source_support_rate": sum(row.get("primary_source_support_rate", 0.0) for row in rows) / len(rows),
        "unsafe_publish_rate": sum(row.get("unsafe_publish_rate", 0.0) for row in rows) / len(rows),
        "unsupported_numeric_claim_rate": sum(row.get("unsupported_numeric_claim_rate", 0.0) for row in rows) / len(rows),
        "primary_source_missing_rate": sum(row.get("primary_source_missing_rate", 0.0) for row in rows) / len(rows),
        "abstention_precision": len(correct_abstentions) / max(1, len(abstained_rows)),
        "abstention_recall": len(correct_abstentions) / max(1, len(should_abstain_rows)),
        "atomic_claim_rate": sum(row.get("atomic_claim_rate", 0.0) for row in rows) / len(rows),
        "claim_extract_success_rate": sum(row.get("claim_extract_success_rate", 0.0) for row in rows) / len(rows),
        "period_match_rate": sum(row.get("period_match_rate", 0.0) for row in rows) / len(rows),
        "currency_match_rate": sum(row.get("currency_match_rate", 0.0) for row in rows) / len(rows),
        "directionality_match_rate": sum(row.get("directionality_match_rate", 0.0) for row in rows) / len(rows),
        "retrieval_hit": sum(row.get("retrieval_hit", 0.0) for row in rows) / len(rows),
        "citation_marker_coverage": sum(row.get("citation_marker_coverage", 0.0) for row in rows) / len(rows),
        "verified_claim_coverage": sum(row.get("verified_claim_coverage", 0.0) for row in rows) / len(rows),
        "unsupported_claim_rate": sum(row.get("unsupported_claim_rate", 0.0) for row in rows) / len(rows),
        "required_fact_recall": sum(row.get("required_fact_recall", 0.0) for row in rows) / len(rows),
        "avg_nli_score": sum(row.get("avg_nli_score", 0.0) for row in rows) / len(rows),
        "answer_quality": sum(row.get("answer_quality", 0.0) for row in rows) / len(rows),
        "subquestion_completion_rate": sum(row.get("subquestion_completion_rate", 0.0) for row in rows) / len(rows),
        "required_subquestion_coverage": sum(row.get("required_subquestion_coverage", 0.0) for row in rows) / len(rows),
        "required_slot_coverage": sum(row.get("required_slot_coverage", 0.0) for row in rows) / len(rows),
        "decision_trace_coverage": sum(row.get("decision_trace_coverage", 0.0) for row in rows) / len(rows),
        "decision_replay_consistency": sum(row.get("decision_replay_consistency", 0.0) for row in rows) / len(rows),
        "refused_subquestion_rate": sum(row.get("refused_subquestion_rate", 0.0) for row in rows) / len(rows),
        "hard_fact_completion_rate": sum(row.get("hard_fact_completion_rate", 0.0) for row in rows) / len(rows),
        "semantic_completion_rate": sum(row.get("semantic_completion_rate", 0.0) for row in rows) / len(rows),
        "wrong_entity_rate": sum(row.get("wrong_entity_rate", 0.0) for row in rows) / len(rows),
        "wrong_period_rate": sum(row.get("wrong_period_rate", 0.0) for row in rows) / len(rows),
        "gold_answer_hit": sum(row.get("gold_answer_hit", 0.0) for row in rows) / len(rows),
        "gold_numeric_hit": sum(row.get("gold_numeric_hit", 0.0) for row in rows) / len(rows),
        "gold_citation_hit": sum(row.get("gold_citation_hit", 0.0) for row in rows) / len(rows),
        "gold_semantic_similarity": sum(row.get("gold_semantic_similarity", 0.0) for row in rows) / len(rows),
        "gold_semantic_hit": sum(row.get("gold_semantic_hit", 0.0) for row in rows) / len(rows),
    }


def build_payload(
    version: str,
    split: str,
    mode: str,
    companies: List[str],
    rows: List[dict],
    *,
    profile_name: str | None = None,
    profile: dict | None = None,
) -> dict:
    averages = compute_averages(rows)
    payload = {
        "version": version,
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "companies": companies,
        "averages": averages,
        "rows": rows,
        "by_category": compute_group_averages(rows, "category"),
        "by_question_family": compute_group_averages(rows, "question_family"),
        "by_question_type": compute_group_averages(rows, "question_family"),
        "by_answer_status": compute_group_averages(rows, "answer_status"),
        "portfolio_summary": build_portfolio_summary(rows),
        "run_summary": build_run_summary(rows, averages),
    }
    if profile_name:
        payload["profile"] = profile_name
    if profile:
        payload["profile_track"] = profile.get("track")
        payload["profile_purpose"] = profile.get("purpose")
        payload["profile_targets"] = profile.get("target_metrics", {})
        payload["profile_constraints"] = profile.get("question_constraints", {})
        if profile.get("target_metrics"):
            payload["target_assessment"] = assess_target_metrics(averages, profile["target_metrics"])
    if split == "all":
        payload["per_split"] = {
            split_name: {
                "count": len([row for row in rows if row["split"] == split_name]),
                "averages": compute_averages([row for row in rows if row["split"] == split_name]),
            }
            for split_name in ("dev", "test")
        }
    return payload


def load_existing_rows(output_path: Path, version: str, split: str, mode: str, profile_name: str | None = None) -> Dict[str, dict]:
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != version:
        raise ValueError(f"Resume output version mismatch: expected {version}, got {payload.get('version')}")
    if payload.get("split") != split:
        raise ValueError(f"Resume output split mismatch: expected {split}, got {payload.get('split')}")
    if payload.get("mode") != mode:
        raise ValueError(f"Resume output mode mismatch: expected {mode}, got {payload.get('mode')}")
    if payload.get("profile") != profile_name:
        raise ValueError(f"Resume output profile mismatch: expected {profile_name}, got {payload.get('profile')}")
    return {row["item_id"]: row for row in payload.get("rows", [])}


def load_previous_payload(output_path: Path, version: str, split: str, mode: str, profile_name: str | None = None) -> Optional[dict]:
    if not output_path.parent.exists():
        return None
    best_match: Optional[tuple[float, dict, Path]] = None
    for candidate in output_path.parent.glob("*.json"):
        if candidate == output_path or not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("version") != version or payload.get("split") != split or payload.get("mode") != mode:
            continue
        if payload.get("profile") != profile_name:
            continue
        mtime = candidate.stat().st_mtime
        if best_match is None or mtime > best_match[0]:
            best_match = (mtime, payload, candidate)
    if best_match is None:
        return None
    payload = dict(best_match[1])
    payload["_baseline_path"] = str(best_match[2])
    return payload


def build_metric_delta(current: dict, baseline: dict) -> dict:
    keys = sorted(set(current) | set(baseline))
    return {
        key: float(current.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in keys
    }


def write_payload(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_benchmark_report(output_path: Path, payload: dict) -> Path:
    artifact_root = benchmark_artifact_root(output_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / "benchmark_report.md"
    report_path.write_text(render_benchmark_report(payload), encoding="utf-8")
    return report_path


def clear_accelerator_cache(skip_gc: bool = False) -> None:
    if skip_gc:
        return
    if not skip_gc:
        gc.collect()
    if torch is None:
        return
    configured_device = (settings.inference_device or "auto").strip().lower()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        configured_device != "cpu"
        and getattr(torch, "mps", None)
        and getattr(torch, "backends", None)
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def run(
    version: str,
    split: str,
    output_path: Path,
    load_models: bool = False,
    resume: bool = False,
    profile_name: str | None = None,
) -> dict:
    benchmark_dir = settings.data_dir / "benchmark" / version
    items = {row["item_id"]: row for row in load_jsonl(benchmark_dir / "items.jsonl")}
    answers = {row["item_id"]: row for row in load_jsonl(benchmark_dir / "answers.jsonl")}
    evidences = {row["item_id"]: row["evidence"] for row in load_jsonl(benchmark_dir / "evidence.jsonl")}
    splits = json.loads((benchmark_dir / "splits.json").read_text(encoding="utf-8"))
    profiles = load_profiles(benchmark_dir)
    item_ids, profile = selection_item_ids(splits, split, profiles, profile_name)
    if profile:
        profile_violations = validate_profile_items(profile_name or "", profile, items, splits)
        if profile_violations:
            raise ValueError(f"Profile {profile_name} contains invalid items: {profile_violations}")
    missing_item_ids = find_missing_item_ids(item_ids, items)
    if missing_item_ids:
        missing_preview = missing_item_ids[:10]
        scope_label = f"profile {profile_name}" if profile_name else f"split {split}"
        raise ValueError(
            f"{scope_label} references {len(missing_item_ids)} unknown item_ids; sample={missing_preview}"
        )
    item_split_map = item_splits(splits, "all")
    companies = sorted({company for item_id in item_ids for company in items[item_id]["companies"]})
    agents: dict[str, BizIntelAgent] = {}
    extractor = ClaimExtractor()
    verifier = EvidenceVerifier(use_dummy_model=(settings.llm_mode == "stub" or not settings.openai_api_key))
    judge = LLMJudge()
    mode = describe_mode(load_models)
    existing_rows = load_existing_rows(output_path, version, split, mode, profile_name=profile_name) if resume else {}
    rows_by_item_id = dict(existing_rows)
    evidence_store_cache: Dict[str, Dict[str, List[str]]] = {}
    max_question_attempts = max(1, int(settings.benchmark_question_max_retries) + 1)

    for item_id in item_ids:
        if item_id in rows_by_item_id:
            continue
        item = items[item_id]
        answer_row = answers[item_id]
        gold_entries = evidences[item_id]
        item_companies = sorted(item["companies"])
        agent_key = benchmark_index_key(item_companies)
        if agent_key not in evidence_store_cache:
            evidence_store_cache[agent_key] = load_evidence_store_for_companies(item_companies)
        item_evidence_store = evidence_store_cache[agent_key]
        if agent_key not in agents:
            index_dir = ensure_benchmark_index(version, item_companies, load_models=load_models)
            agents[agent_key] = BizIntelAgent(
                index_dir=index_dir,
                load_models=load_models,
                verify_report=True,
            )
        agent = agents[agent_key]

        explicit_company = item_companies[0] if len(item_companies) == 1 else None
        attempt_count = 0
        last_error: str | None = None
        while True:
            attempt_count += 1
            try:
                result = agent.research(
                    item["query"],
                    company_id=explicit_company,
                    target_periods=item.get("target_periods", []),
                    required_slots=answer_row.get("must_cover", []),
                    required_source_types=item.get("required_source_types", []),
                    query_types=item.get("query_type", []),
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt_count < max_question_attempts and is_retryable_live_error(exc):
                    logger.warning(
                        "Retrying benchmark item %s after live API error (attempt %s/%s): %s",
                        item_id,
                        attempt_count,
                        max_question_attempts,
                        last_error,
                    )
                    clear_accelerator_cache(skip_gc=False)
                    continue
                raise
        memo_markdown = result["memo_markdown"]
        scorable_markdown = extract_scorable_markdown(memo_markdown)
        source_ids = [source.source_id for source in result["memo_object"].sources]
        metrics = score_markdown(
            scorable_markdown,
            build_required_facts(answer_row),
            item_evidence_store,
            question_text=item["query"],
            extractor=extractor,
            verifier=verifier,
            judge=judge,
        )
        gold_metrics = score_gold_answer_mapping(scorable_markdown, answer_row)
        research_metrics = score_research_trace(
            result,
            build_required_facts(answer_row),
            expected_companies=item_companies,
            target_periods=item.get("target_periods", []),
        )
        combined_metrics = {**metrics, **research_metrics, **gold_metrics}
        matched_count = len(matched_gold_docs(gold_entries, scorable_markdown, source_ids))
        min_sources = item["required_evidence"]["min_distinct_sources"]
        scope_supported = result["memo_object"].contract.get("scope_supported", True)
        combined_metrics["scope_supported"] = scope_supported
        question_family = classify_question_family(item["query"], item.get("query_type", []))
        answer_status_decision = derive_answer_status_decision(
            {**metrics, **research_metrics},
            scorable_markdown,
            question_text=item["query"],
            judge=judge,
        )
        answer_status = answer_status_decision["status"]
        should_abstain_flag = should_abstain(metrics, research_metrics)
        question_error_summary = summarize_question_errors(metrics, research_metrics)
        row = {
            "run_id": f"benchmark_{version}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "item_id": item_id,
            "split": resolve_item_split_label(item_id, item_split_map, split, profile_name=profile_name),
            "system": "bizintel_agent",
            "category": item["category"],
            "difficulty": item["difficulty"],
            "query_type": item.get("query_type", []),
            "question_family": question_family,
            "question_text": item["query"],
            "final_answer": memo_markdown,
            "scoreable_answer": scorable_markdown,
            "answer_status": answer_status,
            "answer_status_reason": answer_status_decision.get("reason"),
            "rule_answer_status": answer_status_decision.get("rule_status"),
            "rule_answer_status_reason": answer_status_decision.get("rule_reason"),
            "llm_answer_status_reviewed": answer_status_decision.get("llm_reviewed", False),
            "llm_answer_status_adjudicated": answer_status_decision.get("llm_adjudicated", False),
            "should_abstain": should_abstain_flag,
            "abstention_precision": (
                1.0 if answer_status == "abstained" and should_abstain_flag
                else 0.0 if answer_status == "abstained"
                else None
            ),
            "abstention_recall": (
                1.0 if should_abstain_flag and answer_status == "abstained"
                else 0.0 if should_abstain_flag
                else None
            ),
            "strong_support_rate": metrics.get("strong_support_rate", 0.0),
            "weak_support_rate": metrics.get("weak_support_rate", 0.0),
            "fabricated_citation_rate": metrics.get("fabricated_citation_rate", 0.0),
            "contradiction_rate": metrics.get("contradiction_rate", 0.0),
            "numeric_exact_match_rate": metrics.get("numeric_exact_match_rate", 0.0),
            "primary_source_support_rate": metrics.get("primary_source_support_rate", 0.0),
            "unsafe_publish_rate": metrics.get("unsafe_publish_rate", 0.0),
            "unsupported_numeric_claim_rate": metrics.get("unsupported_numeric_claim_rate", 0.0),
            "primary_source_missing_rate": metrics.get("primary_source_missing_rate", 0.0),
            "atomic_claim_rate": metrics.get("atomic_claim_rate", 0.0),
            "claim_extract_success_rate": metrics.get("claim_extract_success_rate", 0.0),
            "period_match_rate": metrics.get("period_match_rate", 0.0),
            "currency_match_rate": metrics.get("currency_match_rate", 0.0),
            "directionality_match_rate": metrics.get("directionality_match_rate", 0.0),
            "retrieval_hit": retrieval_hit(gold_entries, memo_markdown, source_ids, min_sources),
            "citation_marker_coverage": metrics.get("citation_marker_coverage", 0.0),
            "verified_claim_coverage": metrics.get("verified_claim_coverage", 0.0),
            "unsupported_claim_rate": metrics["unsupported_claim_rate"],
            "required_fact_recall": metrics["required_fact_recall"],
            "avg_nli_score": metrics["avg_nli_score"],
            "claim_error_counts": metrics.get("claim_error_counts", {}),
            "claim_error_rates": metrics.get("claim_error_rates", {}),
            "claim_type_counts": metrics.get("claim_type_counts", {}),
            "claim_type_rates": metrics.get("claim_type_rates", {}),
            "most_severe_failure_type": metrics.get("most_severe_failure_type"),
            "subquestion_completion_rate": research_metrics["subquestion_completion_rate"],
            "required_subquestion_coverage": research_metrics["required_subquestion_coverage"],
            "required_slot_coverage": research_metrics["required_slot_coverage"],
            "decision_trace_coverage": research_metrics["decision_trace_coverage"],
            "decision_replay_consistency": research_metrics["decision_replay_consistency"],
            "refused_subquestion_rate": research_metrics["refused_subquestion_rate"],
            "hard_fact_completion_rate": research_metrics["hard_fact_completion_rate"],
            "semantic_completion_rate": research_metrics["semantic_completion_rate"],
            "wrong_entity_rate": research_metrics["wrong_entity_rate"],
            "wrong_period_rate": research_metrics["wrong_period_rate"],
            "controller_failure_reasons": research_metrics["controller_failure_reasons"],
            "gold_answer_mode": gold_metrics["gold_answer_mode"],
            "gold_answer_hit": gold_metrics["gold_answer_hit"],
            "gold_numeric_hit": gold_metrics["gold_numeric_hit"],
            "gold_citation_hit": gold_metrics["gold_citation_hit"],
            "scope_supported": scope_supported,
            "answer_quality": answer_quality(item, combined_metrics, matched_count),
            "failure_tags": failure_tags(item, combined_metrics, scorable_markdown, gold_entries, source_ids, matched_count),
            "query": item["query"],
            "item_companies": item_companies,
            "sources_used": source_ids,
            "matched_gold_docs": sorted(matched_gold_docs(gold_entries, scorable_markdown, source_ids)),
            "claim_results": metrics.get("claim_diagnostics", []),
            "final_cited_evidence": summarize_final_cited_evidence(metrics.get("claim_diagnostics", [])),
            "retrieval_candidates": summarize_retrieval_candidates(result),
            "question_error_summary": question_error_summary,
            "question_attempts": attempt_count,
            "runtime_error": last_error,
        }
        row.update(write_case_artifacts(output_path, item, row, result))
        rows_by_item_id[item_id] = row
        checkpoint_rows = [rows_by_item_id[current_item_id] for current_item_id in item_ids if current_item_id in rows_by_item_id]
        write_payload(
            output_path,
            build_payload(version, split, mode, companies, checkpoint_rows, profile_name=profile_name, profile=profile),
        )
        clear_accelerator_cache(skip_gc=(settings.llm_mode == "stub" and not load_models))

    rows = [rows_by_item_id[item_id] for item_id in item_ids if item_id in rows_by_item_id]
    payload = build_payload(version, split, mode, companies, rows, profile_name=profile_name, profile=profile)
    baseline_payload = load_previous_payload(output_path, version, split, mode, profile_name=profile_name)
    if baseline_payload is not None:
        payload["baseline_path"] = baseline_payload.get("_baseline_path")
        payload["baseline_timestamp"] = baseline_payload.get("timestamp")
        payload["delta_from_baseline"] = build_metric_delta(payload.get("averages", {}), baseline_payload.get("averages", {}))
        payload["run_summary"]["delta_from_baseline"] = payload["delta_from_baseline"]
    payload["artifact_root"] = str(benchmark_artifact_root(output_path))
    write_payload(output_path, payload)
    payload["report_path"] = str(write_benchmark_report(output_path, payload))
    write_payload(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local benchmark suite.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument("--profile", default=None, help="Optional benchmark profile name from data/benchmark/<version>/profiles.json.")
    parser.add_argument("--load-models", action="store_true", help="Load real retrieval/reranker models instead of dummy retrieval.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file with matching version/split/mode.")
    parser.add_argument("--strict-live", action="store_true", help="Disable LLM fallback on API errors and retry whole benchmark questions instead.")
    parser.add_argument("--item-retries", type=int, default=None, help="Retry count per benchmark question when a retryable live API error occurs.")
    parser.add_argument("--llm-timeout-seconds", type=float, default=None, help="Override OpenAI-compatible request timeout for this run.")
    parser.add_argument("--llm-request-retries", type=int, default=None, help="Override per-request API retry count for this run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir.parent / "eval" / "results" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    args = parser.parse_args()

    if args.strict_live:
        settings.strict_live_mode = True
        settings.llm_mode = "live"
    if args.item_retries is not None:
        settings.benchmark_question_max_retries = max(0, args.item_retries)
    if args.llm_timeout_seconds is not None:
        settings.llm_request_timeout_seconds = max(1.0, args.llm_timeout_seconds)
    if args.llm_request_retries is not None:
        settings.llm_request_max_retries = max(0, args.llm_request_retries)

    payload = run(
        args.version,
        args.split,
        args.output,
        load_models=args.load_models,
        resume=args.resume,
        profile_name=args.profile,
    )
    print(json.dumps({"output": str(args.output), "rows": len(payload["rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
