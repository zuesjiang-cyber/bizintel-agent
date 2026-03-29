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
from typing import Any, Dict, List

from agent.config import settings
from agent.orchestrator import BizIntelAgent
from eval.evaluator import extract_scorable_markdown, score_markdown, score_research_trace
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
    score = 1
    if verified_claim_coverage(metrics) >= 0.55:
        score += 1
    if metrics["unsupported_claim_rate"] <= 0.35:
        score += 1
    if metrics["required_fact_recall"] >= 0.5:
        score += 1
    if matched_count >= item["required_evidence"]["min_distinct_sources"]:
        score += 1
    if verified_claim_coverage(metrics) >= 0.75 and metrics["unsupported_claim_rate"] <= 0.2:
        score = min(score + 1, 5)
    if metrics.get("required_subquestion_coverage", 1.0) < 0.5:
        score = min(score, 3)
    if metrics.get("required_slot_coverage", 1.0) < 0.5:
        score = min(score, 3)
    if metrics.get("decision_replay_consistency", 1.0) < 1.0:
        score = min(score, 4)
    return min(score, 5)


def failure_tags(item: dict, metrics: dict, memo_markdown: str, gold_entries: List[dict], source_ids: List[str], matched_count: int) -> List[str]:
    tags: List[str] = []
    if verified_claim_coverage(metrics) < 0.4:
        tags.append("R1_missing_primary_source")
    if metrics["unsupported_claim_rate"] > 0.4:
        tags.append("S5_overclaim")
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


def compute_group_averages(rows: List[dict], field: str) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get(field, "unknown"), []).append(row)
    return {name: compute_averages(group_rows) for name, group_rows in sorted(grouped.items())}


def build_portfolio_summary(rows: List[dict]) -> dict:
    category_counts = Counter(row.get("category", "unknown") for row in rows)
    difficulty_counts = Counter(row.get("difficulty", "unknown") for row in rows)
    query_type_counts = Counter()
    for row in rows:
        for query_type in row.get("query_type", []):
            query_type_counts[query_type] += 1
    return {
        "item_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "query_type_counts": dict(sorted(query_type_counts.items())),
    }


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
        }
    return {
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
        "portfolio_summary": build_portfolio_summary(rows),
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


def write_payload(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def clear_accelerator_cache(skip_gc: bool = False) -> None:
    if skip_gc:
        return
    if not skip_gc:
        gc.collect()
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch, "mps", None):
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
    item_split_map = item_splits(splits, "all")
    companies = sorted({company for item_id in item_ids for company in items[item_id]["companies"]})
    agents: dict[str, BizIntelAgent] = {}
    extractor = ClaimExtractor()
    verifier = EvidenceVerifier(use_dummy_model=(settings.llm_mode == "stub" or not settings.openai_api_key))
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
            extractor=extractor,
            verifier=verifier,
        )
        research_metrics = score_research_trace(
            result,
            build_required_facts(answer_row),
            expected_companies=item_companies,
            target_periods=item.get("target_periods", []),
        )
        combined_metrics = {**metrics, **research_metrics}
        matched_count = len(matched_gold_docs(gold_entries, scorable_markdown, source_ids))
        min_sources = item["required_evidence"]["min_distinct_sources"]
        scope_supported = result["memo_object"].contract.get("scope_supported", True)
        combined_metrics["scope_supported"] = scope_supported
        row = {
            "run_id": f"benchmark_{version}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "item_id": item_id,
            "split": item_split_map[item_id],
            "system": "bizintel_agent",
            "category": item["category"],
            "difficulty": item["difficulty"],
            "query_type": item.get("query_type", []),
            "retrieval_hit": retrieval_hit(gold_entries, memo_markdown, source_ids, min_sources),
            "citation_marker_coverage": metrics.get("citation_marker_coverage", 0.0),
            "verified_claim_coverage": metrics.get("verified_claim_coverage", 0.0),
            "unsupported_claim_rate": metrics["unsupported_claim_rate"],
            "required_fact_recall": metrics["required_fact_recall"],
            "avg_nli_score": metrics["avg_nli_score"],
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
            "scope_supported": scope_supported,
            "answer_quality": answer_quality(item, combined_metrics, matched_count),
            "failure_tags": failure_tags(item, combined_metrics, scorable_markdown, gold_entries, source_ids, matched_count),
            "query": item["query"],
            "item_companies": item_companies,
            "sources_used": source_ids,
            "matched_gold_docs": sorted(matched_gold_docs(gold_entries, scorable_markdown, source_ids)),
            "question_attempts": attempt_count,
            "runtime_error": last_error,
        }
        rows_by_item_id[item_id] = row
        checkpoint_rows = [rows_by_item_id[current_item_id] for current_item_id in item_ids if current_item_id in rows_by_item_id]
        write_payload(
            output_path,
            build_payload(version, split, mode, companies, checkpoint_rows, profile_name=profile_name, profile=profile),
        )
        clear_accelerator_cache(skip_gc=(settings.llm_mode == "stub" and not load_models))

    rows = [rows_by_item_id[item_id] for item_id in item_ids if item_id in rows_by_item_id]
    payload = build_payload(version, split, mode, companies, rows, profile_name=profile_name, profile=profile)
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
