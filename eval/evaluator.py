"""
BizIntel Agent evaluator with a same-corpus plain-LLM baseline.

Benchmark design:
- same company pack
- same query
- same model
- same verifier
- different generation path
"""

import getpass
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.artifacts import build_trace_payload
from agent.config import settings
from agent.llm_utils import build_openai_client, generate_text_response
from agent.orchestrator import BizIntelAgent
from agent.prompts.synthesis import MEMO_SYSTEM_PROMPT
from eval.llm_judge import HARD_BLOCKING_ERROR_TAGS, LLMJudge
from tools.doc_parser import load_company_pack
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier
from agent.schemas import Claim

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SLOT_TOKEN_ALIASES = {
    "revenue": {"revenue", "sales", "topline", "top", "line", "net", "sales"},
    "profitability": {"profitability", "margin", "margins", "ebitda", "gross", "operating", "profit"},
    "management": {"management", "commentary", "tone", "outlook", "guidance", "demand", "framing", "narrative"},
    "risk": {"risk", "risks", "headwind", "headwinds", "challenge", "bear"},
    "catalyst": {"catalyst", "catalysts", "driver", "drivers", "bull", "tailwind"},
    "comparison": {"comparison", "compare", "versus", "vs", "change", "shift"},
    "evidence": {"evidence", "support", "supported", "confidence", "distinction", "reconciliation"},
    "business": {"business", "model", "monetization", "mix", "product", "products"},
}

SOURCE_TIER_LABELS = {
    1: "filing",
    2: "official_results",
    3: "transcript",
    4: "other",
}
NUMERIC_CLAIM_KEYWORDS = {
    "ratio", "margin", "margins", "revenue", "sales", "income", "cash", "cash flow", "capex",
    "debt", "eps", "percent", "percentage", "highest", "lowest", "largest", "smallest", "most", "least",
}
COMPARISON_CLAIM_KEYWORDS = {
    "highest", "lowest", "largest", "smallest", "most", "least", "greater", "lower", "higher",
    "increase", "decrease", "grew", "declined", "rose", "fell", "compared", "versus", "vs", "ranking",
}
REASON_CLAIM_KEYWORDS = {
    "because", "due to", "driven by", "driver", "drivers", "reflecting", "primarily", "mainly", "reason",
}


def source_tier_for_source_type(source_type: str | None) -> int:
    lowered = (source_type or "").strip().lower()
    if lowered in {"annual_report", "quarterly_report"}:
        return 1
    if lowered in {"quarterly_results", "shareholder_letter"}:
        return 2
    if lowered in {"earnings_call_transcript"}:
        return 3
    return 4


def build_source_catalog(evidence_store: Dict[str, List[Any]]) -> Dict[str, dict]:
    catalog: Dict[str, dict] = {}
    for source_id, chunks in evidence_store.items():
        normalized_chunks = []
        if chunks and isinstance(chunks[0], dict):
            normalized_chunks = chunks
        elif chunks:
            normalized_chunks = [{"text": str(chunk), "source_id": source_id} for chunk in chunks]
        source_type = None
        is_primary = None
        for chunk in normalized_chunks:
            if source_type is None and chunk.get("source_type"):
                source_type = str(chunk.get("source_type"))
            if is_primary is None and chunk.get("is_primary") is not None:
                is_primary = bool(chunk.get("is_primary"))
        tier = source_tier_for_source_type(source_type)
        catalog[source_id] = {
            "source_type": source_type,
            "is_primary": is_primary,
            "source_tier": tier,
            "source_tier_label": SOURCE_TIER_LABELS[tier],
        }
    return catalog


def classify_claim_type(claim: Claim) -> str:
    if getattr(claim, "claim_type", None):
        mapped = {
            "numeric": "number",
            "descriptive": "fact",
            "comparative": "comparison",
            "causal": "reason",
            "management_commentary": "fact",
        }.get(claim.claim_type)
        if mapped:
            return mapped
    lowered = claim.text.lower()
    if any(keyword in lowered for keyword in REASON_CLAIM_KEYWORDS):
        return "reason"
    if any(keyword in lowered for keyword in COMPARISON_CLAIM_KEYWORDS):
        return "comparison"
    if claim.contains_numbers or any(keyword in lowered for keyword in NUMERIC_CLAIM_KEYWORDS):
        return "number"
    return "fact"


def claim_requires_exact_numeric_check(claim: Claim, claim_type: str) -> bool:
    if claim.contains_numbers:
        return True
    if claim_type == "comparison":
        lowered = claim.text.lower()
        return any(keyword in lowered for keyword in {"highest", "lowest", "largest", "smallest", "most", "least"})
    return False


def infer_numeric_check_result(claim: Claim, verification_result: Any) -> str:
    if not claim_requires_exact_numeric_check(claim, classify_claim_type(claim)):
        return "not_applicable"
    if verification_result.numeric_verified is True:
        if verification_result.failure_reason is None:
            return "exact"
        return "derived_exact"
    if verification_result.numeric_verified is False:
        return "mismatch"
    return "not_checked"


def _extract_unit_tokens(text: str) -> set[str]:
    lowered = text.lower()
    units = set(re.findall(r"\b(?:billion|million|thousand|bn|mn|b|m|k|usd|dollars?)\b", lowered))
    if "$" in text:
        units.add("$")
    if "%" in text:
        units.add("%")
    return units


def _extract_period_tokens(text: str) -> set[str]:
    return {
        match.group(0).lower().replace(" ", "")
        for match in re.finditer(r"\b(?:q[1-4]\s*20\d{2}|fy\s*20\d{2}|20\d{2})\b", text, re.IGNORECASE)
    }


def _extract_direction_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = {
        "increase", "increased", "decrease", "decreased", "up", "down", "grew", "growth",
        "decline", "declined", "rose", "fell", "highest", "lowest", "largest", "smallest",
        "most", "least", "positive", "negative",
    }
    return {token for token in tokens if token in lowered}


def derive_claim_error_tags(
    claim: Claim,
    verification_result: Any,
    source_catalog: Dict[str, dict],
) -> List[str]:
    tags: List[str] = []
    failure_reason = verification_result.failure_reason
    support_text = " ".join(verification_result.supporting_evidence or [])
    claim_type = classify_claim_type(claim)

    if failure_reason == "missing_citation" or claim.cited_sources == ["no_citation"]:
        tags.append("missing_citation")
    if failure_reason == "bad_source_id":
        tags.extend(["source_not_found", "fabricated_citation"])
    if failure_reason == "no_supporting_chunk":
        tags.extend(["unsupported", "fabricated_citation"])

    if failure_reason == "numeric_mismatch":
        tags.append("numeric_mismatch")
        claim_units = _extract_unit_tokens(claim.text)
        support_units = _extract_unit_tokens(support_text)
        if getattr(verification_result, "unit_verified", None) is False or (claim_units and support_units and not (claim_units & support_units)):
            tags.append("unit_mismatch")
        claim_periods = _extract_period_tokens(claim.text)
        support_periods = _extract_period_tokens(support_text)
        if getattr(verification_result, "period_verified", None) is False or (claim_periods and support_periods and not (claim_periods & support_periods)):
            tags.append("period_mismatch")
        claim_directions = _extract_direction_tokens(claim.text)
        support_directions = _extract_direction_tokens(support_text)
        if getattr(verification_result, "directionality_verified", None) is False or (claim_directions and support_directions and not (claim_directions & support_directions)):
            tags.extend(["direction_mismatch", "contradiction"])

    if failure_reason == "period_mismatch" or getattr(verification_result, "period_verified", None) is False:
        tags.append("period_mismatch")
    if failure_reason == "currency_mismatch" or getattr(verification_result, "currency_verified", None) is False:
        tags.append("currency_mismatch")
    if failure_reason == "unit_mismatch" or getattr(verification_result, "unit_verified", None) is False:
        tags.append("unit_mismatch")
    if failure_reason == "direction_mismatch" or getattr(verification_result, "directionality_verified", None) is False:
        tags.append("direction_mismatch")
    if failure_reason == "contradiction" or getattr(verification_result, "contradiction_detected", None):
        tags.append("contradiction")

    if failure_reason == "primary_source_missing":
        tags.append("primary_source_missing")
    if failure_reason == "low_entailment":
        if verification_result.nli_score < 0.2 and claim.cited_sources and claim.cited_sources != ["no_citation"]:
            tags.append("fabricated_citation")
        else:
            tags.append("weak_support")

    if claim_type == "reason" and verification_result.confidence.value != "strong":
        tags.append("reason_not_supported")

    best_tier = min(
        (source_catalog.get(source_id, {}).get("source_tier", 4) for source_id in verification_result.supporting_source_ids or claim.cited_sources),
        default=4,
    )
    if claim_requires_exact_numeric_check(claim, claim_type) and best_tier > 2:
        tags.append("primary_source_missing")

    if verification_result.confidence.value == "unsupported" and not tags:
        tags.append("unsupported")
    if verification_result.confidence.value == "contradicted" and "contradiction" not in tags:
        tags.append("contradiction")

    if verification_result.confidence.value == "weak" and "weak_support" not in tags:
        tags.append("weak_support")

    return sorted(set(tags))


def determine_support_label(claim: Claim, verification_result: Any, error_tags: List[str]) -> str:
    non_blocking_tags = {"primary_source_missing"}
    blocking_error_tags = [tag for tag in error_tags if tag not in non_blocking_tags]
    if "contradiction" in error_tags:
        return "contradiction"
    if "fabricated_citation" in error_tags:
        return "fabricated_citation"
    if verification_result.confidence.value == "strong" and not blocking_error_tags:
        return "strong_support"
    if verification_result.confidence.value in {"moderate", "weak"} or "weak_support" in error_tags:
        return "weak_support"
    return "unsupported"


def build_claim_diagnostics(claims, verification_results, evidence_store: Dict[str, List[Any]]) -> List[dict]:
    source_catalog = build_source_catalog(evidence_store)
    rows = []
    for claim, result in zip(claims, verification_results):
        claim_type = classify_claim_type(claim)
        supporting_source_ids = result.supporting_source_ids or claim.cited_sources
        supporting_tiers = [
            source_catalog.get(source_id, {"source_tier": 4, "source_tier_label": SOURCE_TIER_LABELS[4]})
            for source_id in supporting_source_ids
        ]
        best_source_tier = min((entry["source_tier"] for entry in supporting_tiers), default=4)
        numeric_check_result = infer_numeric_check_result(claim, result)
        error_tags = derive_claim_error_tags(claim, result, source_catalog)
        support_label = determine_support_label(claim, result, error_tags)
        rows.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.text,
                "claim_type": claim_type,
                "cited_sources": claim.cited_sources,
                "cited_chunks": claim.cited_chunks,
                "support_label": support_label,
                "confidence": result.confidence.value,
                "risk_level": getattr(claim, "risk_level", "medium"),
                "atomicity": getattr(claim, "atomicity", True),
                "requires_primary_source": getattr(claim, "requires_primary_source", False),
                "entailment_score": result.nli_score,
                "nli_score": result.nli_score,
                "numeric_check_result": numeric_check_result,
                "numeric_verified": result.numeric_verified,
                "failure_reason": result.failure_reason,
                "failure_stage": getattr(result, "failure_stage", None),
                "supporting_source_ids": result.supporting_source_ids,
                "supporting_chunk_ids": result.supporting_chunk_ids,
                "supporting_evidence": result.supporting_evidence,
                "source_tier": best_source_tier,
                "source_tier_label": SOURCE_TIER_LABELS[best_source_tier],
                "source_tiers": supporting_tiers,
                "primary_source_supported": result.primary_source_supported,
                "period_verified": getattr(result, "period_verified", None),
                "currency_verified": getattr(result, "currency_verified", None),
                "unit_verified": getattr(result, "unit_verified", None),
                "directionality_verified": getattr(result, "directionality_verified", None),
                "contradiction_detected": getattr(result, "contradiction_detected", None),
                "verdict_trace": getattr(result, "verdict_trace", {}),
                "review_notes": getattr(result, "review_notes", []),
                "error_tags": error_tags,
                "explanation": result.explanation,
            }
        )
    return rows


def apply_llm_claim_adjudication(
    claim_diagnostics: List[dict],
    evidence_store: Dict[str, List[Any]],
    *,
    question_text: str = "",
    judge: Optional[LLMJudge] = None,
) -> List[dict]:
    judge = judge or LLMJudge()
    rows: List[dict] = []
    for row in claim_diagnostics:
        updated = dict(row)
        updated.setdefault("rule_support_label", row.get("support_label"))
        updated.setdefault("llm_reviewed", False)
        updated.setdefault("llm_adjudicated", False)
        if set(updated.get("error_tags", [])) & HARD_BLOCKING_ERROR_TAGS:
            rows.append(updated)
            continue
        decision = judge.adjudicate_claim_support(
            question_text=question_text,
            claim_diagnostic=updated,
            evidence_store=evidence_store,
        )
        if not decision:
            rows.append(updated)
            continue

        updated["llm_reviewed"] = True
        updated["llm_support_label"] = decision["support_label"]
        updated["llm_judge_reason"] = decision.get("reason", "")
        if decision["support_label"] != updated.get("support_label"):
            updated["llm_adjudicated"] = True
            updated["support_label"] = decision["support_label"]
            if decision["support_label"] == "strong_support":
                updated["confidence"] = "strong"
                updated["failure_reason"] = None if updated.get("failure_reason") == "low_entailment" else updated.get("failure_reason")
                updated["error_tags"] = [
                    tag
                    for tag in updated.get("error_tags", [])
                    if tag not in {"weak_support", "unsupported", "reason_not_supported"}
                ]
            elif decision["support_label"] == "weak_support":
                updated["confidence"] = "moderate" if updated.get("confidence") == "unsupported" else updated.get("confidence")
                updated["error_tags"] = [
                    tag
                    for tag in updated.get("error_tags", [])
                    if tag != "unsupported"
                ]
                if "weak_support" not in updated["error_tags"]:
                    updated["error_tags"].append("weak_support")
            else:
                updated["confidence"] = "unsupported"
                if "unsupported" not in updated.get("error_tags", []):
                    updated.setdefault("error_tags", []).append("unsupported")
        rows.append(updated)
    return rows


def load_pack_context(company: str) -> Tuple[str, Dict[str, List[str]]]:
    company_dir = settings.company_packs_dir / company
    documents = load_company_pack(company_dir)
    context_parts = []
    evidence_store: Dict[str, List[str]] = {}

    for meta, text in documents:
        evidence_store.setdefault(meta.source_id, []).append(text)
        context_parts.append(
            "\n".join(
                [
                    f"[Source: {meta.source_id}]",
                    f"Title: {meta.title}",
                    f"Date: {meta.date or 'Unknown'}",
                    text,
                ]
            )
        )

    return "\n\n---\n\n".join(context_parts), evidence_store


def required_fact_recall(markdown: str, required_facts: List[str]) -> float:
    if not required_facts:
        return 1.0
    markdown_lower = markdown.lower()
    found = [fact for fact in required_facts if fact.lower() in markdown_lower]
    return len(found) / len(required_facts)


def _normalized_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _tokenize_with_aliases(value: str | None) -> set[str]:
    normalized = _normalized_text(value)
    if not normalized:
        return set()
    tokens = set(normalized.split())
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SLOT_TOKEN_ALIASES.get(token, set()))
    return expanded


def _status_value(status: Any) -> str:
    return getattr(status, "value", str(status))


def _lane_value(lane: Any) -> str:
    return getattr(lane, "value", str(lane))


def _subquestion_covers_slot(result: Any, required_slot: str) -> bool:
    slot_tokens = _tokenize_with_aliases(required_slot)
    if not slot_tokens:
        return False

    subquestion = getattr(result, "subquestion", None)
    haystacks = [
        getattr(subquestion, "fact_slot", ""),
        getattr(subquestion, "metric_family", ""),
        getattr(subquestion, "text", ""),
        getattr(result, "answer_text", ""),
        getattr(result, "supported_content", ""),
    ]
    hay_tokens = set()
    for item in haystacks:
        hay_tokens.update(_tokenize_with_aliases(item))
    if not hay_tokens:
        return False

    overlap = slot_tokens & hay_tokens
    required_overlap = max(1, min(2, len(slot_tokens)))
    return len(overlap) >= required_overlap


def score_research_trace(
    result: dict,
    required_slots: List[str],
    *,
    expected_companies: List[str] | None = None,
    target_periods: List[str] | None = None,
) -> dict:
    subquestion_results = list(result.get("subquestion_results") or [])
    replay = result.get("research_trace")
    if not subquestion_results:
        return {
            "subquestion_count": 0,
            "required_subquestion_count": 0,
            "subquestion_completion_rate": 0.0,
            "required_subquestion_coverage": 0.0,
            "required_slot_coverage": 0.0 if required_slots else 1.0,
            "decision_trace_coverage": 0.0,
            "decision_replay_consistency": 0.0,
            "refused_subquestion_rate": 0.0,
            "hard_fact_completion_rate": 0.0,
            "semantic_completion_rate": 0.0,
            "wrong_entity_rate": 0.0,
            "wrong_period_rate": 0.0,
            "controller_failure_reasons": ["no_subquestions"],
        }

    completed = [item for item in subquestion_results if _status_value(item.status) == "completed"]
    refused = [item for item in subquestion_results if _status_value(item.status) == "refused"]
    required = [
        item for item in subquestion_results
        if getattr(getattr(item, "subquestion", None), "required", True)
    ]
    required_completed = [item for item in required if _status_value(item.status) == "completed"]

    required_slot_hits = [
        slot for slot in required_slots
        if any(_subquestion_covers_slot(item, slot) for item in required_completed)
    ]

    hard_questions = [
        item for item in subquestion_results
        if _lane_value(getattr(getattr(item, "subquestion", None), "lane", "")) == "hard_fact"
    ]
    semantic_questions = [
        item for item in subquestion_results
        if _lane_value(getattr(getattr(item, "subquestion", None), "lane", "")) == "semantic"
    ]

    decisions = list(getattr(replay, "decisions", []) or [])
    decisions_by_question: Dict[str, int] = {}
    for decision in decisions:
        question_id = getattr(decision, "question_id", "")
        if question_id:
            decisions_by_question[question_id] = decisions_by_question.get(question_id, 0) + 1

    replay_subquestions = list(getattr(replay, "subquestions", []) or [])
    replay_status_by_id = {
        getattr(item, "question_id", ""): _status_value(getattr(item, "status", ""))
        for item in replay_subquestions
        if getattr(item, "question_id", "")
    }
    result_status_by_id = {
        getattr(getattr(item, "subquestion", None), "question_id", ""): _status_value(item.status)
        for item in subquestion_results
        if getattr(getattr(item, "subquestion", None), "question_id", "")
    }
    replay_consistent = (
        bool(replay_subquestions)
        and replay_status_by_id == result_status_by_id
        and all(question_id in decisions_by_question for question_id in result_status_by_id)
    )

    expected_company_set = {_normalized_text(company) for company in expected_companies or [] if company}
    target_period_set = {_normalized_text(period) for period in target_periods or [] if period}
    evidence_chunks = [chunk for item in subquestion_results for chunk in getattr(item, "evidence", [])]
    wrong_entity = 0
    wrong_period = 0
    checked_period_chunks = 0
    for chunk in evidence_chunks:
        chunk_company = _normalized_text(getattr(chunk, "company", ""))
        if expected_company_set and chunk_company and chunk_company not in expected_company_set:
            wrong_entity += 1
        chunk_period = _normalized_text(getattr(chunk, "period", ""))
        if target_period_set and chunk_period:
            checked_period_chunks += 1
            if chunk_period not in target_period_set:
                wrong_period += 1

    controller_failure_reasons = sorted(
        {
            getattr(item, "refusal_reason", "").strip()
            for item in refused
            if getattr(item, "refusal_reason", "").strip()
        }
    )

    return {
        "subquestion_count": len(subquestion_results),
        "required_subquestion_count": len(required),
        "subquestion_completion_rate": len(completed) / len(subquestion_results),
        "required_subquestion_coverage": len(required_completed) / max(1, len(required)),
        "required_slot_coverage": (
            len(required_slot_hits) / len(required_slots)
            if required_slots
            else len(required_completed) / max(1, len(required))
        ),
        "decision_trace_coverage": len(decisions_by_question) / max(1, len(result_status_by_id)),
        "decision_replay_consistency": 1.0 if replay_consistent else 0.0,
        "refused_subquestion_rate": len(refused) / len(subquestion_results),
        "hard_fact_completion_rate": (
            len([item for item in hard_questions if _status_value(item.status) == "completed"]) / max(1, len(hard_questions))
        ),
        "semantic_completion_rate": (
            len([item for item in semantic_questions if _status_value(item.status) == "completed"]) / max(1, len(semantic_questions))
        ),
        "wrong_entity_rate": wrong_entity / max(1, len(evidence_chunks)) if evidence_chunks else 0.0,
        "wrong_period_rate": wrong_period / max(1, checked_period_chunks) if checked_period_chunks else 0.0,
        "controller_failure_reasons": controller_failure_reasons,
    }


def extract_scorable_markdown(markdown: str) -> str:
    scorable = markdown
    exec_summary_idx = scorable.find("## Executive Summary")
    if exec_summary_idx != -1:
        scorable = scorable[exec_summary_idx:]

    sources_marker = "\n---\n## Sources"
    if sources_marker in scorable:
        scorable = scorable.split(sources_marker, 1)[0]

    verification_marker = "\n## Verification Summary"
    if verification_marker in scorable:
        scorable = scorable.split(verification_marker, 1)[0]

    filtered_lines = []
    skip_coverage_contract = False
    skip_evidence_gaps = False
    for line in scorable.splitlines():
        stripped = line.strip()
        if stripped == "## Coverage Contract":
            skip_coverage_contract = True
            continue
        if stripped == "## Evidence Gaps":
            skip_evidence_gaps = True
            continue
        if skip_coverage_contract and stripped.startswith("## "):
            skip_coverage_contract = False
        if skip_evidence_gaps and stripped.startswith("## "):
            skip_evidence_gaps = False
        if skip_coverage_contract:
            continue
        if skip_evidence_gaps:
            continue
        if stripped.startswith("*Section confidence:"):
            continue
        if stripped.startswith("*Verified section support:"):
            continue
        if stripped.startswith("*Offline demo heuristic support:"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def fallback_extract_cited_lines(text: str, section_name: str, extractor: ClaimExtractor) -> List[Claim]:
    claims: List[Claim] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "[Source:" not in line and "[Chunk:" not in line:
            continue
        stripped = line.lstrip("-• ").strip()
        if extractor._is_filler(stripped) or extractor._is_subjective(stripped) or extractor._is_conditional(stripped) or extractor._is_insufficient(stripped):
            continue

        cited_sources = [s.strip() for s in extractor.CITATION_PATTERN.findall(stripped)]
        cited_chunks = [c.strip() for c in extractor.CHUNK_CITATION_PATTERN.findall(stripped)]
        clean_text = extractor.CITATION_PATTERN.sub("", stripped)
        clean_text = extractor.CHUNK_CITATION_PATTERN.sub("", clean_text).lstrip("-• ").strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        if len(clean_text) < 15:
            continue

        numbers = extractor.NUMBER_PATTERN.findall(clean_text)
        claim_id = f"fallback-{abs(hash((section_name, clean_text))) % 10**10:010d}"
        claims.append(
            Claim(
                claim_id=claim_id,
                text=clean_text,
                section=section_name,
                cited_sources=cited_sources or ["no_citation"],
                cited_chunks=cited_chunks,
                contains_numbers=bool(numbers),
                extracted_numbers=numbers,
                specificity_score=0.3,
            )
        )
    return claims


def has_unscorable_answer_content(markdown: str) -> bool:
    stripped = markdown.strip()
    if not stripped:
        return False
    if "[Source:" in stripped or "[Chunk:" in stripped:
        return True
    content_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("##")
    ]
    return any(len(line.split()) >= 6 for line in content_lines)


def is_refusal_only_answer(markdown: str, extractor: ClaimExtractor) -> bool:
    stripped = markdown.strip()
    if not stripped:
        return False

    content_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("##")
    ]
    if not content_lines:
        return False

    saw_refusal_language = False
    for line in content_lines:
        normalized = line.lstrip("-• ").strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if extractor._is_insufficient(normalized):
            saw_refusal_language = True
            continue
        if lower.startswith("insufficient evidence in the source pack"):
            saw_refusal_language = True
            continue
        if lower.startswith("insufficient evidence in the provided source pack"):
            saw_refusal_language = True
            continue
        if lower.startswith("generated answer did not produce any verifiable cited claims"):
            saw_refusal_language = True
            continue
        if "best support" in lower and "below" in lower and "threshold" in lower:
            saw_refusal_language = True
            continue
        return False
    return saw_refusal_language


def summarize_claim_diagnostics(claim_diagnostics: List[dict]) -> dict:
    total_claims = len(claim_diagnostics)
    if total_claims == 0:
        return {
            "strong_support_rate": 0.0,
            "weak_support_rate": 0.0,
            "fabricated_citation_rate": 0.0,
            "contradiction_rate": 0.0,
            "numeric_exact_match_rate": 0.0,
            "primary_source_support_rate": 0.0,
            "strict_unsupported_claim_rate": 0.0,
            "atomic_claim_rate": 0.0,
            "claim_extract_success_rate": 0.0,
            "period_match_rate": 0.0,
            "currency_match_rate": 0.0,
            "directionality_match_rate": 0.0,
            "unsupported_numeric_claim_rate": 0.0,
            "primary_source_missing_rate": 0.0,
            "unsafe_publish_rate": 0.0,
            "claim_error_counts": {},
            "claim_error_rates": {},
            "claim_type_counts": {},
            "claim_type_rates": {},
            "most_severe_failure_type": None,
        }

    cited_claims = [
        row for row in claim_diagnostics
        if row.get("cited_sources") and row.get("cited_sources") != ["no_citation"]
    ]
    numeric_claims = [
        row for row in claim_diagnostics
        if row.get("claim_type") in {"number", "comparison"} or row.get("numeric_check_result") != "not_applicable"
    ]
    critical_claims = [
        row for row in claim_diagnostics
        if row.get("claim_type") in {"number", "comparison", "reason"}
    ]

    strong_supported = [row for row in claim_diagnostics if row.get("support_label") == "strong_support"]
    weak_supported = [row for row in claim_diagnostics if row.get("support_label") == "weak_support"]
    fabricated = [row for row in claim_diagnostics if row.get("support_label") == "fabricated_citation"]
    contradictions = [row for row in claim_diagnostics if row.get("support_label") == "contradiction"]
    exact_numeric = [
        row for row in numeric_claims
        if row.get("numeric_check_result") in {"exact", "derived_exact"}
    ]
    primary_supported = [
        row for row in critical_claims
        if row.get("source_tier", 4) <= 2 and row.get("support_label") == "strong_support"
    ]
    atomic_claims = [row for row in claim_diagnostics if row.get("atomicity") is True]
    period_checked_claims = [row for row in claim_diagnostics if row.get("period_verified") is not None]
    currency_checked_claims = [row for row in claim_diagnostics if row.get("currency_verified") is not None]
    direction_checked_claims = [row for row in claim_diagnostics if row.get("directionality_verified") is not None]
    unsupported_numeric_claims = [
        row for row in numeric_claims
        if row.get("support_label") in {"unsupported", "fabricated_citation", "contradiction"}
    ]
    primary_source_missing_claims = [
        row for row in claim_diagnostics
        if "primary_source_missing" in row.get("error_tags", [])
    ]
    strict_failure_tags = {
        "missing_citation",
        "source_not_found",
        "unsupported",
        "fabricated_citation",
        "numeric_mismatch",
        "currency_mismatch",
        "unit_mismatch",
        "period_mismatch",
        "direction_mismatch",
        "contradiction",
        "reason_not_supported",
        "weak_support",
    }
    strict_failures = [
        row for row in claim_diagnostics
        if row.get("support_label") in {"unsupported", "fabricated_citation", "contradiction"}
        or any(tag in strict_failure_tags for tag in row.get("error_tags", []))
    ]
    unsafe_publishes = [
        row for row in claim_diagnostics
        if row.get("support_label") in {"strong_support", "weak_support"}
        and any(tag in strict_failure_tags - {"weak_support"} for tag in row.get("error_tags", []))
    ]

    error_counts = Counter(tag for row in claim_diagnostics for tag in row.get("error_tags", []))
    claim_type_counts = Counter(row.get("claim_type", "fact") for row in claim_diagnostics)
    severity_order = [
        "contradiction",
        "fabricated_citation",
        "numeric_mismatch",
        "direction_mismatch",
        "period_mismatch",
        "unit_mismatch",
        "primary_source_missing",
        "reason_not_supported",
        "weak_support",
        "unsupported",
        "missing_citation",
        "source_not_found",
    ]
    most_severe_failure_type = next((tag for tag in severity_order if error_counts.get(tag)), None)

    return {
        "strong_support_rate": len(strong_supported) / total_claims,
        "weak_support_rate": len(weak_supported) / total_claims,
        "fabricated_citation_rate": len(fabricated) / max(1, len(cited_claims)),
        "contradiction_rate": len(contradictions) / total_claims,
        "numeric_exact_match_rate": len(exact_numeric) / max(1, len(numeric_claims)),
        "primary_source_support_rate": len(primary_supported) / max(1, len(critical_claims)),
        "strict_unsupported_claim_rate": len(strict_failures) / total_claims,
        "atomic_claim_rate": len(atomic_claims) / total_claims,
        "claim_extract_success_rate": 1.0 if total_claims > 0 else 0.0,
        "period_match_rate": (
            sum(1 for row in period_checked_claims if row.get("period_verified") is True) / len(period_checked_claims)
            if period_checked_claims else 0.0
        ),
        "currency_match_rate": (
            sum(1 for row in currency_checked_claims if row.get("currency_verified") is True) / len(currency_checked_claims)
            if currency_checked_claims else 0.0
        ),
        "directionality_match_rate": (
            sum(1 for row in direction_checked_claims if row.get("directionality_verified") is True) / len(direction_checked_claims)
            if direction_checked_claims else 0.0
        ),
        "unsupported_numeric_claim_rate": len(unsupported_numeric_claims) / max(1, len(numeric_claims)),
        "primary_source_missing_rate": len(primary_source_missing_claims) / total_claims,
        "unsafe_publish_rate": len(unsafe_publishes) / total_claims,
        "claim_error_counts": dict(sorted(error_counts.items())),
        "claim_error_rates": {
            tag: count / total_claims
            for tag, count in sorted(error_counts.items())
        },
        "claim_type_counts": dict(sorted(claim_type_counts.items())),
        "claim_type_rates": {
            claim_type: count / total_claims
            for claim_type, count in sorted(claim_type_counts.items())
        },
        "most_severe_failure_type": most_severe_failure_type,
    }


def score_markdown(
    markdown: str,
    required_facts: List[str],
    evidence_store: Dict[str, List[str]],
    *,
    question_text: str = "",
    extractor: ClaimExtractor | None = None,
    verifier: EvidenceVerifier | None = None,
    judge: Optional[LLMJudge] = None,
) -> dict:
    scorable_markdown = extract_scorable_markdown(markdown)
    extractor = extractor or ClaimExtractor()
    verifier = verifier or EvidenceVerifier(use_dummy_model=(settings.llm_mode == "stub"))
    recall = required_fact_recall(scorable_markdown, required_facts)

    claims = extractor.extract_claims(scorable_markdown, "memo")
    if not claims:
        claims = fallback_extract_cited_lines(scorable_markdown, "memo", extractor)
    if not claims:
        refusal_only = is_refusal_only_answer(scorable_markdown, extractor)
        unsupported_rate = 0.0 if refusal_only else (1.0 if has_unscorable_answer_content(scorable_markdown) else 0.0)
        diagnostics = []
        if unsupported_rate > 0.0:
            diagnostics.append(
                {
                    "claim_id": "no_extractable_claims",
                    "claim_text": "No extractable claim could be recovered from the answer text.",
                    "claim_type": "fact",
                    "cited_sources": [],
                    "cited_chunks": [],
                    "support_label": "unsupported",
                    "confidence": "unsupported",
                    "entailment_score": 0.0,
                    "nli_score": 0.0,
                    "numeric_check_result": "not_checked",
                    "numeric_verified": None,
                    "failure_reason": "no_extractable_claims",
                    "supporting_source_ids": [],
                    "supporting_chunk_ids": [],
                    "supporting_evidence": [],
                    "source_tier": 4,
                    "source_tier_label": SOURCE_TIER_LABELS[4],
                    "source_tiers": [],
                    "primary_source_supported": None,
                    "error_tags": ["unsupported"],
                    "explanation": "Answer text contained content, but not in a complete claim form the verifier could score.",
                }
            )
        diagnostic_summary = summarize_claim_diagnostics(diagnostics)
        return {
            "total_claims": 0,
            "citation_marker_coverage": 0.0,
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": unsupported_rate,
            "avg_nli_score": 0.0,
            "required_fact_recall": recall,
            "scoreable_markdown": scorable_markdown,
            "claim_diagnostics": diagnostics,
            **diagnostic_summary,
        }

    verification_results = verifier.verify_memo(claims, evidence_store)
    summary = verifier.summary_stats(verification_results)
    claim_diagnostics = build_claim_diagnostics(claims, verification_results, evidence_store)
    claim_diagnostics = apply_llm_claim_adjudication(
        claim_diagnostics,
        evidence_store,
        question_text=question_text,
        judge=judge,
    )
    diagnostic_summary = summarize_claim_diagnostics(claim_diagnostics)
    unsupported_rate = diagnostic_summary["strict_unsupported_claim_rate"]

    return {
        "total_claims": summary["total_claims"],
        "citation_marker_coverage": summary.get("citation_marker_coverage", 0.0),
        "verified_claim_coverage": summary.get("verified_claim_coverage", 0.0),
        "unsupported_claim_rate": unsupported_rate,
        "avg_nli_score": summary["avg_nli_score"],
        "required_fact_recall": recall,
        "scoreable_markdown": scorable_markdown,
        "claim_diagnostics": claim_diagnostics,
        **diagnostic_summary,
    }


def run_plain_llm(query: str, company: str, model_client) -> str:
    full_context, _ = load_pack_context(company)
    prompt = f"""You are writing a first-pass investment memo using only the evidence below.

User query:
{query}

Requirements:
- Use only the supplied evidence.
- Every nontrivial factual claim must cite its source as [Source: source_id].
- If evidence is missing, explicitly say "insufficient evidence".
- Write these sections:
  1. Executive Summary
  2. Business Model
  3. Revenue Quality
  4. Valuation Context
  5. Competitive Position
  6. Risks
  7. Key Monitorables

Evidence pack:
{full_context[:16000]}
"""

    return generate_text_response(
        model_client,
        model=settings.openai_model,
        system_prompt=MEMO_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=1400,
        temperature=0.2,
    )


def evaluate(benchmark_file: Path, output_file: Path):
    with open(benchmark_file, encoding="utf-8") as f:
        benchmarks = json.load(f)

    logger.info(f"Loaded {len(benchmarks)} benchmark cases.")
    agent = BizIntelAgent()
    model_client = build_openai_client(settings.openai_api_key, settings.openai_api_base)
    extractor = ClaimExtractor()
    verifier = EvidenceVerifier()
    judge = LLMJudge()

    case_results = []
    system_totals = {
        "bizintel": {
            "citation_marker_coverage": 0.0,
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
            "avg_nli_score": 0.0,
            "required_fact_recall": 0.0,
        },
        "plain_llm": {
            "citation_marker_coverage": 0.0,
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
            "avg_nli_score": 0.0,
            "required_fact_recall": 0.0,
        },
    }

    for idx, case in enumerate(benchmarks, start=1):
        query = case["query"]
        company = case["company"]
        required_facts = case.get("required_facts", [])

        logger.info(f"\n[{idx}/{len(benchmarks)}] Evaluating: '{query}'")
        _, evidence_store = load_pack_context(company)

        bizintel_result = agent.research(query=query)
        bizintel_markdown = bizintel_result["memo_markdown"]
        plain_llm_markdown = run_plain_llm(query, company, model_client)

        bizintel_metrics = score_markdown(
            bizintel_markdown,
            required_facts,
            evidence_store,
            question_text=query,
            extractor=extractor,
            verifier=verifier,
            judge=judge,
        )
        bizintel_research_metrics = score_research_trace(
            bizintel_result,
            required_facts,
            expected_companies=[company],
        )
        plain_llm_metrics = score_markdown(
            plain_llm_markdown,
            required_facts,
            evidence_store,
            question_text=query,
            extractor=extractor,
            verifier=verifier,
            judge=judge,
        )

        for system_name, metrics in (("bizintel", bizintel_metrics), ("plain_llm", plain_llm_metrics)):
            for key in system_totals[system_name]:
                system_totals[system_name][key] += metrics[key]

        logger.info(
            "  BizIntel: verified=%0.2f unsupported=%0.2f recall=%0.2f",
            bizintel_metrics["verified_claim_coverage"],
            bizintel_metrics["unsupported_claim_rate"],
            bizintel_metrics["required_fact_recall"],
        )
        logger.info(
            "  Plain LLM: verified=%0.2f unsupported=%0.2f recall=%0.2f",
            plain_llm_metrics["verified_claim_coverage"],
            plain_llm_metrics["unsupported_claim_rate"],
            plain_llm_metrics["required_fact_recall"],
        )

        case_results.append(
            {
                "query": query,
                "company": company,
                "bizintel": bizintel_metrics,
                "plain_llm": plain_llm_metrics,
                "bizintel_trace": {
                    "final_answer": bizintel_markdown,
                    "scoreable_answer": bizintel_metrics["scoreable_markdown"],
                    "pipeline_trace": build_trace_payload(bizintel_result),
                    "claim_diagnostics": bizintel_metrics["claim_diagnostics"],
                    "research_metrics": bizintel_research_metrics,
                },
                "plain_llm_trace": {
                    "final_answer": plain_llm_markdown,
                    "scoreable_answer": plain_llm_metrics["scoreable_markdown"],
                    "claim_diagnostics": plain_llm_metrics["claim_diagnostics"],
                },
                "deltas": {
                    "citation_marker_coverage": bizintel_metrics["citation_marker_coverage"] - plain_llm_metrics["citation_marker_coverage"],
                    "verified_claim_coverage": bizintel_metrics["verified_claim_coverage"] - plain_llm_metrics["verified_claim_coverage"],
                    "unsupported_claim_rate": bizintel_metrics["unsupported_claim_rate"] - plain_llm_metrics["unsupported_claim_rate"],
                    "avg_nli_score": bizintel_metrics["avg_nli_score"] - plain_llm_metrics["avg_nli_score"],
                    "required_fact_recall": bizintel_metrics["required_fact_recall"] - plain_llm_metrics["required_fact_recall"],
                },
            }
        )

    averages = {}
    benchmark_count = len(benchmarks) or 1
    for system_name, totals in system_totals.items():
        averages[system_name] = {
            metric: value / benchmark_count for metric, value in totals.items()
        }

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": settings.openai_model,
        "benchmark_cases": len(benchmarks),
        "averages": averages,
        "cases": case_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("\n--- Evaluation Complete ---")
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    if not settings.openai_api_key:
        print("未检测到 Minimax API Key，baseline 对比评测需要真实的兼容 API：")
        api_key = getpass.getpass("MINIMAX_API_KEY (隐式输入): ")
        if not api_key.strip():
            logger.error("❌必须提供 API Key 才能完成评测")
            sys.exit(1)
        settings.openai_api_key = api_key.strip()

    benchmark_path = settings.eval_cases_dir / "benchmark.json"
    output_path = settings.data_dir.parent / "eval" / "results" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found at {benchmark_path}")
        sys.exit(1)

    evaluate(benchmark_path, output_path)
