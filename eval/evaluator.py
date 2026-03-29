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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent.artifacts import build_trace_payload
from agent.config import settings
from agent.llm_utils import build_openai_client, generate_text_response
from agent.orchestrator import BizIntelAgent
from agent.prompts.synthesis import MEMO_SYSTEM_PROMPT
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


def build_claim_diagnostics(claims, verification_results) -> List[dict]:
    rows = []
    for claim, result in zip(claims, verification_results):
        rows.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.text,
                "cited_sources": claim.cited_sources,
                "confidence": result.confidence.value,
                "nli_score": result.nli_score,
                "numeric_verified": result.numeric_verified,
                "failure_reason": result.failure_reason,
                "supporting_source_ids": result.supporting_source_ids,
                "supporting_chunk_ids": result.supporting_chunk_ids,
                "supporting_evidence": result.supporting_evidence,
                "explanation": result.explanation,
            }
        )
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


def score_markdown(
    markdown: str,
    required_facts: List[str],
    evidence_store: Dict[str, List[str]],
    *,
    extractor: ClaimExtractor | None = None,
    verifier: EvidenceVerifier | None = None,
) -> dict:
    scorable_markdown = extract_scorable_markdown(markdown)
    extractor = extractor or ClaimExtractor()
    verifier = verifier or EvidenceVerifier(use_dummy_model=(settings.llm_mode == "stub"))
    recall = required_fact_recall(scorable_markdown, required_facts)

    claims = extractor.extract_claims(scorable_markdown, "memo")
    if not claims:
        claims = fallback_extract_cited_lines(scorable_markdown, "memo", extractor)
    if not claims:
        unsupported_rate = 1.0 if has_unscorable_answer_content(scorable_markdown) else 0.0
        diagnostics = []
        if unsupported_rate > 0.0:
            diagnostics.append(
                {
                    "claim_id": "no_extractable_claims",
                    "claim_text": "No extractable claim could be recovered from the answer text.",
                    "cited_sources": [],
                    "confidence": "unsupported",
                    "nli_score": 0.0,
                    "numeric_verified": None,
                    "failure_reason": "no_extractable_claims",
                    "supporting_source_ids": [],
                    "supporting_chunk_ids": [],
                    "supporting_evidence": [],
                    "explanation": "Answer text contained content, but not in a complete claim form the verifier could score.",
                }
            )
        return {
            "total_claims": 0,
            "citation_marker_coverage": 0.0,
            "verified_claim_coverage": 0.0,
            "unsupported_claim_rate": unsupported_rate,
            "avg_nli_score": 0.0,
            "required_fact_recall": recall,
            "scoreable_markdown": scorable_markdown,
            "claim_diagnostics": diagnostics,
        }

    verification_results = verifier.verify_memo(claims, evidence_store)
    summary = verifier.summary_stats(verification_results)
    unsupported_rate = (
        summary["unsupported"] / summary["total_claims"] if summary["total_claims"] else 0.0
    )

    return {
        "total_claims": summary["total_claims"],
        "citation_marker_coverage": summary.get("citation_marker_coverage", 0.0),
        "verified_claim_coverage": summary.get("verified_claim_coverage", 0.0),
        "unsupported_claim_rate": unsupported_rate,
        "avg_nli_score": summary["avg_nli_score"],
        "required_fact_recall": recall,
        "scoreable_markdown": scorable_markdown,
        "claim_diagnostics": build_claim_diagnostics(claims, verification_results),
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
            extractor=extractor,
            verifier=verifier,
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
            extractor=extractor,
            verifier=verifier,
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
