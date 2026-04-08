import csv
import io
import json
from pathlib import Path
from typing import Dict, List

from agent.schemas import GeneratedMemo


def build_trace_payload(result: dict) -> dict:
    if "research_trace" in result:
        memo = result.get("memo_object")
        subquestion_results = result.get("subquestion_results", [])
        research_task = result.get("research_task")
        replay = result["research_trace"]
        return {
            "mode": getattr(research_task.mode, "name", "COMPANY") if research_task else "COMPANY",
            "query": research_task.query if research_task else "",
            "steps": [item.subquestion.question_id for item in subquestion_results],
            "contract": memo.contract if memo is not None else {},
            "workflow_events": result.get("workflow_events", []),
            "final_answer": result.get("memo_markdown", ""),
            "step_traces": [
                {
                    "step": item.subquestion.question_id,
                    "search_queries": item.trace[0]["queries"] if item.trace else [],
                    "query_contracts": [
                        {
                            "fact_slot": item.subquestion.fact_slot,
                            "lane": item.subquestion.lane.value,
                            "period": item.subquestion.required_period,
                            "source_types": item.subquestion.allowed_source_types,
                        }
                    ],
                    "generated_content": item.supported_content or item.answer_text,
                    "sources_used": [
                        {"source_id": chunk.source_id, "chunk_id": chunk.chunk_id, "score": chunk.score}
                        for chunk in item.evidence
                    ],
                    "evidence_ledger": [
                        {
                            "fact_slot": item.subquestion.fact_slot,
                            "support_status": item.status.value,
                            "evidence_ids": [chunk.chunk_id for chunk in item.evidence],
                            "uncertainty_note": item.refusal_reason or "",
                        }
                    ],
                    "evidence_notes": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "evidence_text": chunk.text,
                            "evidence_type": item.subquestion.lane.value,
                            "claim": item.subquestion.text,
                        }
                        for chunk in item.evidence[:8]
                    ],
                    "retrieval_trace": item.trace,
                    "generation_context": item.subquestion.text,
                    "previous_findings": "",
                    "covered_facts": [item.subquestion.fact_slot] if item.status.value == "completed" else [],
                    "missing_facts": [] if item.status.value == "completed" else [item.subquestion.fact_slot],
                    "gap_reflection": {"followup_queries": item.followup_queries, "refusal_reason": item.refusal_reason},
                    "gap_iterations": item.trace,
                    "writing_trace": {
                        "answer_text": item.answer_text,
                        "supported_content": item.supported_content,
                        "status": item.status.value,
                    },
                }
                for item in subquestion_results
            ],
            "verification": build_verification_rows(memo) if memo is not None else [],
            "research_tree": [
                {
                    "question_id": item.question_id,
                    "text": item.text,
                    "lane": item.lane.value,
                    "status": item.status.value,
                    "rounds_used": item.rounds_used,
                }
                for item in replay.subquestions
            ],
            "decisions": [
                {
                    "decision_type": decision.decision_type,
                    "question_id": decision.question_id,
                    "reason": decision.reason,
                    "payload": decision.payload,
                    "created_at": decision.created_at,
                }
                for decision in replay.decisions
            ],
            "llm_calls_used": replay.llm_calls_used,
        }

    step_outputs = result.get("step_outputs", {})
    memo = result.get("memo_object")
    return {
        "mode": result["plan"].mode.name,
        "query": result["plan"].user_query,
        "steps": [step.name for step in result["plan"].steps],
        "contract": result["plan"].contract,
        "workflow_events": result["workflow_events"],
        "final_answer": result.get("memo_markdown", ""),
        "step_traces": [
            {
                "step": step.name,
                "search_queries": step.search_queries,
                "query_contracts": getattr(step, "query_contracts", []),
                "generated_content": step_outputs.get(step.name, {}).get("content", ""),
                "sources_used": step_outputs.get(step.name, {}).get("sources_used", []),
                "evidence_ledger": step_outputs.get(step.name, {}).get("evidence_ledger", []),
                "evidence_notes": step_outputs.get(step.name, {}).get("evidence_notes", []),
                "retrieval_trace": step_outputs.get(step.name, {}).get("retrieval_trace", []),
                "generation_context": step_outputs.get(step.name, {}).get("generation_context", ""),
                "previous_findings": step_outputs.get(step.name, {}).get("previous_findings", ""),
                "covered_facts": step_outputs.get(step.name, {}).get("covered_facts", []),
                "missing_facts": step_outputs.get(step.name, {}).get("missing_facts", []),
                "gap_reflection": step_outputs.get(step.name, {}).get("gap_reflection", {}),
                "gap_iterations": step_outputs.get(step.name, {}).get("gap_iterations", []),
                "writing_trace": next(
                    (
                        section.writing_trace
                        for section in getattr(memo, "sections", [])
                        if section.title == step.name
                    ),
                    {},
                ) if memo is not None else {},
            }
            for step in result["plan"].steps
        ],
        "verification": build_verification_rows(memo) if memo is not None else [],
    }


def build_verification_rows(memo: GeneratedMemo) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for section in memo.sections:
        for result in section.verification_results:
            rows.append(
                {
                    "section": section.title,
                    "claim_text": result.claim.text,
                    "confidence": result.confidence.value,
                    "nli_score": round(result.nli_score, 4),
                    "numeric_verified": result.numeric_verified,
                    "failure_reason": result.failure_reason,
                    "cited_sources": ", ".join(result.claim.cited_sources),
                    "cited_chunks": ", ".join(result.claim.cited_chunks),
                    "supporting_source_ids": ", ".join(result.supporting_source_ids),
                    "supporting_chunk_ids": ", ".join(result.supporting_chunk_ids),
                    "primary_source_supported": result.primary_source_supported,
                    "supporting_evidence": " | ".join(result.supporting_evidence),
                    "explanation": result.explanation,
                }
            )
        if section.verification_results or not section.content.strip():
            continue
        rows.append(
            {
                "section": section.title,
                "claim_text": section.content.strip().splitlines()[0][:500],
                "confidence": "not_verified",
                "nli_score": 0.0,
                "numeric_verified": None,
                "failure_reason": "not_verified",
                "cited_sources": "",
                "cited_chunks": "",
                "supporting_source_ids": "",
                "supporting_chunk_ids": "",
                "primary_source_supported": None,
                "supporting_evidence": "",
                "explanation": "Section content was produced, but no claim-level verification rows were available.",
            }
        )
    return rows


def verification_rows_to_csv(rows: List[Dict[str, object]]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "section",
        "claim_text",
        "confidence",
        "nli_score",
        "numeric_verified",
        "failure_reason",
        "cited_sources",
        "cited_chunks",
        "supporting_source_ids",
        "supporting_chunk_ids",
        "primary_source_supported",
        "supporting_evidence",
        "explanation",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_summary_payload(result: dict) -> dict:
    memo = result["memo_object"]
    rows = build_verification_rows(memo)
    return {
        "title": memo.title,
        "query": memo.query,
        "mode": memo.mode.value,
        "generated_at": memo.generated_at,
        "overall_confidence": memo.overall_confidence,
        "section_count": len(memo.sections),
        "source_count": len(memo.sources),
        "verified_claim_count": len(rows),
        "contract": memo.contract,
    }


def write_artifact_bundle(result: dict, artifact_dir: Path) -> Dict[str, Path]:
    memo = result["memo_object"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    memo_path = artifact_dir / "memo.md"
    trace_path = artifact_dir / "trace.json"
    summary_path = artifact_dir / "summary.json"
    verification_path = artifact_dir / "verification.csv"

    memo_path.write_text(result["memo_markdown"], encoding="utf-8")
    trace_path.write_text(
        json.dumps(build_trace_payload(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(build_summary_payload(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    verification_path.write_text(
        verification_rows_to_csv(build_verification_rows(memo)),
        encoding="utf-8",
    )

    return {
        "memo": memo_path,
        "trace": trace_path,
        "summary": summary_path,
        "verification": verification_path,
    }
