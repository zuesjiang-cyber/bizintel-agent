import csv
import io
import json
from pathlib import Path
from typing import Dict, List

from agent.schemas import GeneratedMemo


def build_trace_payload(result: dict) -> dict:
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
