"""
Prepare question-scoped local fact cards for the benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from agent.config import settings
from tools.normalize_benchmark_corpus import normalize_whitespace
from tools.resolve_benchmark_evidence import resolve_entry


MIN_FACT_CHARS = 24
SOFT_MAX_FACT_CHARS = 320


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_documents() -> dict[str, dict]:
    documents: dict[str, dict] = {}
    normalized_root = settings.data_dir / "normalized"
    for company_dir in sorted(normalized_root.iterdir()):
        documents_path = company_dir / "documents.jsonl"
        if not documents_path.exists():
            continue
        for row in load_jsonl(documents_path):
            documents[row["doc_id"]] = row
    return documents


def load_chunks_by_doc() -> dict[str, list[dict]]:
    chunks_by_doc: dict[str, list[dict]] = {}
    normalized_root = settings.data_dir / "normalized"
    for company_dir in sorted(normalized_root.iterdir()):
        chunks_path = company_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        for row in load_jsonl(chunks_path):
            chunks_by_doc.setdefault(row["doc_id"], []).append(row)
    for doc_id in chunks_by_doc:
        chunks_by_doc[doc_id].sort(key=lambda row: row.get("chunk_index", 0))
    return chunks_by_doc


def normalize_for_match(text: str) -> str:
    text = text.lower().replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fact_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$%][a-z0-9$%.\-]{1,}", normalize_for_match(text)))


def split_fact_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"\[Page\s+\d+\]", " ", text)
    cleaned = re.sub(r"\s*[•✓✖]\s*", "\n", cleaned)
    cleaned = normalize_whitespace(cleaned)
    if not cleaned:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    blocks = re.split(r"\n+", cleaned)
    for block in blocks:
        for part in re.split(r"(?<=[.!?;])\s+", block):
            candidate = normalize_whitespace(part)
            if len(candidate) < MIN_FACT_CHARS:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    if candidates:
        return candidates
    return [cleaned]


def candidate_score(candidate: str, anchor_text: str, context_text: str = "") -> tuple[int, float, int, int, int]:
    candidate_norm = normalize_for_match(candidate)
    anchor_norm = normalize_for_match(anchor_text)
    anchor_token_set = fact_tokens(anchor_text)
    exact_match = int(bool(anchor_norm and anchor_norm in candidate_norm))
    candidate_token_set = fact_tokens(candidate)
    overlap = len(anchor_token_set & candidate_token_set)
    coverage = overlap / max(1, len(anchor_token_set))
    if len(anchor_token_set) <= 2 and not exact_match:
        overlap = 0
        coverage = 0.0
    context_overlap = len(fact_tokens(context_text) & candidate_token_set)
    length_penalty = max(0, len(candidate) - SOFT_MAX_FACT_CHARS)
    return (exact_match, coverage, overlap, context_overlap, -length_penalty)


def is_low_quality_candidate(candidate: str) -> bool:
    stripped = candidate.strip()
    if not stripped:
        return True
    if re.search(r"[✓✖]", stripped):
        return True
    if stripped[:1].islower():
        return True
    if len(stripped.split()) < 4:
        return True
    if not re.search(r"[.!?:;%$]", stripped) and stripped.istitle():
        return True
    return False


def select_fact_text(anchor_text: str, chunk_text: str, context_text: str = "") -> str:
    anchor = normalize_whitespace(anchor_text)
    if not chunk_text:
        return anchor

    candidates = split_fact_candidates(chunk_text)
    if not candidates:
        return anchor

    best_candidate = max(
        candidates,
        key=lambda candidate: (candidate_score(candidate, anchor, context_text), -len(candidate)),
    )
    if is_low_quality_candidate(best_candidate):
        if len(fact_tokens(anchor)) <= 2 and anchor:
            return anchor
        cleaner_candidates = [candidate for candidate in candidates if not is_low_quality_candidate(candidate)]
        if cleaner_candidates:
            fallback_candidate = max(
                cleaner_candidates,
                key=lambda candidate: (candidate_score(candidate, anchor, context_text), -len(candidate)),
            )
            fallback_score = candidate_score(fallback_candidate, anchor, context_text)
            if fallback_score[1] > 0 or fallback_score[3] > 0:
                best_candidate = fallback_candidate
    if len(best_candidate) > SOFT_MAX_FACT_CHARS * 2 and len(anchor) >= MIN_FACT_CHARS:
        return anchor
    if candidate_score(best_candidate, anchor, context_text)[0] == 0 and len(anchor) >= MIN_FACT_CHARS:
        return anchor
    return best_candidate


def resolve_chunk(entry: dict, chunks_by_doc: dict[str, list[dict]]) -> tuple[dict, dict]:
    doc_chunks = chunks_by_doc.get(entry["doc_id"], [])
    if not doc_chunks:
        raise ValueError(f"No chunks found for doc_id={entry['doc_id']}")

    chunk_id = entry.get("chunk_id")
    if chunk_id:
        for chunk in doc_chunks:
            if chunk["chunk_id"] == chunk_id:
                resolved_entry = dict(entry)
                resolved_entry["chunk_index"] = chunk.get("chunk_index")
                return resolved_entry, chunk

    resolved_entry = resolve_entry(entry, chunks_by_doc)
    for chunk in doc_chunks:
        if chunk["chunk_id"] == resolved_entry["chunk_id"]:
            return resolved_entry, chunk
    raise ValueError(f"Resolved chunk missing for doc_id={entry['doc_id']} chunk_id={resolved_entry['chunk_id']}")


def build_local_facts(version: str) -> list[dict]:
    benchmark_dir = settings.data_dir / "benchmark" / version
    items = {row["item_id"]: row for row in load_jsonl(benchmark_dir / "items.jsonl")}
    answers = {row["item_id"]: row for row in load_jsonl(benchmark_dir / "answers.jsonl")}
    evidence_rows = load_jsonl(benchmark_dir / "evidence.jsonl")
    documents = load_documents()
    chunks_by_doc = load_chunks_by_doc()

    local_facts: list[dict] = []
    for evidence_row in evidence_rows:
        item_id = evidence_row["item_id"]
        item = items[item_id]
        answer = answers.get(item_id, {})

        for fact_index, entry in enumerate(evidence_row["evidence"], start=1):
            resolved_entry, chunk = resolve_chunk(entry, chunks_by_doc)
            document = documents.get(entry["doc_id"])
            if document is None:
                raise ValueError(f"Missing normalized document metadata for doc_id={entry['doc_id']}")

            context_text = " ".join(
                [
                    entry.get("section_hint", ""),
                    " ".join(item.get("required_evidence", {}).get("must_cover", [])),
                    " ".join(answer.get("must_cover", [])),
                ]
            )
            fact_text = select_fact_text(entry.get("anchor_text", ""), chunk.get("text", ""), context_text)
            local_facts.append(
                {
                    "fact_id": f"{version}__{item_id}__{fact_index:02d}",
                    "version": version,
                    "item_id": item_id,
                    "query": item["query"],
                    "fact": fact_text,
                    "attributes": {
                        "company": entry["company"],
                        "category": item["category"],
                        "difficulty": item["difficulty"],
                        "query_type": item.get("query_type", []),
                        "required_source_types": item.get("required_source_types", []),
                        "required_slots": item.get("required_evidence", {}).get("must_cover", []),
                        "gold_must_cover": answer.get("must_cover", []),
                        "doc_id": entry["doc_id"],
                        "source_type": entry["source_type"],
                        "period": entry["period"],
                        "title": document.get("title"),
                        "author": document.get("issuer"),
                        "issuer": document.get("issuer"),
                        "url": document.get("url"),
                        "published_at": document.get("published_at"),
                        "section_hint": entry.get("section_hint"),
                        "evidence_role": entry.get("evidence_role"),
                        "evidence_strength": entry.get("evidence_strength"),
                        "stance": entry.get("stance"),
                        "anchor_text": entry.get("anchor_text"),
                        "chunk_id": resolved_entry.get("chunk_id"),
                        "chunk_index": resolved_entry.get("chunk_index"),
                        "binding_method": resolved_entry.get("binding_method"),
                    },
                }
            )
    return local_facts


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate atomic local fact cards for benchmark items.")
    parser.add_argument("--version", default="v1", help="Benchmark version to prepare.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/benchmark/<version>/local_facts.jsonl.",
    )
    args = parser.parse_args()

    local_facts = build_local_facts(args.version)
    output_path = args.output or (settings.data_dir / "benchmark" / args.version / "local_facts.jsonl")
    write_jsonl(output_path, local_facts)
    print(
        json.dumps(
            {
                "version": args.version,
                "fact_count": len(local_facts),
                "item_count": len({row["item_id"] for row in local_facts}),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
