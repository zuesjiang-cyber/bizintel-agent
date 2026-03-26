"""
Resolve benchmark evidence anchors to concrete chunk ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent.config import settings


def normalize(text: str) -> str:
    text = text.lower().replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks() -> dict[str, list[dict]]:
    chunks_by_doc: dict[str, list[dict]] = {}
    normalized_root = settings.data_dir / "normalized"
    for company_dir in normalized_root.iterdir():
        chunks_path = company_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        with open(chunks_path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                chunks_by_doc.setdefault(row["doc_id"], []).append(row)
    return chunks_by_doc


def resolve_entry(entry: dict, chunks_by_doc: dict[str, list[dict]]) -> dict:
    doc_chunks = chunks_by_doc.get(entry["doc_id"], [])
    if not doc_chunks:
        raise ValueError(f"No chunks found for doc_id={entry['doc_id']}")

    anchor = entry.get("anchor_text")
    if not anchor:
        raise ValueError(f"Missing anchor_text for doc_id={entry['doc_id']}")
    anchor_norm = normalize(anchor)

    exact_matches = [chunk for chunk in doc_chunks if anchor_norm in normalize(chunk["text"])]
    if exact_matches:
        chunk = exact_matches[0]
        binding_method = "anchor_text_substring"
    else:
        anchor_tokens = set(re.findall(r"[a-z0-9]{3,}", anchor_norm))
        scored = []
        for chunk in doc_chunks:
            chunk_tokens = set(re.findall(r"[a-z0-9]{3,}", normalize(chunk["text"])))
            score = len(anchor_tokens & chunk_tokens)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] == 0:
            raise ValueError(f"Could not resolve anchor for doc_id={entry['doc_id']}")
        chunk = scored[0][1]
        binding_method = "anchor_token_overlap"

    resolved = dict(entry)
    resolved["chunk_id"] = chunk["chunk_id"]
    resolved["chunk_index"] = chunk["chunk_index"]
    resolved["binding_method"] = binding_method
    return resolved


def resolve_file(path: Path) -> list[dict]:
    chunks_by_doc = load_chunks()
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["evidence"] = [resolve_entry(entry, chunks_by_doc) for entry in row["evidence"]]
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve benchmark evidence anchors to chunk ids.")
    parser.add_argument(
        "--file",
        type=Path,
        default=settings.data_dir / "benchmark" / "v1" / "evidence.jsonl",
        help="Evidence JSONL file to rewrite in place.",
    )
    args = parser.parse_args()

    rows = resolve_file(args.file)
    with open(args.file, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"resolved {len(rows)} evidence rows -> {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
