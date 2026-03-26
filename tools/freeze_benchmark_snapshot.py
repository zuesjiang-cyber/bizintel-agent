"""
Freeze a hash-level snapshot of the local benchmark corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent.config import settings
from tools.fetch_benchmark_sources import file_sha256, load_manifest


def _json_sha256(path: Path) -> str:
    return file_sha256(path)


def build_company_snapshot(company: str) -> dict:
    raw_dir = settings.data_dir / "raw" / company
    manifest_path = raw_dir / "manifest.json"
    periods_path = raw_dir / "periods.json"
    receipts_path = raw_dir / "download_receipts.jsonl"

    manifest = load_manifest(company)
    documents = []
    missing_documents = []
    for document in manifest["documents"]:
        relative_path = Path(document["path"])
        absolute_path = settings.data_dir.parent / relative_path
        if not absolute_path.exists():
            missing_documents.append(
                {
                    "doc_id": document["doc_id"],
                    "path": str(relative_path),
                    "is_primary": document.get("is_primary", True),
                }
            )
            continue
        documents.append(
            {
                "doc_id": document["doc_id"],
                "path": str(relative_path),
                "sha256": file_sha256(absolute_path),
                "bytes": absolute_path.stat().st_size,
            }
        )

    documents.sort(key=lambda row: row["doc_id"])
    missing_documents.sort(key=lambda row: row["doc_id"])
    return {
        "company": company,
        "manifest_sha256": _json_sha256(manifest_path),
        "periods_sha256": _json_sha256(periods_path),
        "receipts_sha256": _json_sha256(receipts_path),
        "document_count": len(documents),
        "missing_document_count": len(missing_documents),
        "documents": documents,
        "missing_documents": missing_documents,
    }


def build_snapshot(version: str, companies: list[str]) -> dict:
    companies = sorted(companies)
    payload = {
        "version": version,
        "companies": [build_company_snapshot(company) for company in companies],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload["snapshot_id"] = digest
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a benchmark corpus snapshot.")
    parser.add_argument("--version", default="v1", help="Benchmark version to freeze.")
    parser.add_argument("--company", action="append", required=True, help="Company to include.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional snapshot output path. Defaults to data/benchmark/<version>/corpus_snapshot.json.",
    )
    args = parser.parse_args()

    snapshot = build_snapshot(args.version, args.company)
    output_path = args.output or (settings.data_dir / "benchmark" / args.version / "corpus_snapshot.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
