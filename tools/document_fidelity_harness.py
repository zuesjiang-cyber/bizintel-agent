from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PRIMARY_SOURCE_TYPES = {
    "10-k",
    "10k",
    "10-q",
    "10q",
    "20-f",
    "8-k",
    "annual_report",
    "earnings_release",
    "shareholder_letter",
    "transcript",
}

UNIT_RE = re.compile(r"\b(million|billion|thousand|usd|dollars?|bps|percent|percentage|%|shares?|eps)\b", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_primary_source(source: dict[str, Any]) -> bool:
    explicit = source.get("primary_source")
    if explicit is not None:
        return bool(explicit)
    source_type = str(source.get("source_type") or "").lower().replace(" ", "_")
    return source_type in PRIMARY_SOURCE_TYPES


def is_table_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any("|" in line or "\t" in line for line in lines):
        return True
    numeric_lines = sum(1 for line in lines if len(NUMBER_RE.findall(line)) >= 3)
    return numeric_lines >= 2


def boundary_warning(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty_chunk"
    if len(stripped) < 40:
        return "very_short_chunk"
    if len(stripped) > 3000:
        return "very_long_chunk"
    if stripped[0].islower():
        return "starts_mid_sentence"
    if stripped[-1] not in ".!?)]}\"'":
        return "ends_without_sentence_boundary"
    return ""


def chunk_row(company_id: str, chunk: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    source = source or {}
    text = str(chunk.get("text") or "")
    page = chunk.get("page")
    period = chunk.get("period") or source.get("period") or source.get("date")
    source_type = chunk.get("source_type") or source.get("source_type")
    metadata_issues = []
    if not chunk.get("chunk_id"):
        metadata_issues.append("missing_chunk_id")
    if not chunk.get("source_id"):
        metadata_issues.append("missing_source_id")
    if page is None:
        metadata_issues.append("missing_page")
    if not period:
        metadata_issues.append("missing_period")
    if not source_type:
        metadata_issues.append("missing_source_type")
    if "primary_source" not in chunk and "primary_source" not in source:
        metadata_issues.append("missing_primary_source_flag")

    warning = boundary_warning(text)
    if warning:
        metadata_issues.append(warning)

    return {
        "company_id": company_id,
        "source_id": chunk.get("source_id") or "",
        "chunk_id": chunk.get("chunk_id") or "",
        "page": page,
        "period": period,
        "source_type": source_type,
        "primary_source": is_primary_source(source),
        "text_length": len(text),
        "has_numeric_content": bool(NUMBER_RE.search(text)),
        "has_unit": bool(UNIT_RE.search(text)),
        "has_period_mention": bool(YEAR_RE.search(text)),
        "table_like": is_table_like(text),
        "boundary_warning": warning,
        "metadata_issues": metadata_issues,
    }


def analyze_company(processed_root: Path, company_id: str) -> dict[str, Any]:
    company_dir = processed_root / company_id
    chunks_path = company_dir / "chunks.json"
    sources_path = company_dir / "sources.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    chunks = load_json(chunks_path)
    sources = load_json(sources_path) if sources_path.exists() else []
    source_map = {source.get("source_id"): source for source in sources if isinstance(source, dict)}
    rows = [
        chunk_row(company_id, chunk, source_map.get(chunk.get("source_id")))
        for chunk in chunks
        if isinstance(chunk, dict)
    ]
    issue_counts = Counter(issue for row in rows for issue in row["metadata_issues"])
    summary = {
        "company_id": company_id,
        "chunk_count": len(rows),
        "source_count": len(source_map),
        "issue_counts": dict(issue_counts),
        "chunks_with_numeric_content": sum(1 for row in rows if row["has_numeric_content"]),
        "chunks_with_units": sum(1 for row in rows if row["has_unit"]),
        "chunks_with_period_mentions": sum(1 for row in rows if row["has_period_mention"]),
        "table_like_chunks": sum(1 for row in rows if row["table_like"]),
        "primary_source_chunks": sum(1 for row in rows if row["primary_source"]),
    }
    return {"summary": summary, "rows": rows}


def analyze_processed_root(processed_root: Path, companies: list[str] | None = None) -> dict[str, Any]:
    if companies:
        company_ids = companies
    else:
        company_ids = sorted(
            path.name
            for path in processed_root.iterdir()
            if path.is_dir() and (path / "chunks.json").exists()
        )
    company_reports = [analyze_company(processed_root, company_id) for company_id in company_ids]
    total_issue_counts = Counter()
    for report in company_reports:
        total_issue_counts.update(report["summary"]["issue_counts"])
    return {
        "processed_root": str(processed_root),
        "company_count": len(company_reports),
        "summaries": [report["summary"] for report in company_reports],
        "issue_counts": dict(total_issue_counts),
        "rows": [row for report in company_reports for row in report["rows"]],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Document Fidelity Report",
        "",
        f"- Processed root: `{report['processed_root']}`",
        f"- Companies: {report['company_count']}",
        f"- Rows: {len(report['rows'])}",
        "",
        "## Issue Counts",
        "",
    ]
    if report["issue_counts"]:
        for issue, count in sorted(report["issue_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {issue}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Company Summary", "", "| Company | Chunks | Sources | Numeric | Units | Period Mentions | Table-like | Primary Source |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for summary in report["summaries"]:
        lines.append(
            f"| {summary['company_id']} | {summary['chunk_count']} | {summary['source_count']} | "
            f"{summary['chunks_with_numeric_content']} | {summary['chunks_with_units']} | "
            f"{summary['chunks_with_period_mentions']} | {summary['table_like_chunks']} | "
            f"{summary['primary_source_chunks']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight document fidelity harness over processed source packs.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--company", action="append", default=None, help="Company folder to analyze. Can be repeated.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    report = analyze_processed_root(args.processed_root, args.company)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        args.output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(report["rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
