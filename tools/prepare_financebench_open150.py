"""
Prepare the FinanceBench open-source 150-question subset for BizIntel's benchmark harness.

This tool converts Patronus AI's public FinanceBench JSONL files into:

- benchmark-compatible `items.jsonl`
- benchmark-compatible `answers.jsonl`
- benchmark-compatible `evidence.jsonl`
- a document catalog for the subset
- per-company raw manifests and periods files that can be consumed by the
  existing raw-source fetch + normalization pipeline
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from agent.config import settings


BENCHMARK_VERSION = "financebench_open150"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("evidence_text", "text", "value"):
            if isinstance(value.get(key), str):
                return value[key]
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = [coerce_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def normalize_whitespace(text) -> str:
    return re.sub(r"\s+", " ", coerce_text(text)).strip()


def slugify_company(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"financebench_{slug}"


def normalize_doc_id(doc_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", doc_name.lower()).strip("_")


def infer_source_type(doc_type: str) -> str:
    normalized = (doc_type or "").strip().lower()
    if normalized in {"10k", "10k_annualreport"}:
        return "annual_report"
    if normalized == "10q":
        return "quarterly_report"
    if normalized == "8k":
        return "quarterly_results"
    if normalized == "earnings":
        return "earnings_material"
    return "external_financial_doc"


def infer_period(doc_name: str, doc_period: int | str | None) -> str:
    quarter_match = re.search(r"_(\d{4})Q([1-4])_", doc_name)
    if quarter_match:
        return f"{quarter_match.group(1)}Q{quarter_match.group(2)}"

    dated_match = re.search(r"dated-(\d{4})-\d{2}-\d{2}", doc_name, re.IGNORECASE)
    if dated_match:
        return f"{dated_match.group(1)}_event"

    year_match = re.search(r"_(\d{4})_", doc_name)
    if year_match:
        year = year_match.group(1)
        doc_name_upper = doc_name.upper()
        if "10K" in doc_name_upper or "ANNUALREPORT" in doc_name_upper:
            return f"{year}FY"
        if "10Q" in doc_name_upper:
            return f"{year}_quarterly"
        if "EARNINGS" in doc_name_upper or "8K" in doc_name_upper:
            return f"{year}_event"

    if doc_period:
        return str(doc_period)
    return "unknown_period"


def infer_document_title(company: str, doc_name: str, doc_type: str, period: str) -> str:
    label = doc_type.replace("_", " ").upper()
    return f"{company} {label} {period} ({doc_name})"


def infer_doc_suffix(doc_link: str) -> str:
    match = re.search(r"(\.[a-z0-9]{2,5})(?:\?|$)", doc_link, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ".pdf"


def financebench_pdf_mirror_url(doc_name: str) -> str:
    return f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"


def load_pdf_tree_paths(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["path"].split("/", 1)[1]
        for entry in payload.get("tree", [])
        if entry.get("path", "").startswith("pdfs/")
    }


def extract_numeric_tokens(text: str) -> list[str]:
    patterns = [
        r"\$[\d,.]+(?:\s?(?:billion|million|thousand|bn|mn|m|b))?",
        r"[\d,.]+%",
        r"[\d,.]+\s?(?:basis points|bps)",
        r"[\d,.]+",
    ]
    tokens: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            token = normalize_whitespace(match)
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def derive_query_types(question_type: str, question_reasoning: str | None) -> list[str]:
    query_types: list[str] = []
    reasoning = (question_reasoning or "").lower()
    qtype = (question_type or "").lower()

    if "numerical reasoning" in reasoning or qtype == "metrics-generated":
        query_types.append("numeric_grounding")
    if "information extraction" in reasoning:
        query_types.append("factual_lookup")
    if "logical reasoning" in reasoning or qtype in {"domain-relevant", "novel-generated"}:
        query_types.append("reasoned_analysis")
    if not query_types:
        query_types.append("factual_lookup")
    return sorted(set(query_types))


def derive_difficulty(question_reasoning: str | None) -> str:
    reasoning = (question_reasoning or "").lower()
    if "logical reasoning" in reasoning:
        return "hard"
    if "numerical reasoning" in reasoning:
        return "medium"
    return "easy"


def derive_must_cover(answer: str, evidence: str) -> list[str]:
    answer = normalize_whitespace(answer)
    evidence = normalize_whitespace(evidence)
    numeric_tokens = extract_numeric_tokens(answer)

    must_cover: list[str] = []
    if numeric_tokens:
        must_cover.extend(numeric_tokens[:2])

    compact_answer = answer if 4 <= len(answer) <= 96 else ""
    if compact_answer and compact_answer.lower() not in {"yes", "no"}:
        must_cover.append(compact_answer)

    if answer.lower() in {"yes", "no"} and evidence:
        must_cover.append(evidence[:120].rstrip())
    elif not must_cover and evidence:
        must_cover.append(evidence[:120].rstrip())

    deduped: list[str] = []
    seen: set[str] = set()
    for entry in must_cover:
        key = entry.lower()
        if key and key not in seen:
            deduped.append(entry)
            seen.add(key)
    return deduped[:3]


def build_evaluation_mapping(row: dict, document: dict, company_pack: str) -> dict:
    answer = normalize_whitespace(row["answer"])
    evidence = normalize_whitespace(row.get("evidence"))
    numeric_tokens = extract_numeric_tokens(answer)
    if numeric_tokens:
        mode = "numeric_exact"
    elif answer.lower() in {"yes", "no"}:
        mode = "boolean_with_evidence"
    else:
        mode = "semantic_gold_answer"

    return {
        "source": "financebench_open_source",
        "mode": mode,
        "expected_answer": answer,
        "expected_justification": normalize_whitespace(row.get("justification")),
        "expected_evidence": evidence,
        "normalized_answer_tokens": numeric_tokens,
        "required_doc_id": normalize_doc_id(document["doc_name"]),
        "required_company_pack": company_pack,
        "required_source_type": infer_source_type(document["doc_type"]),
        "required_period": infer_period(document["doc_name"], document.get("doc_period")),
    }


def build_item(row: dict, document: dict, company_pack: str) -> dict:
    evaluation_mapping = build_evaluation_mapping(row, document, company_pack)
    return {
        "item_id": row["financebench_id"],
        "category": "financebench_external",
        "companies": [company_pack],
        "query": row["question"],
        "required_source_types": [evaluation_mapping["required_source_type"]],
        "target_periods": [evaluation_mapping["required_period"]],
        "query_type": derive_query_types(row["question_type"], row.get("question_reasoning")),
        "difficulty": derive_difficulty(row.get("question_reasoning")),
        "required_evidence": {
            "min_distinct_sources": 1,
            "must_cover": derive_must_cover(row["answer"], row.get("evidence", "")),
        },
        "acceptable_uncertainty": False,
        "expected_failure_modes": ["wrong_company", "wrong_period", "answer_not_grounded"],
        "scoring_focus": ["retrieval", "answer_mapping", "claim_verification"],
    }


def build_answer(row: dict, document: dict, company_pack: str) -> dict:
    evaluation_mapping = build_evaluation_mapping(row, document, company_pack)
    must_cover = derive_must_cover(row["answer"], row.get("evidence", ""))
    gold_outline = [normalize_whitespace(row["answer"])]
    justification = normalize_whitespace(row.get("justification"))
    if justification:
        gold_outline.append(justification)
    return {
        "item_id": row["financebench_id"],
        "gold_answer": normalize_whitespace(row["answer"]),
        "gold_justification": justification,
        "gold_evidence": normalize_whitespace(row.get("evidence")),
        "gold_outline": gold_outline,
        "must_cover": must_cover,
        "must_not_claim": [],
        "evaluation_mapping": evaluation_mapping,
    }


def build_evidence(row: dict, document: dict, company_pack: str) -> dict:
    period = infer_period(document["doc_name"], document.get("doc_period"))
    return {
        "item_id": row["financebench_id"],
        "evidence": [
            {
                "company": company_pack,
                "doc_id": normalize_doc_id(document["doc_name"]),
                "source_type": infer_source_type(document["doc_type"]),
                "period": period,
                "anchor_text": normalize_whitespace(row.get("evidence")) or normalize_whitespace(row["answer"]),
                "section_hint": document["doc_name"],
                "evidence_role": "primary",
                "evidence_strength": "high",
                "binding_method": "gold_evidence_text",
            }
        ],
    }


def build_manifest(company_name: str, company_pack: str, docs: list[dict]) -> tuple[dict, dict]:
    periods = sorted({doc["period"] for doc in docs if doc["period"] and doc["period"] != "unknown_period"})
    latest_period = periods[-1] if periods else None
    annual_periods = [period for period in periods if period.endswith("FY")]
    comparison_period = periods[-2] if len(periods) >= 2 else None
    fiscal_year_end = "12-31"

    manifest = {
        "company": company_pack,
        "aliases": [company_name],
        "currency": "USD",
        "documents": [],
    }

    for doc in sorted(docs, key=lambda entry: entry["doc_id"]):
        manifest["documents"].append(
            {
                "doc_id": doc["doc_id"],
                "path": doc["path"],
                "title": doc["title"],
                "source_type": doc["source_type"],
                "period": doc["period"],
                "doc_date": doc.get("doc_date"),
                "published_at": doc.get("published_at"),
                "issuer": company_name,
                "is_primary": True,
                "url": doc["url"],
                "language": "en",
            }
        )

    periods_payload = {
        "company": company_pack,
        "available_periods": periods,
        "context_periods": [],
        "latest_period": latest_period,
        "comparison_period": comparison_period,
        "fiscal_year_end": fiscal_year_end,
    }
    return manifest, periods_payload


def prepare_financebench_dataset(
    open_source_rows: list[dict],
    document_rows: list[dict],
    *,
    available_pdf_paths: set[str] | None = None,
) -> dict:
    docs_by_name = {row["doc_name"]: row for row in document_rows}
    available_pdf_paths = available_pdf_paths or set()
    items: list[dict] = []
    answers: list[dict] = []
    evidence_rows: list[dict] = []
    source_documents: list[dict] = []
    docs_by_pack: dict[str, list[dict]] = defaultdict(list)
    seen_doc_ids: set[tuple[str, str]] = set()

    for row in open_source_rows:
        doc_name = row["doc_name"]
        if doc_name not in docs_by_name:
            raise ValueError(f"Missing document metadata for FinanceBench doc_name={doc_name}")
        document = docs_by_name[doc_name]
        company_pack = slugify_company(row["company"])
        source_type = infer_source_type(document["doc_type"])
        period = infer_period(document["doc_name"], document.get("doc_period"))
        doc_id = normalize_doc_id(document["doc_name"])
        doc_suffix = infer_doc_suffix(document["doc_link"])
        relative_path = f"data/raw/{company_pack}/docs/{doc_id}{doc_suffix}"

        items.append(build_item(row, document, company_pack))
        answers.append(build_answer(row, document, company_pack))
        evidence_rows.append(build_evidence(row, document, company_pack))

        unique_key = (company_pack, doc_id)
        if unique_key not in seen_doc_ids:
            mirror_path = f"{document['doc_name']}.pdf"
            if mirror_path in available_pdf_paths:
                source_url = financebench_pdf_mirror_url(document["doc_name"])
            else:
                source_url = document["doc_link"]
            source_doc = {
                "company_pack": company_pack,
                "company_name": row["company"],
                "doc_id": doc_id,
                "doc_name": document["doc_name"],
                "doc_type": document["doc_type"],
                "source_type": source_type,
                "period": period,
                "title": infer_document_title(row["company"], document["doc_name"], document["doc_type"], period),
                "url": source_url,
                "origin_url": document["doc_link"],
                "path": relative_path,
                "doc_date": None,
                "published_at": None,
                "gics_sector": document.get("gics_sector"),
                "question_count": 0,
            }
            source_documents.append(source_doc)
            docs_by_pack[company_pack].append(source_doc)
            seen_doc_ids.add(unique_key)

    question_counts = Counter(row["company_pack"] for row in source_documents)
    for source_doc in source_documents:
        source_doc["question_count"] = sum(
            1 for row in open_source_rows
            if slugify_company(row["company"]) == source_doc["company_pack"]
            and normalize_doc_id(row["doc_name"]) == source_doc["doc_id"]
        )

    manifests: dict[str, dict] = {}
    periods: dict[str, dict] = {}
    companies_summary = []
    for company_pack, docs in sorted(docs_by_pack.items()):
        company_name = docs[0]["company_name"]
        manifest, periods_payload = build_manifest(company_name, company_pack, docs)
        manifests[company_pack] = manifest
        periods[company_pack] = periods_payload
        companies_summary.append(
            {
                "company_pack": company_pack,
                "company_name": company_name,
                "document_count": len(docs),
                "question_count": sum(1 for row in open_source_rows if slugify_company(row["company"]) == company_pack),
                "available_periods": periods_payload["available_periods"],
            }
        )

    summary = {
        "version": BENCHMARK_VERSION,
        "item_count": len(items),
        "company_count": len(manifests),
        "document_count": len(source_documents),
        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in open_source_rows).items())),
        "source_type_counts": dict(sorted(Counter(doc["source_type"] for doc in source_documents).items())),
        "company_question_counts": {
            company["company_pack"]: company["question_count"] for company in companies_summary
        },
    }

    download_plan = {
        "version": BENCHMARK_VERSION,
        "document_count": len(source_documents),
        "company_packs": [company["company_pack"] for company in companies_summary],
        "commands": [
            {
                "description": "Fetch all FinanceBench source packs",
                "command": "python tools/fetch_benchmark_sources.py "
                + " ".join(f"--company {company['company_pack']}" for company in companies_summary),
            }
        ],
    }

    return {
        "items": items,
        "answers": answers,
        "evidence": evidence_rows,
        "source_documents": source_documents,
        "manifests": manifests,
        "periods": periods,
        "companies": companies_summary,
        "summary": summary,
        "download_plan": download_plan,
    }


def write_prepared_dataset(prepared: dict, benchmark_root: Path, raw_root: Path) -> None:
    benchmark_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(benchmark_root / "items.jsonl", prepared["items"])
    write_jsonl(benchmark_root / "answers.jsonl", prepared["answers"])
    write_jsonl(benchmark_root / "evidence.jsonl", prepared["evidence"])
    write_jsonl(benchmark_root / "source_documents.jsonl", prepared["source_documents"])
    write_json(benchmark_root / "companies.json", {"companies": prepared["companies"]})
    write_json(benchmark_root / "summary.json", prepared["summary"])
    write_json(benchmark_root / "download_plan.json", prepared["download_plan"])

    for company_pack, manifest in prepared["manifests"].items():
        company_root = raw_root / company_pack
        company_root.mkdir(parents=True, exist_ok=True)
        (company_root / "docs").mkdir(parents=True, exist_ok=True)
        write_json(company_root / "manifest.json", manifest)
        write_json(company_root / "periods.json", prepared["periods"][company_pack])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FinanceBench open-source 150 subset.")
    parser.add_argument("--open-source-jsonl", type=Path, required=True, help="Path to financebench_open_source.jsonl")
    parser.add_argument(
        "--document-info-jsonl",
        type=Path,
        required=True,
        help="Path to financebench_document_information.jsonl",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=settings.data_dir / "benchmark" / BENCHMARK_VERSION,
        help="Output directory for benchmark-compatible JSONL files.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=settings.data_dir / "raw",
        help="Raw source root where per-company manifests should be written.",
    )
    parser.add_argument(
        "--pdf-tree-json",
        type=Path,
        default=None,
        help="Optional GitHub tree JSON for FinanceBench pdf mirror availability.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    open_source_rows = load_jsonl(args.open_source_jsonl)
    document_rows = load_jsonl(args.document_info_jsonl)
    available_pdf_paths = load_pdf_tree_paths(args.pdf_tree_json)
    prepared = prepare_financebench_dataset(
        open_source_rows,
        document_rows,
        available_pdf_paths=available_pdf_paths,
    )
    write_prepared_dataset(prepared, args.benchmark_root, args.raw_root)
    print(json.dumps(prepared["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
