"""
Validate and summarize the curated offline company/sample corpus suite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from agent.config import settings
from tools.prepare_benchmark_local_facts import build_local_facts, write_jsonl


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_suite_spec(path: Path | None = None) -> dict:
    suite_path = path or (settings.data_dir / "offline_suite" / "samples.json")
    return load_json(suite_path)


def ensure_local_facts(version: str, rebuild: bool = False) -> Path:
    output_path = settings.data_dir / "benchmark" / version / "local_facts.jsonl"
    if output_path.exists() and not rebuild:
        return output_path
    rows = build_local_facts(version)
    write_jsonl(output_path, rows)
    return output_path


def benchmark_company_items(version: str, company_id: str) -> list[str]:
    items_path = settings.data_dir / "benchmark" / version / "items.jsonl"
    if not items_path.exists():
        return []
    rows = load_jsonl(items_path)
    return [
        row["item_id"]
        for row in rows
        if company_id in row.get("companies", [])
    ]


def collect_company_summary(company_id: str, roles: list[str], benchmark_versions: list[str]) -> dict:
    raw_manifest_path = settings.data_dir / "raw" / company_id / "manifest.json"
    normalized_documents_path = settings.data_dir / "normalized" / company_id / "documents.jsonl"
    normalized_chunks_path = settings.data_dir / "normalized" / company_id / "chunks.jsonl"
    processed_sources_path = settings.data_dir / "processed" / company_id / "sources.json"
    processed_chunks_path = settings.data_dir / "processed" / company_id / "chunks.json"
    legacy_pack_path = settings.company_packs_dir / company_id

    summary = {
        "company_id": company_id,
        "roles": roles,
        "paths": {
            "raw_manifest": str(raw_manifest_path),
            "normalized_documents": str(normalized_documents_path),
            "normalized_chunks": str(normalized_chunks_path),
            "processed_sources": str(processed_sources_path),
            "processed_chunks": str(processed_chunks_path),
            "legacy_company_pack": str(legacy_pack_path),
        },
        "available": {},
        "counts": {},
        "periods": [],
        "source_types": [],
        "benchmark_items": {},
        "missing": [],
    }

    path_map = {
        "raw_manifest": raw_manifest_path,
        "normalized_documents": normalized_documents_path,
        "normalized_chunks": normalized_chunks_path,
        "processed_sources": processed_sources_path,
        "processed_chunks": processed_chunks_path,
        "legacy_company_pack": legacy_pack_path,
    }
    for name, path in path_map.items():
        summary["available"][name] = path.exists()

    if raw_manifest_path.exists():
        manifest = load_json(raw_manifest_path)
        documents = manifest.get("documents", [])
        summary["counts"]["raw_documents"] = len(documents)
        summary["periods"] = sorted({row.get("period") for row in documents if row.get("period")})
        summary["source_types"] = sorted({row.get("source_type") for row in documents if row.get("source_type")})
        summary["aliases"] = list(manifest.get("aliases", []))
    elif "benchmark" in roles:
        summary["missing"].append("raw_manifest")

    if normalized_documents_path.exists():
        documents = load_jsonl(normalized_documents_path)
        summary["counts"]["normalized_documents"] = len(documents)
        source_type_counter = Counter(row.get("source_type", "unknown") for row in documents)
        summary["normalized_source_type_counts"] = dict(sorted(source_type_counter.items()))
    elif "benchmark" in roles:
        summary["missing"].append("normalized_documents")

    if normalized_chunks_path.exists():
        normalized_chunks = load_jsonl(normalized_chunks_path)
        summary["counts"]["normalized_chunks"] = len(normalized_chunks)
    elif "benchmark" in roles:
        summary["missing"].append("normalized_chunks")

    if processed_sources_path.exists():
        processed_sources = load_json(processed_sources_path)
        summary["counts"]["processed_sources"] = len(processed_sources)
    elif "benchmark" in roles or "single_company_smoke" in roles:
        summary["missing"].append("processed_sources")

    if processed_chunks_path.exists():
        processed_chunks = load_json(processed_chunks_path)
        summary["counts"]["processed_chunks"] = len(processed_chunks)
    elif "benchmark" in roles or "single_company_smoke" in roles:
        summary["missing"].append("processed_chunks")

    if "demo" in roles and not legacy_pack_path.exists():
        summary["missing"].append("legacy_company_pack")

    for version in benchmark_versions:
        summary["benchmark_items"][version] = benchmark_company_items(version, company_id)

    return summary


def prepare_suite(versions: Iterable[str], rebuild_local_facts: bool = False) -> dict:
    spec = load_suite_spec()
    versions = [version.strip() for version in versions if version.strip()]

    benchmark_assets = {}
    for version in versions:
        local_facts_path = ensure_local_facts(version, rebuild=rebuild_local_facts)
        benchmark_assets[version] = {
            "local_facts_path": str(local_facts_path),
            "local_fact_count": len(load_jsonl(local_facts_path)),
            "items_path": str(settings.data_dir / "benchmark" / version / "items.jsonl"),
            "profiles_path": str(settings.data_dir / "benchmark" / version / "profiles.json"),
            "splits_path": str(settings.data_dir / "benchmark" / version / "splits.json"),
        }

    companies = [
        collect_company_summary(
            company_id=entry["company_id"],
            roles=entry.get("roles", []),
            benchmark_versions=versions,
        )
        for entry in spec.get("companies", [])
    ]
    missing = {
        row["company_id"]: row["missing"]
        for row in companies
        if row["missing"]
    }

    return {
        "schema_version": 1,
        "sample_companies": [entry["company_id"] for entry in spec.get("companies", [])],
        "benchmark_versions": versions,
        "benchmark_assets": benchmark_assets,
        "companies": companies,
        "missing": missing,
        "all_required_assets_present": not missing,
    }


def write_report(report: dict, output_path: Path | None = None) -> Path:
    report_path = output_path or (settings.data_dir / "offline_suite" / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate the curated offline sample corpus suite.")
    parser.add_argument("--versions", default="v2", help="Comma-separated benchmark versions to validate.")
    parser.add_argument("--rebuild-local-facts", action="store_true", help="Rebuild local_facts.jsonl even if already present.")
    parser.add_argument("--write-report", action="store_true", help="Write a JSON report to data/offline_suite/report.json.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if required offline assets are missing.")
    args = parser.parse_args()

    versions = [item.strip() for item in args.versions.split(",") if item.strip()]
    report = prepare_suite(versions=versions, rebuild_local_facts=args.rebuild_local_facts)
    report_path = None
    if args.write_report:
        report_path = write_report(report)

    print(
        json.dumps(
            {
                "sample_companies": report["sample_companies"],
                "benchmark_versions": report["benchmark_versions"],
                "all_required_assets_present": report["all_required_assets_present"],
                "missing_company_count": len(report["missing"]),
                "report_path": str(report_path) if report_path else None,
            },
            ensure_ascii=False,
        )
    )
    if args.strict and not report["all_required_assets_present"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
