import json

from tools.prepare_offline_suite import load_suite_spec, prepare_suite, write_report


def test_prepare_offline_suite_reports_complete_assets(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    offline_suite_dir = data_dir / "offline_suite"
    raw_dir = data_dir / "raw" / "demo"
    normalized_dir = data_dir / "normalized" / "demo"
    processed_dir = data_dir / "processed" / "demo"
    benchmark_dir = data_dir / "benchmark" / "vtest"
    company_packs_dir = data_dir / "company_packs" / "demo"

    offline_suite_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    benchmark_dir.mkdir(parents=True)
    company_packs_dir.mkdir(parents=True)

    (offline_suite_dir / "samples.json").write_text(
        json.dumps(
            {
                "companies": [
                    {"company_id": "demo", "roles": ["demo", "benchmark"]}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {
                "company": "demo",
                "aliases": ["Demo"],
                "documents": [
                    {
                        "doc_id": "demo_doc",
                        "source_type": "quarterly_results",
                        "period": "2025Q4",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (normalized_dir / "documents.jsonl").write_text(
        json.dumps({"doc_id": "demo_doc", "source_type": "quarterly_results"}) + "\n",
        encoding="utf-8",
    )
    (normalized_dir / "chunks.jsonl").write_text(
        json.dumps({"doc_id": "demo_doc", "chunk_id": "chunk-1", "source_id": "demo_doc", "text": "Revenue was $10."}) + "\n",
        encoding="utf-8",
    )
    (processed_dir / "sources.json").write_text(
        json.dumps([{"source_id": "demo_doc", "source_type": "quarterly_results"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (processed_dir / "chunks.json").write_text(
        json.dumps([{"chunk_id": "chunk-1", "source_id": "demo_doc", "text": "Revenue was $10."}], ensure_ascii=False),
        encoding="utf-8",
    )
    (benchmark_dir / "items.jsonl").write_text(
        json.dumps(
            {
                "item_id": "D-001",
                "category": "company_overview",
                "companies": ["demo"],
                "query": "Summarize Demo's latest quarter with revenue evidence.",
                "required_source_types": ["quarterly_results"],
                "target_periods": ["2025Q4"],
                "query_type": ["numeric_grounding"],
                "difficulty": "medium",
                "required_evidence": {
                    "min_distinct_sources": 1,
                    "must_cover": ["revenue"],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "answers.jsonl").write_text(
        json.dumps({"item_id": "D-001", "must_cover": ["revenue"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "evidence.jsonl").write_text(
        json.dumps(
            {
                "item_id": "D-001",
                "evidence": [
                    {
                        "company": "demo",
                        "doc_id": "demo_doc",
                        "source_type": "quarterly_results",
                        "period": "2025Q4",
                        "anchor_text": "Revenue was $10.",
                        "chunk_id": "chunk-1",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "profiles.json").write_text(json.dumps({"profiles": {}}, ensure_ascii=False), encoding="utf-8")
    (benchmark_dir / "splits.json").write_text(json.dumps({"dev": ["D-001"], "test": []}, ensure_ascii=False), encoding="utf-8")

    from tools import prepare_offline_suite as module

    monkeypatch.setattr(module.settings, "data_dir", data_dir)
    monkeypatch.setattr(module.settings, "company_packs_dir", data_dir / "company_packs")

    report = prepare_suite(["vtest"])

    assert report["all_required_assets_present"] is True
    assert report["companies"][0]["counts"]["processed_chunks"] == 1
    assert report["companies"][0]["benchmark_items"]["vtest"] == ["D-001"]
    assert report["benchmark_assets"]["vtest"]["local_fact_count"] == 1

    report_path = write_report(report, data_dir / "offline_suite" / "report.json")
    assert report_path.exists()


def test_repo_offline_suite_spec_references_curated_sample_companies():
    spec = load_suite_spec()
    company_ids = [row["company_id"] for row in spec["companies"]]

    assert company_ids == ["stripe", "cloudflare", "fastly"]
