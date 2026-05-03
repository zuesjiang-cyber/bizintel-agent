import json

from tools.document_fidelity_harness import analyze_processed_root, render_markdown


def test_document_fidelity_harness_reports_metadata_issues(tmp_path):
    processed = tmp_path / "processed"
    company = processed / "demo_company"
    company.mkdir(parents=True)
    (company / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "demo_10k",
                    "source_type": "10-K",
                    "company": "Demo Co",
                    "date": "2024-02-01",
                    "primary_source": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (company / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": "c1",
                    "source_id": "demo_10k",
                    "page": 4,
                    "text": "Revenue was $10 million in 2023. Gross margin was 40%.",
                },
                {
                    "chunk_id": "",
                    "source_id": "missing_source",
                    "page": None,
                    "text": "short",
                },
            ]
        ),
        encoding="utf-8",
    )

    report = analyze_processed_root(processed, ["demo_company"])
    markdown = render_markdown(report)

    assert report["company_count"] == 1
    assert report["summaries"][0]["chunks_with_numeric_content"] == 1
    assert report["issue_counts"]["missing_chunk_id"] == 1
    assert report["issue_counts"]["missing_page"] == 1
    assert "Document Fidelity Report" in markdown
