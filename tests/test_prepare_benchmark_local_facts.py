import json

from tools.prepare_benchmark_local_facts import build_local_facts, select_fact_text


def test_select_fact_text_prefers_atomic_sentence():
    chunk_text = (
        "Overview. Record fourth quarter revenue of $172.6 million grew 23% year over year. "
        "Management also said demand improved materially."
    )

    fact_text = select_fact_text(
        "Record fourth quarter revenue of $172.6 million grew 23% year over year",
        chunk_text,
    )

    assert fact_text == "Record fourth quarter revenue of $172.6 million grew 23% year over year."


def test_select_fact_text_uses_context_to_skip_heading_noise():
    chunk_text = (
        "[Page 1]\nFourth Quarter 2025 Investor Supplement\n"
        "Product Innovation and Developments Corporate Highlights\n"
        "Revenue by Product (in millions):\n"
        "Network Services Revenue $130.8\n"
    )

    fact_text = select_fact_text("product mix", chunk_text, "revenue mix management emphasis")

    assert "Product Innovation and Developments Corporate Highlights" not in fact_text
    assert "Revenue" in fact_text


def test_build_local_facts_attaches_source_attributes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    benchmark_dir = data_dir / "benchmark" / "vtest"
    normalized_dir = data_dir / "normalized" / "fastly"
    benchmark_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)

    (benchmark_dir / "items.jsonl").write_text(
        json.dumps(
            {
                "item_id": "NUM-001",
                "category": "company_overview",
                "companies": ["fastly"],
                "query": "Summarize the quarter.",
                "required_source_types": ["quarterly_results"],
                "target_periods": ["2025Q4"],
                "query_type": ["numeric_grounding"],
                "difficulty": "medium",
                "required_evidence": {"min_distinct_sources": 1, "must_cover": ["revenue"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "answers.jsonl").write_text(
        json.dumps(
            {"item_id": "NUM-001", "must_cover": ["reported revenue"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "evidence.jsonl").write_text(
        json.dumps(
            {
                "item_id": "NUM-001",
                "evidence": [
                    {
                        "company": "fastly",
                        "doc_id": "fastly_q4_2025_results",
                        "source_type": "quarterly_results",
                        "period": "2025Q4",
                        "anchor_text": "Record fourth quarter revenue of $172.6 million grew 23% year over year",
                        "section_hint": "Headline results",
                        "evidence_role": "primary",
                        "evidence_strength": "high",
                        "chunk_id": "chunk-001",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (normalized_dir / "documents.jsonl").write_text(
        json.dumps(
            {
                "company": "fastly",
                "doc_id": "fastly_q4_2025_results",
                "title": "Fastly Q4 2025 Results",
                "source_type": "quarterly_results",
                "period": "2025Q4",
                "published_at": "2026-02-11",
                "issuer": "Fastly, Inc.",
                "is_primary": True,
                "url": "https://investors.fastly.com/q4-2025-results",
                "text": "Record fourth quarter revenue of $172.6 million grew 23% year over year.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (normalized_dir / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-001",
                "chunk_index": 0,
                "company": "fastly",
                "doc_id": "fastly_q4_2025_results",
                "source_id": "fastly_q4_2025_results",
                "source_type": "quarterly_results",
                "period": "2025Q4",
                "title": "Fastly Q4 2025 Results",
                "page": None,
                "token_count": 18,
                "text": "Record fourth quarter revenue of $172.6 million grew 23% year over year.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    from tools import prepare_benchmark_local_facts as module

    monkeypatch.setattr(module.settings, "data_dir", data_dir)

    facts = build_local_facts("vtest")

    assert len(facts) == 1
    assert facts[0]["fact"] == "Record fourth quarter revenue of $172.6 million grew 23% year over year."
    assert facts[0]["attributes"]["title"] == "Fastly Q4 2025 Results"
    assert facts[0]["attributes"]["author"] == "Fastly, Inc."
    assert facts[0]["attributes"]["url"] == "https://investors.fastly.com/q4-2025-results"
    assert facts[0]["attributes"]["required_slots"] == ["revenue"]
    assert facts[0]["attributes"]["gold_must_cover"] == ["reported revenue"]
