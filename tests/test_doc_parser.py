import json

from tools.doc_parser import load_company_pack


def test_load_company_pack_includes_supplemental_json(tmp_path):
    company_dir = tmp_path / "demo"
    company_dir.mkdir()

    (company_dir / "profile.json").write_text(
        json.dumps(
            {
                "company_name": "DemoCo",
                "founded": "2020",
                "headquarters": "Hong Kong",
                "founders": ["A", "B"],
                "employee_count": "10",
                "latest_valuation": "$100M",
                "total_funding": "$10M",
                "latest_round": "Series A",
                "industry": "Software",
                "key_products": ["Demo"],
                "target_customers": "SMBs",
            }
        ),
        encoding="utf-8",
    )
    (company_dir / "memo_notes.json").write_text(
        json.dumps(
            {
                "source_id": "demo_notes",
                "title": "Demo Notes",
                "source_url": "https://example.com/demo",
                "date": "2024-01-01",
                "content": "Observation: demo facts.\n\nImplication: useful memo source.",
            }
        ),
        encoding="utf-8",
    )

    documents = load_company_pack(company_dir)
    source_ids = [meta.source_id for meta, _ in documents]

    assert "demo_profile" in source_ids
    assert "demo_notes" in source_ids
