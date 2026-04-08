import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_financebench_open150.py"
SPEC = importlib.util.spec_from_file_location("bizintel_prepare_financebench_open150", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

build_evaluation_mapping = MODULE.build_evaluation_mapping
derive_must_cover = MODULE.derive_must_cover
normalize_doc_id = MODULE.normalize_doc_id
prepare_financebench_dataset = MODULE.prepare_financebench_dataset
slugify_company = MODULE.slugify_company
write_prepared_dataset = MODULE.write_prepared_dataset


def test_slugify_company_uses_financebench_prefix():
    assert slugify_company("American Express") == "financebench_american_express"


def test_derive_must_cover_prefers_numeric_answer_tokens():
    must_cover = derive_must_cover("$1577.00", "capital expenditures were $1,577 million")

    assert "$1577.00" in must_cover


def test_build_evaluation_mapping_detects_numeric_mode():
    mapping = build_evaluation_mapping(
        {
            "answer": "$1577.00",
            "justification": "The line item is listed in cash flow.",
            "evidence": "Capital expenditures $1,577",
        },
        {
            "doc_name": "3M_2018_10K",
            "doc_type": "10k",
            "doc_period": 2018,
        },
        "financebench_3m",
    )

    assert mapping["mode"] == "numeric_exact"
    assert mapping["required_doc_id"] == normalize_doc_id("3M_2018_10K")
    assert mapping["required_period"] == "2018FY"


def test_prepare_financebench_dataset_builds_items_answers_and_manifests(tmp_path):
    open_source_rows = [
        {
            "financebench_id": "financebench_id_00001",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "question": "What is the FY2018 capital expenditure amount?",
            "answer": "$1577.00",
            "justification": "The cash flow statement lists capital expenditures.",
            "evidence": "Capital expenditures (1,577)",
            "dataset_subset_label": "OPEN_SOURCE",
        },
        {
            "financebench_id": "financebench_id_00002",
            "company": "3M",
            "doc_name": "3M_2023Q2_10Q",
            "question_type": "novel-generated",
            "question_reasoning": "Logical reasoning (based on numerical reasoning)",
            "question": "Is 3M seeing improving profitability in Q2 2023?",
            "answer": "Yes",
            "justification": "Operating income margin expanded year over year.",
            "evidence": "Operating income margin improved",
            "dataset_subset_label": "OPEN_SOURCE",
        },
    ]
    document_rows = [
        {
            "doc_name": "3M_2018_10K",
            "company": "3M",
            "gics_sector": "Industrials",
            "doc_type": "10k",
            "doc_period": 2018,
            "doc_link": "https://example.com/3m_2018_10k.pdf",
        },
        {
            "doc_name": "3M_2023Q2_10Q",
            "company": "3M",
            "gics_sector": "Industrials",
            "doc_type": "10q",
            "doc_period": 2023,
            "doc_link": "https://example.com/3m_2023q2_10q.pdf",
        },
    ]

    prepared = prepare_financebench_dataset(
        open_source_rows,
        document_rows,
        available_pdf_paths={"3M_2018_10K.pdf"},
    )

    assert len(prepared["items"]) == 2
    assert len(prepared["answers"]) == 2
    assert len(prepared["evidence"]) == 2
    assert prepared["summary"]["item_count"] == 2
    assert prepared["summary"]["document_count"] == 2
    assert "financebench_3m" in prepared["manifests"]
    manifest = prepared["manifests"]["financebench_3m"]
    assert manifest["documents"][0]["path"].startswith("data/raw/financebench_3m/docs/")
    assert manifest["documents"][0]["url"].startswith("https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/")
    assert manifest["documents"][1]["url"] == "https://example.com/3m_2023q2_10q.pdf"

    benchmark_root = tmp_path / "benchmark"
    raw_root = tmp_path / "raw"
    write_prepared_dataset(prepared, benchmark_root, raw_root)

    items = [json.loads(line) for line in (benchmark_root / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [json.loads(line) for line in (benchmark_root / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert items[0]["companies"] == ["financebench_3m"]
    assert answers[0]["evaluation_mapping"]["mode"] == "numeric_exact"
    assert (raw_root / "financebench_3m" / "manifest.json").exists()
    assert (raw_root / "financebench_3m" / "periods.json").exists()
