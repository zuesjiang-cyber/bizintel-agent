import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "download_financebench_missing_docs.py"
SPEC = importlib.util.spec_from_file_location("bizintel_download_financebench_missing_docs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_doc(**overrides):
    payload = {
        "company_pack": "financebench_pepsico",
        "doc_id": "pepsico_2023_8k_dated_2023_05_05",
        "title": "PepsiCo 8K 2023_event",
        "source_type": "quarterly_results",
        "period": "2023_event",
        "issuer": "PepsiCo",
        "target_path": "data/raw/financebench_pepsico/docs/pepsico_2023_8k_dated_2023_05_05.pdf",
        "original_url": "https://pepsico.gcs-web.com/static-files/718629be-2463-4b54-bba3-5e0e776e7d0c",
        "doc_name": "PEPSICO_2023_8K_dated-2023-05-05",
    }
    payload.update(overrides)
    return MODULE.MissingDocument(**payload)


def test_filing_targets_for_dated_event():
    doc = make_doc()
    forms, target_date = MODULE.filing_targets(doc)
    assert forms == ["8-K", "8-K/A"]
    assert target_date.isoformat() == "2023-05-05"


def test_filing_score_prefers_matching_annual_report_date():
    doc = make_doc(
        company_pack="financebench_verizon",
        doc_id="verizon_2021_10k",
        source_type="annual_report",
        period="2021FY",
    )
    good = {
        "form": "10-K",
        "reportDate": "2021-12-31",
        "filingDate": "2022-02-18",
        "primaryDocument": "vzk202110-k.htm",
    }
    bad = {
        "form": "10-K",
        "reportDate": "2020-12-31",
        "filingDate": "2021-02-19",
        "primaryDocument": "vzk202010-k.htm",
    }
    assert MODULE.filing_score(doc, good) > MODULE.filing_score(doc, bad)


def test_rank_index_entry_prefers_ex99_for_earnings():
    doc = make_doc()
    filing = {"primaryDocument": "form8k.htm"}
    assert MODULE.rank_index_entry(doc, filing, "ex99-1.pdf") > MODULE.rank_index_entry(doc, filing, "form8k.htm")


def test_rewrite_manifest_path_updates_manifest_and_source_index(tmp_path, monkeypatch):
    raw_root = tmp_path / "data" / "raw"
    pack_dir = raw_root / "financebench_test"
    pack_dir.mkdir(parents=True)
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "company": "financebench_test",
                "documents": [
                    {
                        "doc_id": "test_doc",
                        "path": "data/raw/financebench_test/docs/test_doc.pdf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_index = tmp_path / "source_documents.jsonl"
    source_index.write_text(
        json.dumps(
            {
                "doc_id": "test_doc",
                "path": "data/raw/financebench_test/docs/test_doc.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(MODULE, "RAW_ROOT", raw_root)
    monkeypatch.setattr(MODULE, "SOURCE_DOC_INDEX", source_index)

    MODULE.rewrite_manifest_path("financebench_test", "test_doc", "data/raw/financebench_test/docs/test_doc.html")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["documents"][0]["path"].endswith("test_doc.html")
    source_rows = [json.loads(line) for line in source_index.read_text().splitlines() if line.strip()]
    assert source_rows[0]["path"].endswith("test_doc.html")


def test_candidate_urls_prefers_known_direct_sec_source():
    session = MODULE.build_session()
    doc = make_doc(
        company_pack="financebench_kraft_heinz",
        doc_id="kraftheinz_2019_10k",
        source_type="annual_report",
        period="2019FY",
        target_path="data/raw/financebench_kraft_heinz/docs/kraftheinz_2019_10k.html",
        original_url="https://ir.kraftheinzcompany.com/static-files/2d2e9a1f-a7bc-4c07-9e5e-77aa60be8f86",
        doc_name="KRAFTHEINZ_2019_10K",
    )

    candidates = MODULE.candidate_urls(session, doc)

    assert candidates[0].label == "preferred_direct"
    assert "sec.gov/Archives/edgar/data/1637459/000163745920000027/form10-k2019.htm" in candidates[0].url
