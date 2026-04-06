import json
from pathlib import Path

from tools.fetch_benchmark_sources import (
    download_document,
    financebench_github_api_url,
    file_sha256,
    load_manifest,
    resolve_destination,
    select_documents,
)


def test_select_documents_filters_by_doc_id():
    manifest = {
        "documents": [
            {"doc_id": "a", "path": "data/raw/cloudflare/docs/a.html", "url": "https://example.com/a"},
            {"doc_id": "b", "path": "data/raw/cloudflare/docs/b.html", "url": "https://example.com/b"},
        ]
    }

    selected = select_documents(manifest, ["b"])

    assert [doc["doc_id"] for doc in selected] == ["b"]


def test_manifests_point_to_docs_subdirectories():
    repo_root = Path(__file__).resolve().parents[1]
    for company in ("cloudflare", "fastly"):
        manifest_path = repo_root / "data" / "raw" / company / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for document in payload["documents"]:
            assert document["path"].startswith(f"data/raw/{company}/docs/")


def test_cloudflare_manifest_uses_verified_q4_exhibit():
    manifest = load_manifest("cloudflare")
    doc_ids = {document["doc_id"] for document in manifest["documents"]}

    assert "cloudflare_q4_2025_results_exhibit_99_1" in doc_ids
    assert "cloudflare_q4_2025_investor_presentation" not in doc_ids


def test_resolve_destination_stays_under_data_raw():
    destination = resolve_destination("data/raw/cloudflare/docs/cloudflare_ir_overview.html", "cloudflare")

    assert destination.as_posix().endswith("/data/raw/cloudflare/docs/cloudflare_ir_overview.html")


def test_resolve_destination_rejects_cross_company_paths():
    try:
        resolve_destination("data/raw/fastly/docs/fastly_ir_overview.html", "cloudflare")
    except ValueError as exc:
        assert "cloudflare/docs" in str(exc)
    else:
        raise AssertionError("expected ValueError for cross-company path")


def test_file_sha256_is_stable(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello", encoding="utf-8")

    assert file_sha256(target) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_download_document_uses_total_timeout(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    company = "cloudflare"
    destination = repo_root / "data" / "raw" / company / "docs" / "timeout_test.pdf"
    if destination.exists():
        destination.unlink()
    part_path = destination.with_suffix(".pdf.part")
    if part_path.exists():
        part_path.unlink()

    calls = {}

    class Completed:
        stdout = '{"content_type":"application/pdf","url_effective":"https://example.com","http_code":200,"size_download":5}\n'

    def fake_run(command, check, capture_output, text):
        calls["command"] = command
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"hello")
        return Completed()

    monkeypatch.setattr("tools.fetch_benchmark_sources.subprocess.run", fake_run)
    monkeypatch.setattr("tools.fetch_benchmark_sources.append_receipt", lambda *args, **kwargs: None)

    status = download_document(
        company,
        {
            "doc_id": "timeout_test",
            "path": f"data/raw/{company}/docs/timeout_test.pdf",
            "url": "https://example.com/timeout.pdf",
        },
        force=True,
    )

    assert "saved timeout_test" in status
    assert "--max-time" in calls["command"]
    assert "120" in calls["command"]
    if destination.exists():
        destination.unlink()


def test_financebench_github_api_url_rewrites_raw_pdf_url():
    source = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/ACTIVISIONBLIZZARD_2019_10K.pdf"

    assert (
        financebench_github_api_url(source)
        == "https://api.github.com/repos/patronus-ai/financebench/contents/pdfs/ACTIVISIONBLIZZARD_2019_10K.pdf?ref=main"
    )
