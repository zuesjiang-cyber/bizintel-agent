from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools.fetch_benchmark_sources import append_receipt, file_sha256


REQUEST_TIMEOUT = 45
MAX_RETRIES = 2
BACKOFF_FACTOR = 1.0
VERIFY_SSL = True
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "BizIntelAgent/1.0 financebench-fetcher")

REPO_ROOT = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
SOURCE_DOC_INDEX = REPO_ROOT / "data" / "benchmark" / "financebench_open150" / "source_documents.jsonl"
CSV_LOG_PATH = REPO_ROOT / "download_results.csv"
JSON_LOG_PATH = REPO_ROOT / "download_results.json"

PREFERRED_DIRECT_URLS = {
    "kraftheinz_2019_10k": "https://www.sec.gov/Archives/edgar/data/1637459/000163745920000027/form10-k2019.htm",
}


# Only the still-missing companies need explicit fallback help.
SEC_CIKS = {
    "financebench_kraft_heinz": "0001637459",
    "financebench_johnson_johnson": "0000200406",
    "financebench_lockheed_martin": "0000936468",
    "financebench_mgm_resorts": "0000789570",
    "financebench_microsoft": "0000789019",
    "financebench_netflix": "0001065280",
    "financebench_nike": "0000320187",
    "financebench_paypal": "0001633917",
    "financebench_pepsico": "0000077476",
    "financebench_pfizer": "0000078003",
    "financebench_ulta_beauty": "0001403568",
    "financebench_verizon": "0000732712",
    "financebench_walmart": "0000104169",
}


@dataclass
class MissingDocument:
    company_pack: str
    doc_id: str
    title: str
    source_type: str
    period: str
    issuer: str
    target_path: str
    original_url: str
    doc_name: str | None = None


@dataclass
class Candidate:
    label: str
    url: str
    filing_accession: str | None = None


@dataclass
class DownloadResult:
    doc_id: str
    company_pack: str
    source_type: str
    period: str
    status: str
    selected_label: str
    selected_url: str
    target_path: str
    http_status: int | None
    bytes_written: int
    sha256: str
    error: str
    started_at: str
    finished_at: str
    duration_sec: float


def build_session() -> requests.Session:
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/pdf,text/html,application/octet-stream,*/*",
        }
    )
    return session


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def source_doc_lookup() -> dict[str, dict]:
    rows = read_jsonl(SOURCE_DOC_INDEX)
    return {row["doc_id"]: row for row in rows}


def existing_paths(path_value: str) -> list[Path]:
    target = (REPO_ROOT / path_value).resolve()
    if target.suffix.lower() == ".pdf":
        return [target, target.with_suffix(".html"), target.with_suffix(".htm")]
    if target.suffix.lower() in {".html", ".htm"}:
        return [target, target.with_suffix(".pdf")]
    return [target]


def iter_missing_documents(
    company_filter: Iterable[str] | None = None,
    doc_id_filter: Iterable[str] | None = None,
) -> list[MissingDocument]:
    company_filter = set(company_filter or [])
    doc_id_filter = set(doc_id_filter or [])
    source_docs = source_doc_lookup()
    missing: list[MissingDocument] = []

    for manifest_path in sorted(RAW_ROOT.glob("financebench_*/manifest.json")):
        company_pack = manifest_path.parent.name
        if company_filter and company_pack not in company_filter:
            continue

        manifest = json.loads(manifest_path.read_text())
        for document in manifest["documents"]:
            doc_id = document["doc_id"]
            if doc_id_filter and doc_id not in doc_id_filter:
                continue
            if any(path.exists() for path in existing_paths(document["path"])):
                continue
            source_row = source_docs.get(doc_id, {})
            missing.append(
                MissingDocument(
                    company_pack=company_pack,
                    doc_id=doc_id,
                    title=document["title"],
                    source_type=document["source_type"],
                    period=document["period"],
                    issuer=document.get("issuer", company_pack),
                    target_path=document["path"],
                    original_url=document["url"],
                    doc_name=source_row.get("doc_name"),
                )
            )
    return missing


def financebench_mirror_url(doc_name: str | None) -> str | None:
    if not doc_name:
        return None
    return f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def quarter_end(year: int, quarter: int) -> date:
    month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = month_day[quarter]
    return date(year, month, day)


def quarter_for_date(value: date) -> int:
    return ((value.month - 1) // 3) + 1


def filing_targets(doc: MissingDocument) -> tuple[list[str], date | None]:
    period = doc.period
    if doc.source_type == "annual_report":
        return ["10-K", "10-K/A"], parse_date(f"{period[:4]}-12-31")
    if doc.source_type == "quarterly_report":
        match = re.fullmatch(r"(\d{4})Q([1-4])", period)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            return ["10-Q", "10-Q/A"], quarter_end(year, quarter)
        return ["10-Q", "10-Q/A"], None
    if doc.source_type in {"earnings_material", "quarterly_results"}:
        dated = re.search(r"dated[_-](\d{4})[_-](\d{2})[_-](\d{2})", doc.doc_id)
        if dated:
            return ["8-K", "8-K/A"], date(int(dated.group(1)), int(dated.group(2)), int(dated.group(3)))
        qmatch = re.fullmatch(r"(\d{4})Q([1-4])", period)
        if qmatch:
            year = int(qmatch.group(1))
            quarter = int(qmatch.group(2))
            # Earnings releases typically land shortly after quarter end.
            offset = 45 if quarter == 4 else 30
            return ["8-K", "8-K/A"], quarter_end(year, quarter) + timedelta(days=offset)
        return ["8-K", "8-K/A"], None
    return [], None


def company_cik(doc: MissingDocument) -> str | None:
    if doc.company_pack in SEC_CIKS:
        return SEC_CIKS[doc.company_pack]
    match = re.search(r"CIK-(\d+)", doc.original_url)
    if match:
        return match.group(1).zfill(10)
    parsed = urlparse(doc.original_url)
    if parsed.netloc == "quotes.quotemedia.com":
        qmatch = re.search(r"cik=(\d+)", parsed.query)
        if qmatch:
            return qmatch.group(1).zfill(10)
    return None


def load_sec_submissions(session: requests.Session, cik: str) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = session.get(url, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
    response.raise_for_status()
    payload = response.json()
    filings: list[dict] = []

    def rows_from(filings_payload: dict) -> list[dict]:
        recent = filings_payload.get("recent", {})
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        report_dates = recent.get("reportDate", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_desc = recent.get("primaryDocDescription", [])
        rows = []
        for idx, accession in enumerate(accessions):
            rows.append(
                {
                    "accessionNumber": accession,
                    "form": forms[idx] if idx < len(forms) else None,
                    "reportDate": report_dates[idx] if idx < len(report_dates) else None,
                    "filingDate": filing_dates[idx] if idx < len(filing_dates) else None,
                    "primaryDocument": primary_docs[idx] if idx < len(primary_docs) else None,
                    "primaryDocDescription": primary_desc[idx] if idx < len(primary_desc) else None,
                }
            )
        return rows

    filings.extend(rows_from(payload.get("filings", {})))

    for older in payload.get("filings", {}).get("files", []):
        name = older.get("name")
        if not name:
            continue
        older_url = f"https://data.sec.gov/submissions/{name}"
        older_response = session.get(older_url, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
        older_response.raise_for_status()
        filings.extend(rows_from(older_response.json()))

    return filings


def filing_score(doc: MissingDocument, filing: dict) -> float:
    forms, target_date = filing_targets(doc)
    if forms and filing.get("form") not in forms:
        return -1e9

    filing_date = parse_date(filing.get("filingDate"))
    report_date = parse_date(filing.get("reportDate"))
    score = 0.0

    if report_date and target_date:
        if doc.source_type == "annual_report":
            if report_date.year == target_date.year:
                score += 100
        elif doc.source_type == "quarterly_report":
            if report_date.year == target_date.year and quarter_for_date(report_date) == quarter_for_date(target_date):
                score += 100
        else:
            # Earnings / results often reference the prior quarter while filing in the next one.
            report_gap = abs((report_date - target_date).days)
            score += max(0, 40 - min(report_gap, 40))

    if filing_date and target_date:
        filing_gap = abs((filing_date - target_date).days)
        score += max(0, 60 - min(filing_gap, 60))

    if filing.get("primaryDocument", "").lower().endswith((".htm", ".html", ".pdf")):
        score += 5
    return score


def rank_index_entry(doc: MissingDocument, filing: dict, item_name: str) -> float:
    name = item_name.lower()
    score = 0.0
    primary = (filing.get("primaryDocument") or "").lower()
    if name == primary:
        score += 30 if doc.source_type in {"earnings_material", "quarterly_results"} else 50
    if name.endswith(".pdf"):
        score += 25
    elif name.endswith((".htm", ".html")):
        score += 15

    if doc.source_type in {"annual_report", "quarterly_report"}:
        if "10-k" in name or "10k" in name:
            score += 20
        if "10-q" in name or "10q" in name:
            score += 20
        if "annual" in name or "quarter" in name or "report" in name:
            score += 8
    else:
        for token in ("99", "99.1", "ex99", "press", "release", "earnings", "results", "exhibit"):
            if token in name:
                score += 12
        if "99" in name and name.endswith(".pdf"):
            score += 18
    return score


def sec_fallback_candidates(session: requests.Session, doc: MissingDocument) -> list[Candidate]:
    cik = company_cik(doc)
    if not cik:
        return []

    try:
        filings = load_sec_submissions(session, cik)
    except Exception:
        return []

    ranked = sorted(filings, key=lambda filing: filing_score(doc, filing), reverse=True)
    top = [filing for filing in ranked[:6] if filing_score(doc, filing) > -1e8]
    candidates: list[Candidate] = []
    cik_no_pad = str(int(cik))

    for filing in top:
        accession = filing.get("accessionNumber")
        if not accession:
            continue
        accession_no_dash = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_pad}/{accession_no_dash}/index.json"
        try:
            index_response = session.get(index_url, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
            index_response.raise_for_status()
            index_payload = index_response.json()
        except Exception:
            continue

        items = index_payload.get("directory", {}).get("item", [])
        if not items:
            continue
        ranked_items = sorted(
            (item["name"] for item in items if item.get("name")),
            key=lambda name: rank_index_entry(doc, filing, name),
            reverse=True,
        )
        for item_name in ranked_items[:3]:
            candidates.append(
                Candidate(
                    label=f"sec:{filing.get('form')}:{accession}",
                    url=f"https://www.sec.gov/Archives/edgar/data/{cik_no_pad}/{accession_no_dash}/{item_name}",
                    filing_accession=accession,
                )
            )
    return candidates


def candidate_urls(session: requests.Session, doc: MissingDocument) -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []

    def add(label: str, url: str | None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        candidates.append(Candidate(label=label, url=url))

    add("preferred_direct", PREFERRED_DIRECT_URLS.get(doc.doc_id))
    add("original", doc.original_url)
    add("financebench_mirror", financebench_mirror_url(doc.doc_name))
    for sec_candidate in sec_fallback_candidates(session, doc):
        if sec_candidate.url in seen:
            continue
        seen.add(sec_candidate.url)
        candidates.append(sec_candidate)
    return candidates


def extension_from_response(final_url: str, content_type: str) -> str:
    lowered_type = (content_type or "").lower()
    lowered_url = final_url.lower()
    if "pdf" in lowered_type or lowered_url.endswith(".pdf"):
        return ".pdf"
    if "html" in lowered_type or lowered_url.endswith((".htm", ".html")):
        return ".html"
    return ".pdf"


def rewrite_manifest_path(company_pack: str, doc_id: str, new_relative_path: str) -> None:
    manifest_path = RAW_ROOT / company_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    changed = False
    for document in manifest["documents"]:
        if document["doc_id"] == doc_id and document["path"] != new_relative_path:
            document["path"] = new_relative_path
            changed = True
    if changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = read_jsonl(SOURCE_DOC_INDEX)
    updated = False
    for row in rows:
        if row["doc_id"] == doc_id and row["path"] != new_relative_path:
            row["path"] = new_relative_path
            updated = True
    if updated:
        with SOURCE_DOC_INDEX.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_response(
    company_pack: str,
    doc: MissingDocument,
    response: requests.Response,
    final_url: str,
) -> tuple[Path, int, str]:
    preferred_target = (REPO_ROOT / doc.target_path).resolve()
    actual_suffix = extension_from_response(final_url, response.headers.get("Content-Type", ""))
    destination = preferred_target if preferred_target.suffix.lower() == actual_suffix else preferred_target.with_suffix(actual_suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    bytes_written = 0
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            bytes_written += len(chunk)

    if bytes_written == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Downloaded file is empty")

    temp_path.replace(destination)
    relative_path = destination.relative_to(REPO_ROOT).as_posix()
    rewrite_manifest_path(company_pack, doc.doc_id, relative_path)
    return destination, bytes_written, file_sha256(destination)


def download_missing_doc(session: requests.Session, doc: MissingDocument) -> DownloadResult:
    started_at = now_iso()
    start_ts = time.perf_counter()
    last_error = ""

    for candidate in candidate_urls(session, doc):
        try:
            with session.get(candidate.url, stream=True, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL) as response:
                http_status = response.status_code
                response.raise_for_status()
                final_url = str(response.url)
                destination, bytes_written, sha256 = save_response(doc.company_pack, doc, response, final_url)
                append_receipt(
                    doc.company_pack,
                    {
                        "doc_id": doc.doc_id,
                        "path": destination.relative_to(REPO_ROOT).as_posix(),
                        "source_url": doc.original_url,
                        "effective_url": final_url,
                        "content_type": response.headers.get("Content-Type"),
                        "http_code": http_status,
                        "size_download": bytes_written,
                        "bytes_local": destination.stat().st_size,
                        "sha256": sha256,
                        "downloaded_at": now_iso(),
                        "resolver": candidate.label,
                    },
                )
                finished_at = now_iso()
                return DownloadResult(
                    doc_id=doc.doc_id,
                    company_pack=doc.company_pack,
                    source_type=doc.source_type,
                    period=doc.period,
                    status="success",
                    selected_label=candidate.label,
                    selected_url=final_url,
                    target_path=destination.as_posix(),
                    http_status=http_status,
                    bytes_written=bytes_written,
                    sha256=sha256,
                    error="",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_sec=round(time.perf_counter() - start_ts, 3),
                )
        except Exception as exc:  # pragma: no cover - exercised via script runs
            last_error = f"{candidate.label}: {exc!r}"

    finished_at = now_iso()
    return DownloadResult(
        doc_id=doc.doc_id,
        company_pack=doc.company_pack,
        source_type=doc.source_type,
        period=doc.period,
        status="failed",
        selected_label="",
        selected_url="",
        target_path=doc.target_path,
        http_status=None,
        bytes_written=0,
        sha256="",
        error=last_error or "No candidate succeeded",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=round(time.perf_counter() - start_ts, 3),
    )


def write_csv(results: list[DownloadResult], path: Path) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_json(results: list[DownloadResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in results], handle, ensure_ascii=False, indent=2)


def print_summary(results: list[DownloadResult]) -> None:
    success = sum(1 for row in results if row.status == "success")
    failed = sum(1 for row in results if row.status == "failed")
    print("\n===== Download Summary =====")
    print(f"Total   : {len(results)}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    if failed:
        print("\nFailed items:")
        for row in results:
            if row.status == "failed":
                print(f"- {row.doc_id}: {row.error}")
    print(f"\nCSV log : {CSV_LOG_PATH}")
    print(f"JSON log: {JSON_LOG_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-download unresolved FinanceBench source documents.")
    parser.add_argument("--company", action="append", help="Optional financebench_* pack to restrict.")
    parser.add_argument("--doc-id", action="append", help="Optional doc_id to restrict.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    missing = iter_missing_documents(company_filter=args.company, doc_id_filter=args.doc_id)
    print(f"Preparing to process {len(missing)} missing files...")
    session = build_session()
    results: list[DownloadResult] = []

    for idx, doc in enumerate(missing, start=1):
        print(f"[{idx:02d}/{len(missing)}] {doc.doc_id}")
        result = download_missing_doc(session, doc)
        results.append(result)
        print(
            f"    -> {result.status}"
            + (f" | via={result.selected_label}" if result.selected_label else "")
            + (f" | bytes={result.bytes_written}" if result.bytes_written else "")
        )

    write_csv(results, CSV_LOG_PATH)
    write_json(results, JSON_LOG_PATH)
    print_summary(results)


if __name__ == "__main__":
    main()
