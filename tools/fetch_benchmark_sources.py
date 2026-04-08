"""
Download raw benchmark source-pack files from per-company manifests.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
USER_AGENT = "BizIntelBenchmarkFetcher/1.0 (+https://example.invalid)"


def load_manifest(company: str) -> dict:
    manifest_path = DATA_ROOT / company / "manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def select_documents(manifest: dict, requested_doc_ids: Iterable[str]) -> List[dict]:
    requested = set(requested_doc_ids)
    documents = manifest["documents"]
    if not requested:
        return documents
    return [doc for doc in documents if doc["doc_id"] in requested]


def resolve_destination(path_value: str, company: str) -> Path:
    destination = (REPO_ROOT / path_value).resolve()
    company_docs_root = (REPO_ROOT / "data" / "raw" / company / "docs").resolve()
    if company_docs_root not in destination.parents:
        raise ValueError(f"Refusing to write outside {company_docs_root}: {destination}")
    return destination


def receipt_path(company: str) -> Path:
    return DATA_ROOT / company / "download_receipts.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_receipt(company: str, payload: dict) -> None:
    target = receipt_path(company)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def financebench_github_api_url(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if parsed.netloc != "raw.githubusercontent.com":
        return None
    path = parsed.path.lstrip("/")
    prefix = "patronus-ai/financebench/main/"
    if not path.startswith(prefix):
        return None
    relative_path = path[len(prefix):]
    return f"https://api.github.com/repos/patronus-ai/financebench/contents/{relative_path}?ref=main"


def download_document(company: str, document: dict, force: bool = False, dry_run: bool = False) -> str:
    destination = resolve_destination(document["path"], company)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.part")

    if destination.exists() and not force:
        return f"skip {document['doc_id']} -> {destination} (already exists)"

    if dry_run:
        return f"plan {document['doc_id']} -> {destination} <= {document['url']}"

    command = [
        "curl",
        "-fsSL",
        "--http1.1",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        "-A",
        USER_AGENT,
        "-w",
        '{"content_type":"%{content_type}","url_effective":"%{url_effective}","http_code":%{http_code},"size_download":%{size_download}}\n',
        "-o",
        str(temp_path),
        document["url"],
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        temp_path.replace(destination)
    except subprocess.CalledProcessError:
        if temp_path.exists():
            temp_path.unlink()

        fallback_url = financebench_github_api_url(document["url"])
        if not fallback_url:
            raise

        fallback_command = [
            "curl",
            "-fsSL",
            "--http1.1",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "-H",
            "Accept: application/vnd.github.raw",
            "-A",
            USER_AGENT,
            "-w",
            '{"content_type":"%{content_type}","url_effective":"%{url_effective}","http_code":%{http_code},"size_download":%{size_download}}\n',
            "-o",
            str(temp_path),
            fallback_url,
        ]
        result = subprocess.run(fallback_command, check=True, capture_output=True, text=True)
        temp_path.replace(destination)

    metadata = json.loads(result.stdout.strip() or "{}")
    append_receipt(
        company,
        {
            "doc_id": document["doc_id"],
            "path": document["path"],
            "source_url": document["url"],
            "effective_url": metadata.get("url_effective"),
            "content_type": metadata.get("content_type"),
            "http_code": metadata.get("http_code"),
            "size_download": metadata.get("size_download"),
            "bytes_local": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return f"saved {document['doc_id']} -> {destination}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download benchmark source-pack files.")
    parser.add_argument("--company", action="append", required=True, help="Company manifest to use.")
    parser.add_argument("--doc-id", action="append", default=[], help="Optional document ids to restrict download.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    for company in args.company:
        manifest = load_manifest(company)
        documents = select_documents(manifest, args.doc_id)
        if not documents:
            parser.error(f"No documents matched for company={company!r} doc_ids={args.doc_id!r}")

        print(f"[{company}] {len(documents)} document(s)")
        failed_primary = False
        for document in documents:
            try:
                print(download_document(company, document, force=args.force, dry_run=args.dry_run))
            except subprocess.CalledProcessError as exc:
                print(f"error {document['doc_id']} -> {exc}")
                if document.get("is_primary", True):
                    failed_primary = True

        if failed_primary:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
