"""
Normalize raw benchmark source packs into document- and chunk-level corpora.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import lxml.html
import pdfplumber

from agent.config import settings
from agent.schemas import DocumentMeta
from retrieval.chunking import TextChunker


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
DROP_TAGS = {"script", "style", "noscript", "svg", "iframe"}


def load_manifest(company: str) -> dict:
    manifest_path = settings.data_dir / "raw" / company / "manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html_document(path: Path) -> str:
    raw = path.read_bytes()
    document = lxml.html.fromstring(raw)

    for node in document.xpath("//comment()"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    for element in list(document.iter()):
        tag_name = getattr(element, "tag", None)
        if not isinstance(tag_name, str):
            continue
        local_name = tag_name.split("}", 1)[-1].lower()
        style = (element.attrib.get("style") or "").lower().replace(" ", "")
        css_class = (element.attrib.get("class") or "").lower()
        element_id = (element.attrib.get("id") or "").lower()

        should_drop = (
            local_name in DROP_TAGS
            or local_name in {"header"}
            or local_name.startswith("ix:")
            or local_name in {"hidden", "header"}
            or "display:none" in style
            or "visibility:hidden" in style
            or "aspnethidden" in css_class
            or "__viewstate" in element_id
        )
        if should_drop:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    body = document.find("body")
    root = body if body is not None else document
    paragraphs = []
    seen = set()

    for element in root.iter():
        tag_name = getattr(element, "tag", None)
        if not isinstance(tag_name, str):
            continue
        local_name = tag_name.split("}", 1)[-1].lower()
        if local_name not in BLOCK_TAGS:
            continue
        text = normalize_whitespace(element.text_content())
        if len(text) < 40:
            continue
        if text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)

    if not paragraphs:
        fallback = normalize_whitespace(root.text_content())
        return fallback

    return "\n\n".join(paragraphs)


def parse_pdf_document(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = normalize_whitespace(page.extract_text() or "")
            if text:
                pages.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(pages)


def parse_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return parse_html_document(path)
    if suffix == ".pdf":
        return parse_pdf_document(path)
    return normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))


def build_meta(company: str, doc: dict) -> DocumentMeta:
    return DocumentMeta(
        source_id=doc["doc_id"],
        source_type=doc["source_type"],
        title=doc["title"],
        url=doc.get("url"),
        date=doc.get("doc_date"),
        company=company,
        period=doc.get("period"),
        published_at=doc.get("published_at"),
        issuer=doc.get("issuer"),
        is_primary=doc.get("is_primary", True),
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_company(company: str) -> dict:
    manifest = load_manifest(company)
    chunker = TextChunker()

    documents_out = []
    chunks_out = []
    processed_chunks = []
    processed_sources = []

    for doc in manifest["documents"]:
        path = Path(doc["path"])
        absolute_path = settings.data_dir.parent / path
        if not absolute_path.exists():
            print(f"skip missing {doc['doc_id']} -> {absolute_path}")
            continue

        meta = build_meta(company, doc)
        text = parse_document(absolute_path)
        if not text:
            print(f"skip empty {doc['doc_id']}")
            continue

        documents_out.append(
            {
                "company": company,
                "doc_id": doc["doc_id"],
                "path": doc["path"],
                "title": doc["title"],
                "source_type": doc["source_type"],
                "period": doc["period"],
                "doc_date": doc.get("doc_date"),
                "published_at": doc.get("published_at"),
                "issuer": doc.get("issuer"),
                "is_primary": doc.get("is_primary", True),
                "url": doc.get("url"),
                "language": doc.get("language", "en"),
                "text": text,
            }
        )

        processed_sources.append(
            {
                "source_id": meta.source_id,
                "doc_id": doc["doc_id"],
                "source_type": meta.source_type,
                "title": meta.title,
                "url": meta.url,
                "date": meta.date,
                "company": meta.company,
                "period": doc["period"],
                "is_primary": doc.get("is_primary", True),
            }
        )

        chunks = chunker.chunk_document(meta, text)
        for index, chunk in enumerate(chunks):
            chunk_row = {
                "chunk_id": chunk.chunk_id,
                "chunk_index": index,
                "company": company,
                "doc_id": doc["doc_id"],
                "source_id": chunk.source_id,
                "source_type": doc["source_type"],
                "period": doc["period"],
                "title": doc["title"],
                "page": chunk.page,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            chunks_out.append(chunk_row)
            processed_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_id": chunk.source_id,
                    "doc_id": doc["doc_id"],
                    "company": company,
                    "source_type": doc["source_type"],
                    "period": doc["period"],
                    "title": doc["title"],
                    "page": chunk.page,
                    "token_count": chunk.token_count,
                    "is_primary": doc.get("is_primary", True),
                }
            )

    normalized_dir = settings.data_dir / "normalized" / company
    processed_dir = settings.data_dir / "processed" / company
    normalized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(normalized_dir / "documents.jsonl", documents_out)
    write_jsonl(normalized_dir / "chunks.jsonl", chunks_out)

    with open(processed_dir / "chunks.json", "w", encoding="utf-8") as handle:
        json.dump(processed_chunks, handle, indent=2, ensure_ascii=False)
    with open(processed_dir / "sources.json", "w", encoding="utf-8") as handle:
        json.dump(processed_sources, handle, indent=2, ensure_ascii=False)

    summary = {
        "company": company,
        "documents": len(documents_out),
        "chunks": len(chunks_out),
        "normalized_dir": str(normalized_dir),
        "processed_dir": str(processed_dir),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize benchmark raw corpus into documents/chunks.")
    parser.add_argument("--company", action="append", required=True, help="Company to normalize.")
    args = parser.parse_args()

    for company in args.company:
        normalize_company(company)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
