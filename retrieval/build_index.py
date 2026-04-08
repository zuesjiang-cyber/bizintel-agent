"""
构建检索索引

运行方式：
    python -m retrieval.build_index --company stripe
    python -m retrieval.build_index --all
"""

import argparse
import json

from agent.config import settings
from retrieval.hybrid_retriever import HybridRetriever


def load_processed_company_chunks(company_name: str) -> list[dict]:
    processed_dir = settings.data_dir / "processed" / company_name
    chunks_path = processed_dir / "chunks.json"
    if not chunks_path.exists():
        return []

    with open(chunks_path, encoding="utf-8") as handle:
        chunks = json.load(handle)

    sources_path = processed_dir / "sources.json"
    source_meta_by_id = {}
    if sources_path.exists():
        with open(sources_path, encoding="utf-8") as handle:
            source_meta_by_id = {
                row["source_id"]: row
                for row in json.load(handle)
            }

    enriched = []
    for chunk in chunks:
        merged = dict(chunk)
        meta = source_meta_by_id.get(chunk.get("source_id"), {})
        merged.setdefault("company", meta.get("company", company_name))
        merged.setdefault("doc_id", meta.get("doc_id") or chunk.get("source_id"))
        merged.setdefault("source_type", meta.get("source_type"))
        merged.setdefault("period", meta.get("period"))
        merged.setdefault("title", meta.get("title"))
        if "is_primary" not in merged:
            merged["is_primary"] = meta.get("is_primary")
        enriched.append(merged)
    return enriched


def build_index(company_name: str = None):
    retriever = HybridRetriever(
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )

    all_chunks = []
    processed_dir = settings.data_dir / "processed"
    if not processed_dir.exists():
        print(f"Processed data directory not found: {processed_dir}. Run ingest first.")
        return

    if company_name:
        companies = [company_name]
    else:
        companies = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    for company in companies:
        chunks = load_processed_company_chunks(company)
        if chunks:
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {company}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks found. Nothing to index.")
        return

    retriever.index(all_chunks)

    index_dir = settings.data_dir / "index"
    retriever.save_index(index_dir)
    print(f"Index saved to {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        build_index()
    elif args.company:
        build_index(args.company)
