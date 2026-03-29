"""
数据入库：读取 company pack → 分 chunk → 存储

运行方式：
    python -m retrieval.ingest --company stripe
    python -m retrieval.ingest --all
"""

import argparse
import json
from typing import List

from agent.config import settings
from agent.schemas import TextChunk
from tools.doc_parser import load_company_pack
from retrieval.chunking import TextChunker


def ingest_company(company_name: str) -> List[TextChunk]:
    company_dir = settings.company_packs_dir / company_name
    if not company_dir.exists():
        raise FileNotFoundError(f"Company pack not found: {company_dir}")

    # 1. 加载原始文档
    documents = load_company_pack(company_dir)
    print(f"Loaded {len(documents)} documents for {company_name}")

    # 2. 分 chunk
    chunker = TextChunker()
    all_chunks = []
    all_metas = []

    for meta, text in documents:
        chunks = chunker.chunk_document(meta, text)
        all_chunks.extend(chunks)
        all_metas.append(meta)
        print(f"  {meta.source_id}: {len(chunks)} chunks")

    # 3. 保存到 JSON（MVP 用文件存储，后续可以换数据库）
    output_dir = settings.data_dir / "processed" / company_name
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_id": c.source_id,
            "page": c.page,
            "token_count": c.token_count,
            "company": next((m.company for m in all_metas if m.source_id == c.source_id), company_name),
            "doc_id": c.source_id,
            "source_type": next((m.source_type for m in all_metas if m.source_id == c.source_id), None),
            "period": next((m.period for m in all_metas if m.source_id == c.source_id), None),
            "title": next((m.title for m in all_metas if m.source_id == c.source_id), None),
            "is_primary": next((m.is_primary for m in all_metas if m.source_id == c.source_id), None),
        }
        for c in all_chunks
    ]
    with open(output_dir / "chunks.json", "w") as f:
        json.dump(chunks_data, f, indent=2)

    metas_data = [
        {
            "source_id": m.source_id,
            "source_type": m.source_type,
            "title": m.title,
            "url": m.url,
            "date": m.date,
            "company": m.company,
            "period": m.period,
            "is_primary": m.is_primary,
        }
        for m in all_metas
    ]
    with open(output_dir / "sources.json", "w") as f:
        json.dump(metas_data, f, indent=2)

    print(f"Total: {len(all_chunks)} chunks saved to {output_dir}")
    return all_chunks


def ingest_all():
    for company_dir in sorted(settings.company_packs_dir.iterdir()):
        if company_dir.is_dir():
            try:
                ingest_company(company_dir.name)
            except Exception as e:
                print(f"Error ingesting {company_dir.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        ingest_all()
    elif args.company:
        ingest_company(args.company)
    else:
        print("Usage: --company <name> or --all")
