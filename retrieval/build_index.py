"""
构建检索索引

运行方式：
    python -m retrieval.build_index --company stripe
    python -m retrieval.build_index --all
"""

import argparse
import json
from pathlib import Path

from agent.config import settings
from retrieval.hybrid_retriever import HybridRetriever


def build_index(company_name: str = None):
    retriever = HybridRetriever(
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )

    all_chunks = []
    processed_dir = settings.data_dir / "processed"

    if company_name:
        companies = [company_name]
    else:
        companies = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    for company in companies:
        chunks_path = processed_dir / company / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {company}")

    print(f"\nTotal chunks: {len(all_chunks)}")

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
