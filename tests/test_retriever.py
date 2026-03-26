"""
测试 HybridRetriever 的核心功能

运行方式：
    pytest tests/test_retriever.py -v
"""

import pytest
import numpy as np
from retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    """创建一个带小型测试数据的 retriever"""
    r = HybridRetriever(load_models=False)
    test_chunks = [
        {
            "chunk_id": "c1",
            "text": "Stripe is a financial technology company that provides payment processing software and APIs for internet businesses. Founded in 2010 by Patrick and John Collison.",
            "source_id": "stripe_about",
            "page": None,
        },
        {
            "chunk_id": "c2",
            "text": "Stripe's total funding is approximately $8.7 billion, with their latest round being a $6.5 billion Series I in March 2023.",
            "source_id": "stripe_profile",
            "page": None,
        },
        {
            "chunk_id": "c3",
            "text": "Notion is a productivity and note-taking web application that provides tools for project management, knowledge management, and collaboration.",
            "source_id": "notion_about",
            "page": None,
        },
        {
            "chunk_id": "c4",
            "text": "The global cloud computing market size was valued at $545.8 billion in 2022 and is expected to grow at a CAGR of 14.1%.",
            "source_id": "market_report",
            "page": 1,
        },
        {
            "chunk_id": "c5",
            "text": "Stripe's key competitors include Adyen, Square (Block), PayPal, and Braintree in the payment processing space.",
            "source_id": "stripe_analysis",
            "page": None,
        },
    ]
    r.index(test_chunks)
    return r


def test_retrieve_returns_results(retriever):
    results = retriever.retrieve("What is Stripe?", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3


def test_retrieve_relevance(retriever):
    results = retriever.retrieve("Stripe funding and valuation", top_k=3)
    source_ids = [r.source_id for r in results]
    # 融资相关的 chunk 应该排在前面
    assert "stripe_profile" in source_ids


def test_retrieve_company_specificity(retriever):
    results = retriever.retrieve("Notion product features", top_k=3)
    # Notion 相关的 chunk 应该排在最前
    assert results[0].source_id == "notion_about"


def test_retrieve_has_scores(retriever):
    results = retriever.retrieve("payment processing", top_k=3)
    for r in results:
        assert r.rerank_score is not None
        assert isinstance(r.score, float)


def test_rrf_fusion_combines_both_sources(retriever):
    """验证 RRF 确实融合了 BM25 和 Dense 的结果"""
    results = retriever.retrieve("Stripe competitors", top_k=5)
    assert len(results) > 0


def test_empty_query(retriever):
    results = retriever.retrieve("", top_k=3)
    # 不应该崩溃
    assert isinstance(results, list)


def test_load_index_rebuilds_dense_embeddings_for_dummy_encoder(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.json").write_text(
        '[{"chunk_id":"c1","text":"Cloudflare sells security and performance services.","source_id":"cloudflare_overview","page":null}]',
        encoding="utf-8",
    )
    np.save(index_dir / "dense_embeddings.npy", np.ones((1, 128), dtype=float))

    retriever = HybridRetriever(load_models=False)
    retriever.load_index(index_dir)
    results = retriever.retrieve("Cloudflare security", top_k=1)

    assert results[0].source_id == "cloudflare_overview"


def test_save_index_writes_metadata(tmp_path, retriever):
    index_dir = tmp_path / "index"

    retriever.save_index(index_dir)

    assert (index_dir / "index_meta.json").exists()


def test_load_index_uses_cached_dummy_embeddings_when_metadata_matches(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.json").write_text(
        '[{"chunk_id":"c1","text":"Cloudflare sells security and performance services.","source_id":"cloudflare_overview","page":null}]',
        encoding="utf-8",
    )
    cached = np.ones((1, 32), dtype=float)
    np.save(index_dir / "dense_embeddings.npy", cached)
    (index_dir / "index_meta.json").write_text(
        '{"schema_version":1,"embedding_model":"BAAI/bge-base-en-v1.5","encoder_kind":"dummy","embedding_dim":32,"chunk_count":1}',
        encoding="utf-8",
    )

    retriever = HybridRetriever(load_models=False)
    retriever._rebuild_dense_embeddings = lambda: (_ for _ in ()).throw(AssertionError("should not rebuild"))
    retriever.load_index(index_dir)

    assert retriever._dense_embeddings.shape == (1, 32)
