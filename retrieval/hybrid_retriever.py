"""
混合检索引擎：BM25 + Dense Retrieval + RRF Fusion + Cross-encoder Reranker

为什么自己写而不用框架：
- 展示对检索核心算法的理解
- 可以精确控制每个阶段的参数和行为
- 面试时能讲清楚每个设计决策
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # Allows offline tests with dummy models
    CrossEncoder = None
    SentenceTransformer = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight environments
    torch = None

from agent.config import settings
from agent.schemas import RetrievedChunk

logger = logging.getLogger(__name__)
_ENCODER_CACHE: Dict[Tuple[str, str], object] = {}
_RERANKER_CACHE: Dict[Tuple[str, str], object] = {}
SearchHit = Tuple[int, int, float]


class _DummyEncoder:
    """Lightweight encoder for offline tests."""
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False, normalize_embeddings: bool = True, batch_size: int = 64):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for token in re.findall(r"\b\w+\b", text.lower()):
                token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[token_hash % self.dim] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=float)


class _DummyReranker:
    """Lightweight reranker for offline tests."""
    def predict(self, pairs, batch_size: int = 32):
        scores = []
        for query, text in pairs:
            q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
            t_tokens = set(re.findall(r"\b\w+\b", text.lower()))
            scores.append(float(len(q_tokens & t_tokens)))
        return np.array(scores, dtype=float)


def _resolve_inference_device() -> str:
    configured = (settings.inference_device or "auto").strip().lower()
    if configured != "auto":
        return configured
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_shared_encoder(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key not in _ENCODER_CACHE:
        logger.info("Loading embedding model on %s...", device)
        _ENCODER_CACHE[cache_key] = SentenceTransformer(model_name, device=device)
    return _ENCODER_CACHE[cache_key]


def _load_shared_reranker(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key not in _RERANKER_CACHE:
        logger.info("Loading reranker model on %s...", device)
        _RERANKER_CACHE[cache_key] = CrossEncoder(model_name, device=device)
    return _RERANKER_CACHE[cache_key]


class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
        encoder=None,
        reranker=None,
        load_models: bool = True,
    ):
        self.device = _resolve_inference_device()
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        if encoder is not None:
            self.encoder = encoder
        elif load_models:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required to load embedding models.")
            try:
                self.encoder = _load_shared_encoder(embedding_model, self.device)
            except Exception as exc:
                logger.warning("Falling back to dummy embedding encoder because model loading failed: %s", exc)
                self.encoder = _DummyEncoder()
        else:
            self.encoder = _DummyEncoder()

        if reranker is not None:
            self.reranker = reranker
        elif load_models:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers is required to load reranker models.")
            try:
                self.reranker = _load_shared_reranker(reranker_model, self.device)
            except Exception as exc:
                logger.warning("Falling back to dummy reranker because model loading failed: %s", exc)
                self.reranker = _DummyReranker()
        else:
            self.reranker = _DummyReranker()

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # 内部状态
        self._chunks: List[dict] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._dense_embeddings: Optional[np.ndarray] = None

        # 缓存每次检索的中间排名（用于结果对象和 debug）
        self._last_bm25_ranks: Dict[int, int] = {}
        self._last_dense_ranks: Dict[int, int] = {}

    def index(self, chunks: List[dict]):
        """
        构建索引

        chunks: [{"chunk_id": str, "text": str, "source_id": str, "page": int | None}, ...]
        """
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # BM25 索引
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        tokenized = [self._tokenize(t) for t in texts]
        self._bm25_index = BM25Okapi(tokenized)

        # Dense 索引
        logger.info(f"Encoding {len(chunks)} chunks for dense index...")
        self._dense_embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        logger.info("Indexing complete.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 50,
        rerank_top_n: int = 20,
        mode: str = "full_hybrid"
    ) -> List[RetrievedChunk]:
        results, _ = self.retrieve_with_trace(
            query,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            rerank_top_n=rerank_top_n,
            mode=mode,
        )
        return results

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 50,
        rerank_top_n: int = 20,
        mode: str = "full_hybrid"
    ) -> tuple[List[RetrievedChunk], dict]:
        """
        检索模式：
        - bm25_only: 只用 BM25
        - dense_only: 只用 Dense
        - hybrid_no_rerank: BM25 + Dense + RRF
        - full_hybrid: 三阶段完整管线
        """
        if not self._chunks:
            raise RuntimeError("No chunks indexed. Call index() first.")
        if self._bm25_index is None or self._dense_embeddings is None:
            raise RuntimeError("Index not built or loaded. Call index() or load_index() first.")

        bm25_results = self._bm25_search(query, top_k=candidate_pool_size)
        dense_results = self._dense_search(query, top_k=candidate_pool_size)

        self._last_bm25_ranks = {idx: rank for idx, rank, _ in bm25_results}
        self._last_dense_ranks = {idx: rank for idx, rank, _ in dense_results}
        trace = {
            "query": query,
            "mode": mode,
            "candidate_pool_size": candidate_pool_size,
            "rerank_top_n": rerank_top_n,
            "bm25_candidates": self._trace_candidates(bm25_results),
            "dense_candidates": self._trace_candidates(dense_results),
        }

        if mode == "bm25_only":
            candidates = [idx for idx, _, _ in sorted(bm25_results, key=lambda x: x[1])][:top_k]
            results = self._build_results_no_rerank(candidates, score_map={i: float(1.0/(r+1)) for i, r, _ in bm25_results})
            trace["final_results"] = self._serialize_results(results)
            return results, trace

        elif mode == "dense_only":
            candidates = [idx for idx, _, _ in sorted(dense_results, key=lambda x: x[1])][:top_k]
            results = self._build_results_no_rerank(candidates, score_map={i: float(1.0/(r+1)) for i, r, _ in dense_results})
            trace["final_results"] = self._serialize_results(results)
            return results, trace

        fused_indices = self._reciprocal_rank_fusion(bm25_results, dense_results)
        trace["fused_candidates"] = self._trace_indices(fused_indices)

        if mode == "hybrid_no_rerank":
            candidates = fused_indices[:top_k]
            # mock scores using reciprocal rank position
            score_map = {idx: float(1.0/(i+1)) for i, idx in enumerate(candidates)}
            results = self._build_results_no_rerank(candidates, score_map=score_map)
            trace["final_results"] = self._serialize_results(results)
            return results, trace

        # mode == "full_hybrid"
        candidates = fused_indices[:rerank_top_n]
        trace["rerank_input"] = self._trace_indices(candidates)
        reranked = self._rerank(query, candidates)
        trace["rerank_output"] = self._serialize_results(reranked)

        final_results = reranked[:top_k]
        trace["final_results"] = self._serialize_results(final_results)
        return final_results, trace

    def _build_results_no_rerank(self, indices: List[int], score_map: Dict[int, float]) -> List[RetrievedChunk]:
        results = []
        for idx in indices:
            chunk = self._chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source_id=chunk["source_id"],
                page=chunk.get("page"),
                score=score_map.get(idx, 0.0),
                bm25_rank=self._last_bm25_ranks.get(idx, -1),
                dense_rank=self._last_dense_ranks.get(idx, -1),
                rerank_score=score_map.get(idx, 0.0),
            ))
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[SearchHit]:
        """BM25 检索，返回 [(chunk_index, rank), ...]"""
        tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), rank, float(scores[idx])) for rank, idx in enumerate(top_indices)]

    def _dense_search(self, query: str, top_k: int) -> List[SearchHit]:
        """Dense 语义检索，返回 [(chunk_index, rank), ...]"""
        query_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        )
        similarities = (self._dense_embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), rank, float(similarities[idx])) for rank, idx in enumerate(top_indices)]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchHit],
        dense_results: List[SearchHit],
    ) -> List[int]:
        """
        Reciprocal Rank Fusion

        RRF_score(d) = Σ w_i / (k + rank_i(d))

        为什么用 RRF 而不是分数归一化后加权：
        - BM25 分数和 cosine similarity 的尺度完全不同
        - 分数归一化需要知道全局分布，在不同 query 间不可比
        - RRF 只用排名，更鲁棒
        """
        scores: Dict[int, float] = {}

        for idx, rank, _ in bm25_results:
            scores[idx] = scores.get(idx, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        for idx, rank, _ in dense_results:
            scores[idx] = scores.get(idx, 0.0) + self.dense_weight / (self.rrf_k + rank)

        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices

    def _rerank(self, query: str, candidate_indices: List[int]) -> List[RetrievedChunk]:
        """Cross-encoder 精排"""
        if not candidate_indices:
            return []

        pairs = [(query, self._chunks[idx]["text"]) for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs, batch_size=32)

        results = []
        for i, idx in enumerate(candidate_indices):
            chunk = self._chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source_id=chunk["source_id"],
                page=chunk.get("page"),
                score=float(rerank_scores[i]),
                bm25_rank=self._last_bm25_ranks.get(idx, -1),
                dense_rank=self._last_dense_ranks.get(idx, -1),
                rerank_score=float(rerank_scores[i]),
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results

    def _trace_candidates(self, results: List[SearchHit]) -> List[dict]:
        traced = []
        for idx, rank, score in results:
            chunk = self._chunks[idx]
            traced.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "page": chunk.get("page"),
                    "rank": rank,
                    "score": score,
                }
            )
        return traced

    def _trace_indices(self, indices: List[int]) -> List[dict]:
        traced = []
        for idx in indices:
            chunk = self._chunks[idx]
            traced.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "page": chunk.get("page"),
                    "bm25_rank": self._last_bm25_ranks.get(idx, -1),
                    "dense_rank": self._last_dense_ranks.get(idx, -1),
                }
            )
        return traced

    def _serialize_results(self, results: List[RetrievedChunk]) -> List[dict]:
        return [
            {
                "chunk_id": result.chunk_id,
                "source_id": result.source_id,
                "page": result.page,
                "score": result.score,
                "bm25_rank": result.bm25_rank,
                "dense_rank": result.dense_rank,
                "rerank_score": result.rerank_score,
            }
            for result in results
        ]

    def _tokenize(self, text: str) -> List[str]:
        """
        MVP 分词：小写 + 按空格/标点分割 + 去停用词

        TODO: 后续可以换 spaCy 或领域词典
        """
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'and', 'but',
                     'or', 'not', 'no', 'it', 'its', 'this', 'that'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def save_index(self, path: Path):
        """保存索引到磁盘（避免每次重建）"""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "dense_embeddings.npy", self._dense_embeddings)
        with open(path / "chunks.json", "w") as f:
            json.dump(self._chunks, f)
        with open(path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(self._build_index_metadata(), f, ensure_ascii=False, indent=2)

    def _rebuild_dense_embeddings(self):
        texts = [c["text"] for c in self._chunks]
        self._dense_embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        )

    def _encoder_kind(self) -> str:
        if isinstance(self.encoder, _DummyEncoder):
            return "dummy"
        return type(self.encoder).__name__

    def _probe_embedding_dim(self) -> int:
        if isinstance(self.encoder, _DummyEncoder):
            return int(self.encoder.dim)
        probe_embedding = self.encoder.encode(["retriever healthcheck"], normalize_embeddings=True)
        return int(probe_embedding.shape[1])

    def _build_index_metadata(self) -> dict:
        embedding_dim = None
        if self._dense_embeddings is not None and len(self._dense_embeddings.shape) == 2:
            embedding_dim = int(self._dense_embeddings.shape[1])
        return {
            "schema_version": 1,
            "embedding_model": self.embedding_model_name,
            "encoder_kind": self._encoder_kind(),
            "embedding_dim": embedding_dim,
            "chunk_count": len(self._chunks),
        }

    def _metadata_matches_runtime(self, metadata: dict, cached_dim: int, expected_rows: int) -> bool:
        if not metadata:
            return False
        if metadata.get("embedding_model") != self.embedding_model_name:
            return False
        if metadata.get("encoder_kind") != self._encoder_kind():
            return False
        if int(metadata.get("chunk_count", -1)) != expected_rows:
            return False
        stored_dim = metadata.get("embedding_dim")
        if stored_dim is None:
            return False
        return int(stored_dim) == cached_dim

    def load_index(self, path: Path):
        """从磁盘加载索引"""
        with open(path / "chunks.json") as f:
            self._chunks = json.load(f)
        tokenized = [self._tokenize(c["text"]) for c in self._chunks]
        self._bm25_index = BM25Okapi(tokenized)

        dense_path = path / "dense_embeddings.npy"
        metadata_path = path / "index_meta.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        if isinstance(self.encoder, _DummyEncoder):
            cached_embeddings = np.load(dense_path)
            expected_rows = len(self._chunks)
            if self._metadata_matches_runtime(metadata or {}, int(cached_embeddings.shape[1]), expected_rows):
                self._dense_embeddings = cached_embeddings
            else:
                logger.info("Rebuilding dense embeddings with dummy encoder for loaded index.")
                self._rebuild_dense_embeddings()
                self.save_index(path)
            logger.info(f"Loaded index: {len(self._chunks)} chunks")
            return

        self._dense_embeddings = np.load(dense_path)
        expected_rows = len(self._chunks)
        expected_dim = None
        if self._dense_embeddings.shape[0] != expected_rows:
            expected_dim = self._probe_embedding_dim()
        elif metadata and self._metadata_matches_runtime(metadata, int(self._dense_embeddings.shape[1]), expected_rows):
            expected_dim = int(self._dense_embeddings.shape[1])
        else:
            expected_dim = self._probe_embedding_dim()
        if self._dense_embeddings.shape[0] != expected_rows or self._dense_embeddings.shape[1] != expected_dim:
            logger.warning(
                "Rebuilding dense embeddings because cached index shape %s does not match current encoder output (%s, %s).",
                self._dense_embeddings.shape,
                expected_rows,
                expected_dim,
            )
            self._rebuild_dense_embeddings()
            self.save_index(path)
        elif not metadata or not self._metadata_matches_runtime(metadata, int(self._dense_embeddings.shape[1]), expected_rows):
            logger.info("Refreshing index metadata for %s.", path)
            self.save_index(path)
        logger.info(f"Loaded index: {len(self._chunks)} chunks")
