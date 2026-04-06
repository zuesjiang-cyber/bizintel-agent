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
HARD_FACT_METRICS: Dict[str, Tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "net sales", "net revenue", "top line", "营收", "收入"),
    "margin": ("margin", "gross margin", "operating margin", "毛利率", "利润率"),
    "cash_flow": ("cash flow", "free cash flow", "fcf", "现金流", "自由现金流"),
    "profitability": ("profit", "operating income", "net income", "ebitda", "eps", "盈利", "利润"),
    "guidance": ("guidance", "outlook", "forecast", "指引", "展望"),
    "valuation": ("valuation", "multiple", "ev", "估值", "倍数"),
    "funding": ("funding", "valuation", "round", "raised", "融资", "估值"),
}

SOURCE_TYPE_ALIASES: Dict[str, set[str]] = {
    "company_profile": {"company_profile", "profile", "ir_overview", "webpage"},
    "results_release": {"results_release", "quarterly_results", "financial"},
    "annual_report": {"annual_report"},
    "quarterly_report": {"quarterly_report"},
    "quarterly_results": {"quarterly_results", "results_release", "financial"},
    "earnings_call_transcript": {"earnings_call_transcript"},
    "shareholder_letter": {"shareholder_letter"},
    "investor_presentation": {"investor_presentation"},
    "investor_supplement": {"investor_supplement"},
    "proxy_statement": {"proxy_statement"},
    "event_page": {"event_page"},
    "profile": {"profile", "company_profile"},
    "webpage": {"webpage", "ir_overview"},
    "financial": {"financial", "quarterly_results", "results_release"},
}


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
                _ENCODER_CACHE[(embedding_model, self.device)] = self.encoder
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
                _RERANKER_CACHE[(reranker_model, self.device)] = self.reranker
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

    def attach_source_metadata(self, source_meta_by_id: Dict[str, dict]) -> None:
        """Backfill chunk metadata from processed source manifests."""
        if not source_meta_by_id:
            return
        for chunk in self._chunks:
            source_id = str(chunk.get("source_id", "")).strip()
            if not source_id or source_id not in source_meta_by_id:
                continue
            meta = source_meta_by_id[source_id]
            chunk.setdefault("company", meta.get("company"))
            chunk.setdefault("doc_id", meta.get("doc_id") or source_id)
            chunk.setdefault("source_type", meta.get("source_type"))
            chunk.setdefault("period", meta.get("period"))
            chunk.setdefault("title", meta.get("title"))
            if "is_primary" not in chunk:
                chunk["is_primary"] = meta.get("is_primary")

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
        mode: str = "full_hybrid",
        filters: Optional[dict] = None,
        strategy: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        results, _ = self.retrieve_with_trace(
            query,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            rerank_top_n=rerank_top_n,
            mode=mode,
            filters=filters,
            strategy=strategy,
        )
        return results

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 50,
        rerank_top_n: int = 20,
        mode: str = "full_hybrid",
        filters: Optional[dict] = None,
        strategy: Optional[str] = None,
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

        normalized_filters = self._normalize_filters(filters)
        eligible_indices = self._eligible_indices(normalized_filters)
        trace = {
            "query": query,
            "mode": mode,
            "strategy": strategy or "default",
            "candidate_pool_size": candidate_pool_size,
            "rerank_top_n": rerank_top_n,
            "filters": normalized_filters,
            "eligible_chunk_count": len(eligible_indices),
        }
        if not eligible_indices:
            trace["bm25_candidates"] = []
            trace["dense_candidates"] = []
            trace["final_results"] = []
            return [], trace

        bm25_results = self._bm25_search(query, top_k=candidate_pool_size, eligible_indices=eligible_indices)
        dense_results = self._dense_search(query, top_k=candidate_pool_size, eligible_indices=eligible_indices)

        self._last_bm25_ranks = {idx: rank for idx, rank, _ in bm25_results}
        self._last_dense_ranks = {idx: rank for idx, rank, _ in dense_results}
        trace["bm25_candidates"] = self._trace_candidates(bm25_results)
        trace["dense_candidates"] = self._trace_candidates(dense_results)

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

        candidates = fused_indices[:candidate_pool_size]
        if strategy == "hard_fact":
            ranked_scores = self._hard_fact_rank(query, candidates, normalized_filters)
            trace["hard_fact_scores"] = [
                {"chunk_id": self._chunks[idx]["chunk_id"], "score": score}
                for idx, score in ranked_scores[:rerank_top_n]
            ]
            rerank_input = [idx for idx, _ in ranked_scores[:rerank_top_n]]
            trace["rerank_input"] = self._trace_indices(rerank_input)
            reranked = self._rerank(query, rerank_input)
            rerank_map = {result.chunk_id: result.rerank_score for result in reranked}
            final_indices = [
                idx for idx, _ in sorted(
                    ranked_scores[:rerank_top_n],
                    key=lambda item: (
                        item[1],
                        rerank_map.get(self._chunks[item[0]]["chunk_id"], 0.0),
                    ),
                    reverse=True,
                )[:top_k]
            ]
            final_results = self._build_ranked_results(
                final_indices,
                score_map={idx: score for idx, score in ranked_scores},
                rerank_map=rerank_map,
            )
            trace["rerank_output"] = self._serialize_results(reranked)
            trace["final_results"] = self._serialize_results(final_results)
            return final_results, trace

        # mode == "full_hybrid"
        rerank_input = candidates[:rerank_top_n]
        trace["rerank_input"] = self._trace_indices(rerank_input)
        reranked = self._rerank(query, rerank_input)
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
                company=chunk.get("company"),
                doc_id=chunk.get("doc_id"),
                source_type=chunk.get("source_type"),
                period=chunk.get("period"),
                title=chunk.get("title"),
                is_primary=chunk.get("is_primary"),
                trust_level=chunk.get("trust_level") or self._derive_trust_level(chunk),
                content_type=chunk.get("content_type") or self._derive_content_type(chunk.get("text", "")),
                metric_signals=chunk.get("metric_signals") or self._metric_signals(chunk.get("text", "")),
            ))
        return results

    def _build_ranked_results(
        self,
        indices: List[int],
        score_map: Dict[int, float],
        rerank_map: Dict[str, float],
    ) -> List[RetrievedChunk]:
        results = []
        for idx in indices:
            chunk = self._chunks[idx]
            rerank_score = float(rerank_map.get(chunk["chunk_id"], 0.0))
            results.append(RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source_id=chunk["source_id"],
                page=chunk.get("page"),
                score=float(score_map.get(idx, 0.0)),
                bm25_rank=self._last_bm25_ranks.get(idx, -1),
                dense_rank=self._last_dense_ranks.get(idx, -1),
                rerank_score=rerank_score,
                company=chunk.get("company"),
                doc_id=chunk.get("doc_id"),
                source_type=chunk.get("source_type"),
                period=chunk.get("period"),
                title=chunk.get("title"),
                is_primary=chunk.get("is_primary"),
                trust_level=chunk.get("trust_level") or self._derive_trust_level(chunk),
                content_type=chunk.get("content_type") or self._derive_content_type(chunk.get("text", "")),
                metric_signals=chunk.get("metric_signals") or self._metric_signals(chunk.get("text", "")),
            ))
        return results

    def _bm25_search(self, query: str, top_k: int, eligible_indices: Optional[List[int]] = None) -> List[SearchHit]:
        """BM25 检索，返回 [(chunk_index, rank), ...]"""
        tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        candidate_indices = self._top_indices(scores, top_k=top_k, eligible_indices=eligible_indices)
        top_indices = candidate_indices
        return [(int(idx), rank, float(scores[idx])) for rank, idx in enumerate(top_indices)]

    def _dense_search(self, query: str, top_k: int, eligible_indices: Optional[List[int]] = None) -> List[SearchHit]:
        """Dense 语义检索，返回 [(chunk_index, rank), ...]"""
        query_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        )
        similarities = (self._dense_embeddings @ query_emb.T).flatten()
        candidate_indices = self._top_indices(similarities, top_k=top_k, eligible_indices=eligible_indices)
        top_indices = candidate_indices
        return [(int(idx), rank, float(similarities[idx])) for rank, idx in enumerate(top_indices)]

    def _top_indices(self, scores: np.ndarray, top_k: int, eligible_indices: Optional[List[int]] = None) -> List[int]:
        if eligible_indices is None:
            return [int(idx) for idx in np.argsort(scores)[::-1][:top_k]]
        if not eligible_indices:
            return []
        allowed = np.array(eligible_indices, dtype=int)
        allowed_scores = scores[allowed]
        order = np.argsort(allowed_scores)[::-1][:top_k]
        return [int(idx) for idx in allowed[order]]

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
                company=chunk.get("company"),
                doc_id=chunk.get("doc_id"),
                source_type=chunk.get("source_type"),
                period=chunk.get("period"),
                title=chunk.get("title"),
                is_primary=chunk.get("is_primary"),
                trust_level=chunk.get("trust_level") or self._derive_trust_level(chunk),
                content_type=chunk.get("content_type") or self._derive_content_type(chunk.get("text", "")),
                metric_signals=chunk.get("metric_signals") or self._metric_signals(chunk.get("text", "")),
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
                    "company": chunk.get("company"),
                    "source_type": chunk.get("source_type"),
                    "period": chunk.get("period"),
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
                    "company": chunk.get("company"),
                    "source_type": chunk.get("source_type"),
                    "period": chunk.get("period"),
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
                "company": result.company,
                "source_type": result.source_type,
                "period": result.period,
                "score": result.score,
                "bm25_rank": result.bm25_rank,
                "dense_rank": result.dense_rank,
                "rerank_score": result.rerank_score,
                "trust_level": result.trust_level,
                "content_type": result.content_type,
                "metric_signals": result.metric_signals,
            }
            for result in results
        ]

    def _hard_fact_rank(self, query: str, candidate_indices: List[int], filters: dict) -> List[Tuple[int, float]]:
        query_numbers = self._extract_numeric_tokens(query)
        query_metrics = self._metric_signals(query)
        exact_periods = set(filters.get("periods", []))
        ranked: List[Tuple[int, float]] = []
        for idx in candidate_indices:
            chunk = self._chunks[idx]
            text = chunk.get("text", "")
            chunk_period = self._normalize_period(chunk.get("period"))
            chunk_numbers = self._extract_numeric_tokens(text)
            chunk_metrics = chunk.get("metric_signals") or self._metric_signals(text)
            trust_level = chunk.get("trust_level") or self._derive_trust_level(chunk)
            content_type = chunk.get("content_type") or self._derive_content_type(text)
            score = 0.0

            if exact_periods and chunk_period in exact_periods:
                score += 4.0
            elif exact_periods:
                score -= 1.0

            if query_metrics and set(query_metrics) & set(chunk_metrics):
                score += 3.0
            if query_numbers and query_numbers & chunk_numbers:
                score += 3.0
            if content_type in {"quantitative", "mixed"}:
                score += 1.5
            if chunk.get("is_primary"):
                score += 2.0
            score += trust_level * 0.2
            bm25_rank = self._last_bm25_ranks.get(idx, 999)
            dense_rank = self._last_dense_ranks.get(idx, 999)
            score += 1.5 / (1 + max(bm25_rank, 0))
            score += 0.5 / (1 + max(dense_rank, 0))
            ranked.append((idx, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _normalize_filters(self, filters: Optional[dict]) -> dict:
        filters = filters or {}
        companies = [
            self._normalize_text(value)
            for value in filters.get("companies", [])
            if self._normalize_text(value)
        ]
        periods = [
            self._normalize_period(value)
            for value in filters.get("periods", [])
            if self._normalize_period(value)
        ]
        source_types = self._normalize_source_types(filters.get("source_types", []))
        return {
            "companies": companies,
            "periods": periods,
            "source_types": source_types,
        }

    def _eligible_indices(self, filters: dict) -> List[int]:
        companies = set(filters.get("companies", []))
        periods = set(filters.get("periods", []))
        source_types = set(filters.get("source_types", []))
        eligible = []
        for idx, chunk in enumerate(self._chunks):
            if companies:
                chunk_company = self._normalize_text(chunk.get("company"))
                if chunk_company not in companies:
                    continue
            if periods:
                chunk_period = self._normalize_period(chunk.get("period"))
                if not self._period_matches(chunk_period, periods):
                    continue
            if source_types:
                chunk_source_type = self._normalize_text(chunk.get("source_type"))
                if chunk_source_type not in source_types:
                    continue
            eligible.append(idx)
        return eligible

    def _normalize_source_types(self, values: List[str]) -> List[str]:
        normalized: set[str] = set()
        for value in values:
            key = self._normalize_text(value)
            if not key:
                continue
            normalized.update(SOURCE_TYPE_ALIASES.get(key, {key}))
        return sorted(normalized)

    def _period_matches(self, chunk_period: str, allowed_periods: set[str]) -> bool:
        if not allowed_periods:
            return True
        if not chunk_period:
            return False
        if chunk_period in allowed_periods:
            return True
        chunk_parts = self._parse_period_parts(chunk_period)
        for allowed in allowed_periods:
            if allowed == chunk_period:
                return True
            allowed_parts = self._parse_period_parts(allowed)
            if not allowed_parts:
                continue
            if allowed_parts.get("year") and allowed_parts["year"] != chunk_parts.get("year"):
                continue
            allowed_quarter = allowed_parts.get("quarter")
            if allowed_quarter and allowed_quarter != chunk_parts.get("quarter"):
                continue
            allowed_fiscal = allowed_parts.get("fiscal")
            if allowed_fiscal and allowed_fiscal != chunk_parts.get("fiscal"):
                continue
            return True
        return False

    def _parse_period_parts(self, value: str) -> dict:
        normalized = self._normalize_period(value)
        if not normalized:
            return {}
        parts: dict[str, str] = {}
        year_match = re.search(r"(20\d{2})", normalized)
        if year_match:
            parts["year"] = year_match.group(1)
        quarter_match = re.search(r"q([1-4])", normalized)
        if quarter_match:
            parts["quarter"] = quarter_match.group(1)
        if "fy" in normalized:
            parts["fiscal"] = "fy"
        return parts

    def _normalize_period(self, value: Optional[str]) -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return ""
        normalized = normalized.replace("fiscalyear", "fy")
        normalized = normalized.replace("fiscal", "fy")
        return normalized

    def _normalize_text(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip().lower().replace(" ", "").replace("-", "")

    def _extract_numeric_tokens(self, text: str) -> set[str]:
        return {
            match.group(0).lower().replace(",", "").strip()
            for match in re.finditer(
                r'\$?[\d]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k|%|bps))?',
                text or "",
                re.IGNORECASE,
            )
            if match.group(0).strip()
        }

    def _metric_signals(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        matched = []
        for metric, aliases in HARD_FACT_METRICS.items():
            if any(alias in lowered for alias in aliases):
                matched.append(metric)
        return matched

    def _derive_trust_level(self, chunk: dict) -> int:
        source_type = self._normalize_text(chunk.get("source_type"))
        if source_type in {"annual_report", "quarterly_report", "quarterly_results", "results_release"}:
            return 5
        if source_type in {"earnings_call_transcript", "shareholder_letter"}:
            return 4
        if chunk.get("is_primary"):
            return 4
        if source_type in {"investor_presentation", "investorsupplement", "profile", "company_profile", "ir_overview", "webpage"}:
            return 3
        return 2

    def _derive_content_type(self, text: str) -> str:
        tokens = self._extract_numeric_tokens(text)
        if len(tokens) >= 2:
            return "quantitative"
        if len(tokens) == 1:
            return "mixed"
        return "qualitative"

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
