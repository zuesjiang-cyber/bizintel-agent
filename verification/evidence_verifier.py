"""
验证从 memo 提取出的 claim 是否有原文支撑。
"""

import logging
import re
from typing import Dict, List, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - exercised in lightweight environments
    CrossEncoder = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight environments
    torch = None

from agent.config import settings
from agent.schemas import Claim, ConfidenceLevel, VerificationResult

logger = logging.getLogger(__name__)
_NLI_MODEL_CACHE: Dict[tuple[str, str], object] = {}
NUMERIC_TOKEN_PATTERN = re.compile(
    r'\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?%?(?![A-Za-z])',
    re.IGNORECASE,
)


def _resolve_inference_device() -> str:
    configured = (settings.inference_device or "auto").strip().lower()
    if configured != "auto":
        return configured
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _DummyNLIModel:
    def predict(self, pairs, show_progress_bar: bool = False, batch_size: int = 32):
        rows = []
        for evidence, claim in pairs:
            evidence_tokens = set(evidence.lower().split())
            claim_tokens = set(claim.lower().split())
            overlap = len(evidence_tokens & claim_tokens)
            contradiction = 0.2
            entailment = min(3.0, 0.5 + overlap / 3.0)
            neutral = 0.3
            rows.append([contradiction, entailment, neutral])
        return np.array(rows, dtype=float)


class EvidenceVerifier:
    PRIMARY_SOURCE_TYPES = {
        "annual_report",
        "quarterly_report",
        "quarterly_results",
        "earnings_call_transcript",
        "shareholder_letter",
    }
    FINANCIAL_KEYWORDS = {
        "revenue", "margin", "profit", "profitability", "cash flow", "free cash flow",
        "capex", "guidance", "debt", "eps", "ratio", "gross margin", "operating margin",
        "ebitda", "qoq", "yoy", "year-over-year", "quarter-over-quarter", "growth",
    }
    DIRECTIONALITY_TOKENS = {
        "up", "down", "increase", "increased", "decrease", "decreased", "grew", "growth",
        "decline", "declined", "expanded", "improved", "fell", "rose",
    }

    def __init__(self, use_dummy_model: bool = False):
        self._nli_model: Optional[CrossEncoder] = None
        self.use_dummy_model = use_dummy_model
        self.strong_threshold = settings.entailment_strong_threshold
        self.moderate_threshold = settings.entailment_moderate_threshold
        self.device = _resolve_inference_device()

    def _get_model(self) -> CrossEncoder:
        if self._nli_model is None:
            if self.use_dummy_model:
                self._nli_model = _DummyNLIModel()
            else:
                cache_key = (settings.nli_model, self.device)
                cached_model = _NLI_MODEL_CACHE.get(cache_key)
                if cached_model is not None:
                    self._nli_model = cached_model
            if self._nli_model is None and CrossEncoder is None:
                self._nli_model = _DummyNLIModel()
            elif self._nli_model is None:
                try:
                    logger.info("Loading NLI model for verification on %s...", self.device)
                    self._nli_model = CrossEncoder(settings.nli_model, device=self.device)
                    _NLI_MODEL_CACHE[(settings.nli_model, self.device)] = self._nli_model
                except Exception as exc:  # pragma: no cover - depends on local model availability
                    logger.warning(
                        "Falling back to dummy NLI verifier because the configured model could not be loaded: %s",
                        exc,
                    )
                    self._nli_model = _DummyNLIModel()
        return self._nli_model

    def _empty_device_cache(self) -> None:
        if torch is None:
            return
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and getattr(torch, "mps", None):
            torch.mps.empty_cache()

    def _predict_pairs(self, model: CrossEncoder, pairs_to_score: List[tuple[str, str]]) -> np.ndarray:
        batch_size = max(1, int(settings.nli_batch_size))
        while True:
            try:
                batch_scores = []
                for start in range(0, len(pairs_to_score), batch_size):
                    batch = pairs_to_score[start:start + batch_size]
                    scores = np.asarray(
                        model.predict(batch, show_progress_bar=False, batch_size=batch_size)
                    )
                    if scores.ndim == 1:
                        scores = np.expand_dims(scores, axis=0)
                    batch_scores.append(scores)
                    if self.device == "mps":
                        self._empty_device_cache()
                return np.concatenate(batch_scores, axis=0) if batch_scores else np.zeros((0, 3), dtype=float)
            except RuntimeError as exc:
                oom = "out of memory" in str(exc).lower()
                if self.device == "mps" and oom and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning("MPS OOM during NLI verification; retrying with batch size=%s", batch_size)
                    self._empty_device_cache()
                    continue
                raise

    def _normalize_evidence_chunks(self, source_id: str, chunks: List[object]) -> List[dict]:
        normalized = []
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "chunk_id": str(chunk.get("chunk_id") or f"{source_id}__{idx}"),
                        "source_id": str(chunk.get("source_id") or source_id),
                        "text": text,
                        "source_type": chunk.get("source_type"),
                        "is_primary": chunk.get("is_primary"),
                    }
                )
            else:
                text = str(chunk).strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "chunk_id": f"{source_id}__{idx}",
                        "source_id": source_id,
                        "text": text,
                        "source_type": None,
                        "is_primary": None,
                    }
                )
        return normalized

    def _select_candidate_chunks(self, claim: Claim, chunks: List[object]) -> List[dict]:
        if chunks and not isinstance(chunks[0], dict):
            chunks = self._normalize_evidence_chunks("anonymous_source", chunks)
        limit = max(1, int(settings.verification_max_chunks_per_source))
        if len(chunks) <= limit:
            return chunks

        claim_tokens = set(re.findall(r"\b\w+\b", claim.text.lower()))
        claim_numbers = {num.replace("$", "").replace(",", "").strip() for num in claim.extracted_numbers}

        scored = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_lower = chunk_text.lower()
            chunk_tokens = set(re.findall(r"\b\w+\b", chunk_lower))
            overlap = len(claim_tokens & chunk_tokens)
            numeric_hits = sum(1 for num in claim_numbers if num and num in chunk_text)
            score = (numeric_hits * 1000) + overlap
            scored.append((score, idx, chunk))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        selected = [chunk for _, _, chunk in scored[:limit]]
        if not any(score > 0 for score, _, _ in scored[:limit]):
            return chunks[:limit]
        return selected

    def verify_memo(self, claims: List[Claim], evidence_store: Dict[str, List[object]]) -> List[VerificationResult]:
        """
        验证一组 claim。
        evidence_store: source_id -> [text_chunks]
        """
        if not claims:
            return []

        results = []
        pairs_to_score = []
        claim_evidence_map = []  # 记录每个 pair 对应的 (claim_idx, evidence_chunk)

        # 1. 准备待打分的 (evidence, claim) 必须对
        # NLI 模型的输入顺序通常是 (Premise, Hypothesis) = (Evidence, Claim)
        for i, claim in enumerate(claims):
            if not claim.cited_sources or all(source_id == "no_citation" for source_id in claim.cited_sources):
                results.append(VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if claim.contains_numbers else None,
                    supporting_evidence=[],
                    explanation="Claim does not include a valid citation marker.",
                    failure_reason="missing_citation",
                ))
                continue

            has_valid_source = False
            has_available_chunks = False
            for source_id in claim.cited_sources:
                if source_id in evidence_store:
                    has_valid_source = True
                    normalized_chunks = self._normalize_evidence_chunks(source_id, evidence_store[source_id])
                    if claim.cited_chunks:
                        normalized_chunks = [
                            chunk for chunk in normalized_chunks
                            if chunk["chunk_id"] in claim.cited_chunks
                        ]
                    if normalized_chunks:
                        has_available_chunks = True
                    selected_chunks = self._select_candidate_chunks(claim, normalized_chunks)
                    for chunk in selected_chunks:
                        pairs_to_score.append((chunk["text"], claim.text))
                        claim_evidence_map.append((i, chunk))

            if not has_valid_source:
                # 没有任何有效来源，直接判无支撑
                results.append(VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if claim.contains_numbers else None,
                    supporting_evidence=[],
                    explanation="Claim cites sources that are not present in the evidence store.",
                    failure_reason="bad_source_id",
                ))
                continue

            if not has_available_chunks:
                results.append(VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if claim.contains_numbers else None,
                    supporting_evidence=[],
                    explanation="Claim cites a known source, but no evidence chunks were available for scoring.",
                    failure_reason="no_supporting_chunk",
                ))

        if not pairs_to_score:
            return results

        # 2. 批量跑 NLI 模型
        # NLI 输出通常是 3 分类：[Contradiction, Entailment, Neutral]
        # 这里用的是 cross-encoder/nli-deberta-v3-base，输出为 [Contradiction, Entailment, Neutral]
        logger.info(f"Running NLI verification for {len(pairs_to_score)} pairs...")
        model = self._get_model()
        scores = self._predict_pairs(model, pairs_to_score)

        # 提取 Entailment (蕴含) 的分数。对于 deberta-v3-base，Entailment 在 index 1
        # 但我们为了稳妥，先转成 prob，取 index 1
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        entailment_scores = probs[:, 1]

        # 3. 聚合结果
        # 一个 claim 可能对应多个 evidence chunk，取最高分
        best_scores = {i: 0.0 for i in range(len(claims))}
        best_evidence = {i: None for i in range(len(claims))}

        for pair_idx, (claim_idx, chunk) in enumerate(claim_evidence_map):
            score = float(entailment_scores[pair_idx])
            if score > best_scores[claim_idx]:
                best_scores[claim_idx] = score
                best_evidence[claim_idx] = chunk

        # 4. 生成最终判定
        # 需要注意的是，我们之前可能已经提前把没有 source 的 claim 放进了 results。
        # 为了保持顺序，我们用一个临时字典
        temp_results = {}
        for r in results:
            temp_results[r.claim.claim_id] = r

        for i, claim in enumerate(claims):
            if claim.claim_id in temp_results:
                continue

            score = best_scores[i]
            evidence = best_evidence[i]
            primary_source_supported = self._has_primary_source(evidence)

            # 规则 1：检查数字是否匹配
            numeric_verified = None
            if self._claim_requires_numeric_gate(claim):
                numeric_verified = self._verify_numeric_alignment(claim, evidence["text"] if evidence else "")

            # 规则 2：综合 NLI 和规则得出最终置信度
            confidence = ConfidenceLevel.WEAK
            if score >= self.strong_threshold:
                confidence = ConfidenceLevel.STRONG
            elif score >= self.moderate_threshold:
                confidence = ConfidenceLevel.MODERATE

            failure_reason = None
            # 如果数字校验失败，降级
            if numeric_verified is False:
                failure_reason = "numeric_mismatch"
                confidence = ConfidenceLevel.UNSUPPORTED
            elif self._claim_requires_primary_source(claim) and primary_source_supported is False:
                failure_reason = "primary_source_missing"
                confidence = ConfidenceLevel.UNSUPPORTED
            elif score < self.moderate_threshold:
                failure_reason = "low_entailment"

            explanation = (
                f"NLI Entailment Score: {score:.2f}. "
                f"Numeric match: {'Yes' if numeric_verified else ('No' if numeric_verified is False else 'N/A')}. "
                f"Primary source: {'Yes' if primary_source_supported else ('No' if primary_source_supported is False else 'N/A')}."
            )

            res = VerificationResult(
                claim=claim,
                confidence=confidence,
                nli_score=score,
                numeric_verified=numeric_verified,
                supporting_evidence=[evidence["text"]] if evidence else [],
                explanation=explanation,
                failure_reason=failure_reason,
                supporting_source_ids=[evidence["source_id"]] if evidence else [],
                supporting_chunk_ids=[evidence["chunk_id"]] if evidence else [],
                primary_source_supported=primary_source_supported,
            )
            temp_results[claim.claim_id] = res

        # 按原顺序返回
        final_results = []
        for claim in claims:
            final_results.append(temp_results[claim.claim_id])

        return final_results

    def score_hypothesis_against_chunks(self, hypothesis: str, chunks: List[object]) -> List[dict]:
        normalized_chunks = self._normalize_evidence_chunks("anonymous_source", chunks)
        if not normalized_chunks:
            return []

        model = self._get_model()
        pairs = [(chunk["text"], hypothesis) for chunk in normalized_chunks]
        scores = self._predict_pairs(model, pairs)
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        results = []
        for chunk, row in zip(normalized_chunks, probs):
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "text": chunk["text"],
                    "source_type": chunk.get("source_type"),
                    "is_primary": chunk.get("is_primary"),
                    "contradiction": float(row[0]),
                    "entailment": float(row[1]),
                    "neutral": float(row[2]),
                }
            )
        return results

    def _has_primary_source(self, evidence: Optional[dict]) -> Optional[bool]:
        if evidence is None:
            return None
        source_type = (evidence.get("source_type") or "").strip().lower()
        if source_type:
            return source_type in self.PRIMARY_SOURCE_TYPES
        is_primary = evidence.get("is_primary")
        if is_primary is None:
            return None
        return bool(is_primary)

    def _claim_requires_primary_source(self, claim: Claim) -> bool:
        return self._claim_requires_numeric_gate(claim)

    def _claim_requires_numeric_gate(self, claim: Claim) -> bool:
        if claim.contains_numbers:
            return True
        text = claim.text.lower()
        return any(keyword in text for keyword in self.FINANCIAL_KEYWORDS)

    def _verify_numeric_alignment(self, claim: Claim, evidence_text: str) -> bool:
        if not evidence_text:
            return False

        evidence_lower = evidence_text.lower()
        extracted_numbers = [num for num in claim.extracted_numbers if num.strip()]
        if extracted_numbers:
            for num in extracted_numbers:
                clean_num = self._normalize_numeric_token(num)
                if clean_num not in self._normalize_numeric_haystack(evidence_text):
                    return False

        for token in self._extract_period_tokens(claim.text):
            if token.lower() not in evidence_lower:
                return False

        for token in self._extract_currency_tokens(claim.text):
            if token.lower() not in evidence_lower:
                return False

        claim_directionality = self._extract_directionality_tokens(claim.text)
        evidence_directionality = self._extract_directionality_tokens(evidence_text)
        if claim_directionality and not (claim_directionality & evidence_directionality):
            return False

        return True

    def _normalize_numeric_token(self, token: str) -> str:
        return token.lower().replace("$", "").replace(",", "").strip()

    def _normalize_numeric_haystack(self, text: str) -> set[str]:
        return {
            self._normalize_numeric_token(match.group(0))
            for match in NUMERIC_TOKEN_PATTERN.finditer(text)
            if match.group(0).strip()
        }

    def _extract_period_tokens(self, text: str) -> set[str]:
        return {
            match.group(0)
            for match in re.finditer(r'\b(?:q[1-4]\s*20\d{2}|fy\s*20\d{2}|20\d{2})\b', text, re.IGNORECASE)
        }

    def _extract_currency_tokens(self, text: str) -> set[str]:
        tokens = set()
        if "$" in text:
            tokens.add("$")
        for token in re.findall(r'\b(?:usd|dollars?)\b', text, re.IGNORECASE):
            tokens.add(token.lower())
        return tokens

    def _extract_directionality_tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        return {token for token in self.DIRECTIONALITY_TOKENS if token in lowered}

    def summary_stats(self, results: List[VerificationResult]) -> dict:
        """生成验证统计信息"""
        total = len(results)
        if total == 0:
            return {
                "total_claims": 0,
                "strong": 0, "moderate": 0, "weak": 0, "unsupported": 0,
                "citation_marker_coverage": 0.0, "verified_claim_coverage": 0.0, "avg_nli_score": 0.0,
                "numeric_error_rate": 0.0, "primary_source_coverage": 0.0,
            }

        counts = {c: 0 for c in ConfidenceLevel}
        for r in results:
            counts[r.confidence] += 1

        supported = total - counts[ConfidenceLevel.UNSUPPORTED]
        cited = sum(
            1
            for r in results
            if any(source_id != "no_citation" for source_id in r.claim.cited_sources) or bool(r.claim.cited_chunks)
        )
        numeric_claims = [r for r in results if r.numeric_verified is not None]
        numeric_errors = [r for r in numeric_claims if r.numeric_verified is False]
        primary_required = [r for r in results if self._claim_requires_primary_source(r.claim)]
        primary_supported = [r for r in primary_required if r.primary_source_supported]

        return {
            "total_claims": total,
            "strong": counts[ConfidenceLevel.STRONG],
            "moderate": counts[ConfidenceLevel.MODERATE],
            "weak": counts[ConfidenceLevel.WEAK],
            "unsupported": counts[ConfidenceLevel.UNSUPPORTED],
            "citation_marker_coverage": cited / total if total > 0 else 0.0,
            "verified_claim_coverage": supported / total if total > 0 else 0.0,
            "avg_nli_score": sum(r.nli_score for r in results) / total,
            "numeric_error_rate": len(numeric_errors) / len(numeric_claims) if numeric_claims else 0.0,
            "primary_source_coverage": len(primary_supported) / len(primary_required) if primary_required else 0.0,
        }
