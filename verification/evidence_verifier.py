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
TOKEN_PATTERN = re.compile(r"\b[\w$%.]+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
PASSAGE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "is",
    "what", "which", "when", "where", "into", "than", "then", "them", "they", "their",
    "have", "has", "had", "will", "would", "could", "should", "about", "into", "most",
    "important", "report", "reported", "evidence", "source", "text", "company", "quarter",
}


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

    def _extract_relevant_excerpt(self, text: str, focus_text: str, max_chars: int = 900) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if len(normalized) <= max_chars:
            return normalized

        focus_tokens = self._focus_tokens(focus_text)
        if not focus_tokens:
            return normalized[:max_chars].strip()

        segments = [seg.strip() for seg in SENTENCE_SPLIT_PATTERN.split(normalized) if seg.strip()]
        candidates: List[str] = []

        if segments:
            for idx in range(len(segments)):
                window = segments[idx]
                candidates.append(window)
                if idx + 1 < len(segments):
                    candidates.append(f"{segments[idx]} {segments[idx + 1]}")

        words = normalized.split()
        if words:
            window_size = 120
            stride = 60
            for start in range(0, len(words), stride):
                snippet = " ".join(words[start:start + window_size]).strip()
                if snippet:
                    candidates.append(snippet)

        if not candidates:
            return normalized[:max_chars].strip()

        best = max(candidates, key=lambda candidate: self._score_excerpt(candidate, focus_tokens))
        return best[:max_chars].strip()

    def _focus_tokens(self, text: str) -> List[str]:
        tokens: List[str] = []
        lowered = (text or "").lower()
        for match in NUMERIC_TOKEN_PATTERN.finditer(text or ""):
            cleaned = self._normalize_numeric_token(match.group(0))
            if cleaned:
                tokens.append(cleaned)
        for token in TOKEN_PATTERN.findall(lowered):
            clean = token.strip().lower()
            if len(clean) <= 2:
                continue
            if clean in PASSAGE_STOPWORDS:
                continue
            tokens.append(clean)
        deduped = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped[:20]

    def _score_excerpt(self, text: str, focus_tokens: List[str]) -> float:
        lowered = text.lower()
        score = 0.0
        for token in focus_tokens:
            if token in lowered:
                score += 3.0 if any(char.isdigit() for char in token) else 1.0
        word_count = max(1, len(lowered.split()))
        density_bonus = score / max(1.0, word_count / 40.0)
        length_penalty = min(1.0, word_count / 400.0)
        return score + density_bonus - length_penalty

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
            unique_cited_sources = list(dict.fromkeys(source_id for source_id in claim.cited_sources if source_id != "no_citation"))
            cited_chunk_ids = [chunk_id for chunk_id in claim.cited_chunks if chunk_id]
            requires_combined_candidate = len(list(dict.fromkeys(cited_chunk_ids))) > 1
            combined_chunks: List[dict] = []
            for source_id in unique_cited_sources:
                if source_id in evidence_store:
                    has_valid_source = True
                    normalized_chunks = self._normalize_evidence_chunks(source_id, evidence_store[source_id])
                    if claim.cited_chunks:
                        normalized_chunks = [
                            chunk for chunk in normalized_chunks
                            if chunk["chunk_id"] in claim.cited_chunks
                        ]
                    if requires_combined_candidate:
                        combined_chunks.extend(normalized_chunks)
                        continue
                    if normalized_chunks:
                        has_available_chunks = True
                    selected_chunks = self._select_candidate_chunks(claim, normalized_chunks)
                    for chunk in selected_chunks:
                        excerpt = self._extract_relevant_excerpt(chunk["text"], claim.text)
                        chunk_for_scoring = {**chunk, "excerpt_text": excerpt}
                        pairs_to_score.append((excerpt, claim.text))
                        claim_evidence_map.append((i, chunk_for_scoring))

            if requires_combined_candidate:
                ordered_chunks = self._order_chunks_for_claim(claim, combined_chunks)
                expected_chunk_ids = list(dict.fromkeys(cited_chunk_ids))
                if expected_chunk_ids and {chunk["chunk_id"] for chunk in ordered_chunks} >= set(expected_chunk_ids):
                    combined_candidate = self._combine_chunks_for_claim(claim, ordered_chunks)
                    pairs_to_score.append((combined_candidate["excerpt_text"], claim.text))
                    claim_evidence_map.append((i, combined_candidate))
                    has_available_chunks = True

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
                supporting_evidence=(evidence.get("excerpt_texts") or [evidence.get("excerpt_text") or evidence["text"]]) if evidence else [],
                explanation=explanation,
                failure_reason=failure_reason,
                supporting_source_ids=(evidence.get("source_ids") or [evidence["source_id"]]) if evidence else [],
                supporting_chunk_ids=(evidence.get("chunk_ids") or [evidence["chunk_id"]]) if evidence else [],
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
        pairs = []
        scored_chunks = []
        for chunk in normalized_chunks:
            excerpt = self._extract_relevant_excerpt(chunk["text"], hypothesis)
            scored_chunks.append({**chunk, "excerpt_text": excerpt})
            pairs.append((excerpt, hypothesis))
        scores = self._predict_pairs(model, pairs)
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        results = []
        for chunk, row in zip(scored_chunks, probs):
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "text": chunk["text"],
                    "excerpt_text": chunk.get("excerpt_text", chunk["text"]),
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
        if "is_primary" in evidence and evidence.get("is_primary") is not None:
            return bool(evidence.get("is_primary"))
        source_type = (evidence.get("source_type") or "").strip().lower()
        if source_type:
            return source_type in self.PRIMARY_SOURCE_TYPES
        return None

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
        normalized_haystack = self._normalize_numeric_haystack(evidence_text)
        missing_numbers = []
        if extracted_numbers:
            for num in extracted_numbers:
                clean_num = self._normalize_numeric_token(num)
                if clean_num not in normalized_haystack:
                    missing_numbers.append(clean_num)
            if missing_numbers and not self._supports_derived_numeric_claim(claim, evidence_text):
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

    def _supports_derived_numeric_claim(self, claim: Claim, evidence_text: str) -> bool:
        lowered = claim.text.lower()
        if "quick ratio" in lowered:
            return self._supports_quick_ratio_claim(claim, evidence_text)
        if "working capital" in lowered:
            return self._supports_working_capital_claim(claim, evidence_text)
        return False

    def _supports_quick_ratio_claim(self, claim: Claim, evidence_text: str) -> bool:
        claim_values = self._parse_claim_numeric_values(claim)
        if not claim_values:
            return False
        claimed_ratio = claim_values[0]
        current_liabilities = self._extract_labeled_numeric_value(evidence_text, ["total current liabilities"])
        if current_liabilities in (None, 0):
            return False

        current_assets = self._extract_labeled_numeric_value(evidence_text, ["total current assets"])
        inventories = self._extract_labeled_numeric_value(evidence_text, ["inventories"])
        if current_assets is not None and inventories is not None:
            computed = (current_assets - inventories) / current_liabilities
            if abs(computed - claimed_ratio) <= 0.03:
                return True

        cash = self._extract_labeled_numeric_value(evidence_text, ["cash and cash equivalents"])
        short_term_investments = self._extract_labeled_numeric_value(evidence_text, ["short-term investments"])
        accounts_receivable = self._extract_labeled_numeric_value(
            evidence_text,
            ["accounts receivable, net", "accounts receivable"],
        )
        if cash is None or short_term_investments is None or accounts_receivable is None:
            return False
        computed = (cash + short_term_investments + accounts_receivable) / current_liabilities
        return abs(computed - claimed_ratio) <= 0.03

    def _supports_working_capital_claim(self, claim: Claim, evidence_text: str) -> bool:
        current_assets = self._extract_labeled_numeric_value(evidence_text, ["total current assets"])
        current_liabilities = self._extract_labeled_numeric_value(evidence_text, ["total current liabilities"])
        if current_assets is None or current_liabilities is None:
            return False
        working_capital = current_assets - current_liabilities
        lowered = claim.text.lower()
        if "positive" in lowered and working_capital < 0:
            return False
        if "negative" in lowered and working_capital >= 0:
            return False
        claim_values = self._parse_claim_numeric_values(claim)
        if not claim_values:
            return True
        return abs(abs(working_capital) - abs(claim_values[0])) <= max(1.0, abs(working_capital) * 0.02)

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

    def _order_chunks_for_claim(self, claim: Claim, chunks: List[dict]) -> List[dict]:
        if not claim.cited_chunks:
            return chunks
        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        ordered = [chunk_map[chunk_id] for chunk_id in claim.cited_chunks if chunk_id in chunk_map]
        seen = {chunk["chunk_id"] for chunk in ordered}
        ordered.extend(chunk for chunk in chunks if chunk["chunk_id"] not in seen)
        return ordered

    def _combine_chunks_for_claim(self, claim: Claim, chunks: List[dict]) -> dict:
        excerpt_texts = [
            self._extract_relevant_excerpt(chunk["text"], claim.text)
            for chunk in chunks
        ]
        source_types = {
            str(chunk.get("source_type") or "").strip().lower()
            for chunk in chunks
            if chunk.get("source_type")
        }
        primary_flags = [self._has_primary_source(chunk) for chunk in chunks]
        is_primary = None
        if primary_flags and all(flag is True for flag in primary_flags):
            is_primary = True
        elif any(flag is False for flag in primary_flags):
            is_primary = False
        return {
            "chunk_id": "|".join(chunk["chunk_id"] for chunk in chunks),
            "source_id": "|".join(dict.fromkeys(chunk["source_id"] for chunk in chunks)),
            "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "source_ids": list(dict.fromkeys(chunk["source_id"] for chunk in chunks)),
            "text": "\n".join(chunk["text"] for chunk in chunks),
            "excerpt_text": " ".join(excerpt_texts),
            "excerpt_texts": excerpt_texts,
            "source_type": source_types.pop() if len(source_types) == 1 else None,
            "is_primary": is_primary,
        }

    def _extract_labeled_numeric_value(self, text: str, labels: List[str]) -> Optional[float]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return None
        for label in labels:
            pattern = re.compile(
                rf"{re.escape(label)}.{{0,40}}?(\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?)",
                re.IGNORECASE,
            )
            match = pattern.search(normalized)
            if not match:
                continue
            value = self._parse_numeric_literal(match.group(1))
            if value is not None:
                return value
        return None

    def _parse_claim_numeric_values(self, claim: Claim) -> List[float]:
        values: List[float] = []
        for raw in claim.extracted_numbers:
            value = self._parse_numeric_literal(raw)
            if value is not None:
                values.append(value)
        return values

    def _parse_numeric_literal(self, literal: str) -> Optional[float]:
        cleaned = str(literal or "").strip().lower().replace("$", "").replace(",", "").replace(" ", "")
        if not cleaned:
            return None
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        multiplier = 1.0
        for suffix, factor in (
            ("billion", 1_000.0),
            ("bn", 1_000.0),
            ("b", 1_000.0),
            ("million", 1.0),
            ("mn", 1.0),
            ("m", 1.0),
            ("thousand", 0.001),
            ("k", 0.001),
        ):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                multiplier = factor
                break
        try:
            value = float(cleaned) * multiplier
        except ValueError:
            return None
        return -value if negative else value

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
