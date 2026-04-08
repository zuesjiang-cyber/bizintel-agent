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
from verification.claim_normalizer import ClaimNormalizer
from verification.evidence_selector import EvidenceCandidate, EvidenceSelector
from verification.rule_engine import RuleCheckResult, RuleEngine

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
    PRIMARY_SOURCE_TYPES = RuleEngine.PRIMARY_SOURCE_TYPES
    FINANCIAL_KEYWORDS = RuleEngine.FINANCIAL_KEYWORDS
    DIRECTIONALITY_TOKENS = RuleEngine.DIRECTIONALITY_TOKENS
    LINE_ITEM_CLAIM_SPECS = RuleEngine.LINE_ITEM_CLAIM_SPECS

    def __init__(self, use_dummy_model: bool = False):
        self._nli_model: Optional[CrossEncoder] = None
        self.use_dummy_model = use_dummy_model
        self.strong_threshold = settings.entailment_strong_threshold
        self.moderate_threshold = settings.entailment_moderate_threshold
        self.device = _resolve_inference_device()
        self.claim_normalizer = ClaimNormalizer()
        self.evidence_selector = EvidenceSelector()
        self.rule_engine = RuleEngine()

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
        return self.evidence_selector.normalize_evidence_chunks(source_id, chunks)

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
        normalized = chunks
        if normalized and not isinstance(normalized[0], dict):
            normalized = self._normalize_evidence_chunks("anonymous_source", list(normalized))
        candidates = self.evidence_selector.select_from_chunks(claim, normalized)
        selected = []
        for candidate in candidates:
            if len(candidate.chunk_ids) != 1:
                continue
            selected.append(
                {
                    "chunk_id": candidate.chunk_ids[0],
                    "source_id": candidate.source_id,
                    "text": candidate.text,
                    "source_type": candidate.source_type,
                    "is_primary": candidate.is_primary,
                }
            )
        limit = max(1, int(settings.verification_max_chunks_per_source))
        return selected[:limit] if selected else list(normalized)[:limit]

    def verify_memo(self, claims: List[Claim], evidence_store: Dict[str, List[object]]) -> List[VerificationResult]:
        """
        验证一组 claim。
        evidence_store: source_id -> [text_chunks]
        """
        if not claims:
            return []

        normalized_claims = self.claim_normalizer.normalize_claims(claims)
        results = []
        pairs_to_score = []
        claim_candidate_map: List[tuple[str, EvidenceCandidate]] = []
        temp_results: Dict[str, VerificationResult] = {}

        for claim in normalized_claims:
            structure_check = self.rule_engine.check_claim_structure(claim, evidence_store)
            if not structure_check.passed:
                temp_results[claim.claim_id] = self._build_failed_result(claim, structure_check)
                continue

            candidates = self.evidence_selector.select_candidates(claim, evidence_store)
            if not candidates:
                temp_results[claim.claim_id] = VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if self._claim_requires_numeric_gate(claim) else None,
                    supporting_evidence=[],
                    explanation="Claim cites a known source, but no evidence chunks were available for scoring.",
                    failure_reason="no_supporting_chunk",
                    failure_stage="evidence",
                    supporting_source_ids=structure_check.details.get("known_sources", []),
                    primary_source_supported=None,
                    verdict_trace={
                        "structure_check": self._serialize_rule_check(structure_check),
                        "candidate_count": 0,
                    },
                    review_notes=["No candidate evidence chunk survived source-aware selection."],
                )
                continue

            for candidate in candidates:
                excerpt = self._extract_relevant_excerpt(candidate.text, claim.text)
                candidate.excerpt_text = excerpt
                if len(candidate.chunk_ids) > 1:
                    candidate.excerpt_texts = [excerpt]
                pairs_to_score.append((excerpt, claim.text))
                claim_candidate_map.append((claim.claim_id, candidate))

        if not pairs_to_score:
            return [temp_results[claim.claim_id] for claim in normalized_claims]

        # 2. 批量跑 NLI 模型
        logger.info(f"Running NLI verification for {len(pairs_to_score)} pairs...")
        model = self._get_model()
        scores = self._predict_pairs(model, pairs_to_score)

        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        entailment_scores = probs[:, 1]
        contradiction_scores = probs[:, 0]

        candidate_scores: Dict[str, List[dict]] = {}
        for pair_idx, (claim_id, candidate) in enumerate(claim_candidate_map):
            candidate_scores.setdefault(claim_id, []).append(
                {
                    "candidate": candidate,
                    "entailment": float(entailment_scores[pair_idx]),
                    "contradiction": float(contradiction_scores[pair_idx]),
                }
            )

        for claim in normalized_claims:
            if claim.claim_id in temp_results:
                continue
            scored_candidates = candidate_scores.get(claim.claim_id, [])
            if not scored_candidates:
                temp_results[claim.claim_id] = VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if self._claim_requires_numeric_gate(claim) else None,
                    supporting_evidence=[],
                    explanation="No evidence candidates were available after scoring.",
                    failure_reason="no_supporting_chunk",
                    failure_stage="evidence",
                    supporting_source_ids=[source_id for source_id in claim.cited_sources if source_id != "no_citation"],
                    review_notes=["Candidate scoring produced no usable evidence rows."],
                )
                continue

            explicit_chunk_set = set(dict.fromkeys(chunk_id for chunk_id in claim.cited_chunks if chunk_id))
            best_entry = max(
                scored_candidates,
                key=lambda item: (
                    1 if explicit_chunk_set and explicit_chunk_set.issubset(set(item["candidate"].chunk_ids)) else 0,
                    item["entailment"],
                    item["candidate"].match_score,
                    -item["contradiction"],
                ),
            )
            candidate = best_entry["candidate"]
            score = float(best_entry["entailment"])
            contradiction_score = float(best_entry["contradiction"])
            rule_check = self.rule_engine.evaluate_candidate(claim, candidate)
            contradiction_detected = bool(rule_check.details.get("contradiction_detected"))
            confidence = ConfidenceLevel.WEAK
            failure_reason = None
            failure_stage = None
            if score >= self.strong_threshold:
                confidence = ConfidenceLevel.STRONG
            elif score >= self.moderate_threshold:
                confidence = ConfidenceLevel.MODERATE

            if contradiction_detected or (
                contradiction_score >= self.strong_threshold and contradiction_score > score + 0.1
            ):
                confidence = ConfidenceLevel.CONTRADICTED
                failure_reason = "contradiction"
                failure_stage = "semantics" if rule_check.passed else rule_check.stage
            elif not rule_check.passed:
                confidence = ConfidenceLevel.UNSUPPORTED
                failure_reason = rule_check.failure_reason
                failure_stage = rule_check.stage
            elif score < self.moderate_threshold:
                failure_reason = "low_entailment"
                failure_stage = "semantics"

            explanation = (
                f"NLI entailment={score:.2f}; contradiction={contradiction_score:.2f}. "
                f"Numeric match={'Yes' if rule_check.details.get('numeric_verified') else ('No' if rule_check.details.get('numeric_verified') is False else 'N/A')}. "
                f"Primary source={'Yes' if rule_check.details.get('primary_source_supported') else ('No' if rule_check.details.get('primary_source_supported') is False else 'N/A')}."
            )

            temp_results[claim.claim_id] = VerificationResult(
                claim=claim,
                confidence=confidence,
                nli_score=score,
                numeric_verified=rule_check.details.get("numeric_verified"),
                supporting_evidence=[candidate.excerpt_text or candidate.text],
                explanation=explanation,
                failure_reason=failure_reason,
                failure_stage=failure_stage,
                supporting_source_ids=[candidate.source_id],
                supporting_chunk_ids=list(candidate.chunk_ids),
                primary_source_supported=rule_check.details.get("primary_source_supported"),
                period_verified=rule_check.details.get("period_verified"),
                currency_verified=rule_check.details.get("currency_verified"),
                unit_verified=rule_check.details.get("unit_verified"),
                directionality_verified=rule_check.details.get("directionality_verified"),
                contradiction_detected=contradiction_detected or confidence == ConfidenceLevel.CONTRADICTED,
                verdict_trace={
                    "selected_candidate": self._serialize_candidate_trace(best_entry),
                    "candidate_scores": [
                        self._serialize_candidate_trace(entry)
                        for entry in sorted(scored_candidates, key=lambda item: (item["entailment"], item["candidate"].match_score), reverse=True)[:5]
                    ],
                    "rule_check": self._serialize_rule_check(rule_check),
                },
                review_notes=list(rule_check.details.get("review_notes", [])),
            )

        return [temp_results[claim.claim_id] for claim in normalized_claims]

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
        if isinstance(evidence, dict):
            candidate = EvidenceCandidate(
                source_id=str(evidence.get("source_id", "")),
                chunk_ids=list(evidence.get("chunk_ids") or [str(evidence.get("chunk_id", ""))]),
                text=str(evidence.get("text", "")),
                source_type=evidence.get("source_type"),
                is_primary=evidence.get("is_primary"),
            )
            return self.rule_engine._has_primary_source(candidate)
        return self.rule_engine._has_primary_source(evidence)

    def _claim_requires_primary_source(self, claim: Claim) -> bool:
        return self.rule_engine.claim_requires_primary_source(claim)

    def _claim_requires_numeric_gate(self, claim: Claim) -> bool:
        return self.rule_engine.claim_requires_numeric_gate(claim)

    def _verify_numeric_alignment(self, claim: Claim, evidence_text: str) -> bool:
        return self.rule_engine.verify_numeric_alignment(claim, evidence_text)

    def _supports_derived_numeric_claim(self, claim: Claim, evidence_text: str) -> bool:
        return self.rule_engine._supports_derived_numeric_claim(claim, evidence_text)

    def _supports_line_item_numeric_claim(self, claim: Claim, evidence_text: str) -> bool:
        return self.rule_engine._supports_line_item_numeric_claim(claim, evidence_text)

    def _supports_quick_ratio_claim(self, claim: Claim, evidence_text: str) -> bool:
        return self.rule_engine._supports_quick_ratio_claim(claim, evidence_text)

    def _supports_working_capital_claim(self, claim: Claim, evidence_text: str) -> bool:
        return self.rule_engine._supports_working_capital_claim(claim, evidence_text)

    def _normalize_numeric_token(self, token: str) -> str:
        return self.rule_engine._normalize_numeric_token(token)

    def _normalize_numeric_haystack(self, text: str) -> set[str]:
        return self.rule_engine._normalize_numeric_haystack(text)

    def _extract_period_tokens(self, text: str) -> set[str]:
        return self.rule_engine._extract_period_tokens(text)

    def _extract_currency_tokens(self, text: str) -> set[str]:
        return self.rule_engine._extract_currency_tokens(text)

    def _extract_directionality_tokens(self, text: str) -> set[str]:
        return self.rule_engine._extract_directionality_tokens(text)

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
        return self.rule_engine._extract_labeled_numeric_value(text, labels)

    def _parse_claim_numeric_values(self, claim: Claim) -> List[float]:
        return self.rule_engine._parse_claim_numeric_values(claim)

    def _parse_numeric_literal(self, literal: str) -> Optional[float]:
        return self.rule_engine._parse_numeric_literal(literal)

    def summary_stats(self, results: List[VerificationResult]) -> dict:
        """生成验证统计信息"""
        total = len(results)
        if total == 0:
            return {
                "total_claims": 0,
                "strong": 0, "moderate": 0, "weak": 0, "unsupported": 0, "contradicted": 0,
                "citation_marker_coverage": 0.0, "verified_claim_coverage": 0.0, "avg_nli_score": 0.0,
                "numeric_error_rate": 0.0, "primary_source_coverage": 0.0,
            }

        counts = {c: 0 for c in ConfidenceLevel}
        for r in results:
            counts[r.confidence] += 1

        supported = total - counts[ConfidenceLevel.UNSUPPORTED] - counts[ConfidenceLevel.CONTRADICTED]
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
            "contradicted": counts[ConfidenceLevel.CONTRADICTED],
            "citation_marker_coverage": cited / total if total > 0 else 0.0,
            "verified_claim_coverage": supported / total if total > 0 else 0.0,
            "avg_nli_score": sum(r.nli_score for r in results) / total,
            "numeric_error_rate": len(numeric_errors) / len(numeric_claims) if numeric_claims else 0.0,
            "primary_source_coverage": len(primary_supported) / len(primary_required) if primary_required else 0.0,
        }

    def _build_failed_result(self, claim: Claim, rule_check: RuleCheckResult) -> VerificationResult:
        details = rule_check.details or {}
        requires_numeric = self._claim_requires_numeric_gate(claim)
        supporting_sources = details.get("known_sources", [])
        explanation = details.get("review_notes", ["Verification failed before semantic scoring."])[0]
        return VerificationResult(
            claim=claim,
            confidence=ConfidenceLevel.UNSUPPORTED,
            nli_score=0.0,
            numeric_verified=False if requires_numeric else None,
            supporting_evidence=[],
            explanation=explanation,
            failure_reason=rule_check.failure_reason,
            failure_stage=rule_check.stage,
            supporting_source_ids=list(supporting_sources),
            primary_source_supported=None,
            verdict_trace={"structure_check": self._serialize_rule_check(rule_check)},
            review_notes=list(details.get("review_notes", [])),
        )

    def _serialize_rule_check(self, rule_check: RuleCheckResult) -> dict:
        return {
            "passed": rule_check.passed,
            "stage": rule_check.stage,
            "failure_reason": rule_check.failure_reason,
            "details": dict(rule_check.details or {}),
        }

    def _serialize_candidate_trace(self, entry: dict) -> dict:
        candidate = entry["candidate"]
        return {
            "source_id": candidate.source_id,
            "chunk_ids": list(candidate.chunk_ids),
            "match_score": candidate.match_score,
            "match_reasons": list(candidate.match_reasons),
            "entailment": entry["entailment"],
            "contradiction": entry["contradiction"],
        }
