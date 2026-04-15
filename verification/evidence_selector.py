"""
Evidence candidate selection for claim verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agent.config import settings
from agent.schemas import Claim


NUMBER_PATTERN = re.compile(
    r"\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?%?(?![A-Za-z])",
    re.IGNORECASE,
)
PERIOD_PATTERN = re.compile(r"\b(?:q[1-4]\s*20\d{2}|fy\s*20\d{2}|20\d{2}|ttm|ltm)\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\b[\w$%.]+\b")
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "which", "when", "where",
    "year", "fiscal", "company", "report", "reported", "results", "statement", "financial",
    "analysis", "section", "question", "during", "major", "using",
}
PRIMARY_SOURCE_TYPES = {
    "annual_report",
    "quarterly_report",
    "quarterly_results",
    "earnings_call_transcript",
    "shareholder_letter",
}
FINANCIAL_LABEL_PHRASES = (
    "cash and cash equivalents",
    "short-term investments",
    "accounts receivable",
    "total current assets",
    "total current liabilities",
    "capital expenditures",
    "capital expenditure",
    "capex",
    "net sales",
    "revenue",
    "operating income",
    "net income",
    "ebitda",
    "ebitdar",
    "legal proceedings",
    "material legal proceedings",
    "revolving credit agreement",
    "aggregate commitments",
    "borrowing capacity",
    "proposal",
    "vote",
    "products",
    "services",
    "customer concentration",
    "major customer",
    "significant customer",
)


@dataclass
class EvidenceCandidate:
    source_id: str
    chunk_ids: List[str]
    text: str
    source_type: Optional[str] = None
    is_primary: Optional[bool] = None
    match_score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    excerpt_text: Optional[str] = None
    excerpt_texts: List[str] = field(default_factory=list)


class EvidenceSelector:
    def normalize_evidence_chunks(self, source_id: str, chunks: List[object]) -> List[dict]:
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
                        "position": idx,
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
                        "position": idx,
                    }
                )
        return normalized

    def select_candidates(self, claim: Claim, evidence_store: Dict[str, List[object]]) -> List[EvidenceCandidate]:
        limit = max(1, int(settings.verification_max_chunks_per_source))
        seen_keys = set()
        selected: List[EvidenceCandidate] = []
        processed_sources = set()

        for source_id in self._ordered_source_ids(claim):
            if source_id == "no_citation" or source_id not in evidence_store:
                continue
            processed_sources.add(source_id)
            chunks = self.normalize_evidence_chunks(source_id, evidence_store[source_id])
            if not chunks:
                continue
            selected.extend(self._select_candidates_from_source(claim, chunks, limit=limit, seen_keys=seen_keys))

        if self._claim_requires_primary_support(claim):
            supplemental_sources = []
            for source_id, raw_chunks in evidence_store.items():
                if source_id == "no_citation" or source_id in processed_sources:
                    continue
                chunks = self.normalize_evidence_chunks(source_id, raw_chunks)
                if not chunks or not self._source_has_primary_support(chunks):
                    continue
                supplemental_sources.append((source_id, chunks))
            supplemental_sources.sort(
                key=lambda item: (
                    1 if self._source_has_primary_support(item[1]) else 0,
                    len(item[1]),
                    item[0],
                ),
                reverse=True,
            )
            for source_id, chunks in supplemental_sources:
                selected.extend(self._select_candidates_from_source(claim, chunks, limit=limit, seen_keys=seen_keys))

        selected.sort(key=lambda item: item.match_score, reverse=True)
        return selected

    def select_primary_rebinding_candidates(
        self,
        claim: Claim,
        evidence_store: Dict[str, List[object]],
        *,
        exclude_source_ids: Optional[Sequence[str]] = None,
        max_candidates: int = 8,
    ) -> List[EvidenceCandidate]:
        limit = max(1, int(settings.verification_max_chunks_per_source))
        seen_keys = set()
        excluded = set(exclude_source_ids or [])
        selected: List[EvidenceCandidate] = []

        for source_id, raw_chunks in evidence_store.items():
            if source_id in excluded or source_id == "no_citation":
                continue
            chunks = self.normalize_evidence_chunks(source_id, raw_chunks)
            if not chunks or not self._source_has_primary_support(chunks):
                continue
            for candidate in self._select_candidates_from_source(claim, chunks, limit=limit, seen_keys=seen_keys):
                if self._candidate_has_primary_support(candidate):
                    selected.append(candidate)

        selected.sort(key=lambda item: item.match_score, reverse=True)
        return selected[:max_candidates]

    def select_from_chunks(self, claim: Claim, chunks: Sequence[dict], *, limit: Optional[int] = None) -> List[EvidenceCandidate]:
        return self._select_candidates_from_source(
            claim,
            list(chunks),
            limit=limit or max(1, int(settings.verification_max_chunks_per_source)),
            seen_keys=set(),
        )

    def _select_candidates_from_source(
        self,
        claim: Claim,
        chunks: List[dict],
        *,
        limit: int,
        seen_keys: set,
    ) -> List[EvidenceCandidate]:
        if not chunks:
            return []
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: self._score_chunk(claim, chunk)[0],
            reverse=True,
        )

        candidates: List[EvidenceCandidate] = []
        top_ranked = ranked_chunks[: max(limit, 2)]
        for chunk in top_ranked:
            score, reasons = self._score_chunk(claim, chunk)
            candidate = EvidenceCandidate(
                source_id=chunk["source_id"],
                chunk_ids=[chunk["chunk_id"]],
                text=chunk["text"],
                source_type=chunk.get("source_type"),
                is_primary=chunk.get("is_primary"),
                match_score=score,
                match_reasons=reasons,
            )
            key = tuple(candidate.chunk_ids)
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(candidate)

            neighbor = self._neighbor_window(claim, chunk, chunks)
            if neighbor is not None:
                key = tuple(neighbor.chunk_ids)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(neighbor)

        ordered_cited = self._combined_cited_candidate(claim, chunks)
        if ordered_cited is not None:
            key = tuple(ordered_cited.chunk_ids)
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(ordered_cited)

        candidates.sort(key=lambda item: item.match_score, reverse=True)
        return candidates[: max(limit + 1, 2)]

    def _ordered_source_ids(self, claim: Claim) -> List[str]:
        seen = set()
        ordered = []
        for source_id in claim.cited_sources:
            if source_id in seen:
                continue
            seen.add(source_id)
            ordered.append(source_id)
        return ordered

    def _combined_cited_candidate(self, claim: Claim, chunks: List[dict]) -> Optional[EvidenceCandidate]:
        unique_cited_chunks = list(dict.fromkeys(chunk_id for chunk_id in claim.cited_chunks if chunk_id))
        if len(unique_cited_chunks) < 2:
            return None
        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        if not all(chunk_id in chunk_map for chunk_id in unique_cited_chunks):
            return None
        ordered_chunks = [chunk_map[chunk_id] for chunk_id in unique_cited_chunks]
        score = 0.0
        reasons = ["explicit_chunk_binding", "multi_chunk_window"]
        for chunk in ordered_chunks:
            chunk_score, _ = self._score_chunk(claim, chunk)
            score += chunk_score
        return EvidenceCandidate(
            source_id=ordered_chunks[0]["source_id"],
            chunk_ids=[chunk["chunk_id"] for chunk in ordered_chunks],
            text="\n".join(chunk["text"] for chunk in ordered_chunks),
            source_type=ordered_chunks[0].get("source_type"),
            is_primary=self._resolve_primary_flag(ordered_chunks),
            match_score=score + 50.0,
            match_reasons=reasons,
        )

    def _neighbor_window(self, claim: Claim, chunk: dict, chunks: List[dict]) -> Optional[EvidenceCandidate]:
        position = chunk.get("position")
        if position is None:
            return None
        by_position = {item.get("position"): item for item in chunks}
        neighbor = by_position.get(position + 1)
        if neighbor is None:
            return None
        base_score, reasons = self._score_chunk(claim, chunk)
        neighbor_score, neighbor_reasons = self._score_chunk(claim, neighbor)
        combined_reasons = sorted(set(reasons + neighbor_reasons + ["neighbor_window"]))
        match_score = base_score + (neighbor_score * 0.35)
        if chunk["chunk_id"] in claim.cited_chunks or neighbor["chunk_id"] in claim.cited_chunks:
            match_score = max(base_score, neighbor_score) - 5.0
        return EvidenceCandidate(
            source_id=chunk["source_id"],
            chunk_ids=[chunk["chunk_id"], neighbor["chunk_id"]],
            text=f"{chunk['text']}\n{neighbor['text']}",
            source_type=chunk.get("source_type") or neighbor.get("source_type"),
            is_primary=self._resolve_primary_flag([chunk, neighbor]),
            match_score=match_score,
            match_reasons=combined_reasons,
        )

    def _score_chunk(self, claim: Claim, chunk: dict) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        chunk_text = chunk["text"]
        chunk_lower = chunk_text.lower()
        requires_primary = self._claim_requires_primary_support(claim)
        has_primary_support = self._chunk_has_primary_support(chunk)

        if chunk["chunk_id"] in claim.cited_chunks:
            score += 120.0
            reasons.append("explicit_chunk_binding")
        if chunk["source_id"] in claim.cited_sources:
            score += 40.0
            reasons.append("explicit_source_binding")

        claim_periods = self._extract_period_tokens(claim.period or claim.text)
        chunk_periods = self._extract_period_tokens(chunk_text)
        if claim_periods and chunk_periods and (claim_periods & chunk_periods):
            score += 35.0
            reasons.append("period_match")

        if claim.metric and claim.metric.replace("_", " ") in chunk_lower:
            score += 30.0
            reasons.append("metric_match")

        phrase_hits = [phrase for phrase in self._claim_label_phrases(claim) if phrase in chunk_lower]
        if phrase_hits:
            score += 24.0 + (8.0 * min(2, len(phrase_hits)))
            reasons.append("financial_label_match")

        claim_numbers = {self._normalize_numeric_token(value) for value in claim.extracted_numbers if value}
        chunk_numbers = {self._normalize_numeric_token(match.group(0)) for match in NUMBER_PATTERN.finditer(chunk_text)}
        numeric_overlap = claim_numbers & chunk_numbers
        if numeric_overlap:
            score += 100.0 * len(numeric_overlap)
            reasons.append("numeric_match")

        claim_units = self._extract_unit_tokens(claim.unit or claim.text)
        chunk_units = self._extract_unit_tokens(chunk_text)
        if claim_units and chunk_units and (claim_units & chunk_units):
            score += 15.0
            reasons.append("unit_match")

        if requires_primary and has_primary_support:
            score += 32.0
            reasons.append("primary_source_preferred")
        elif requires_primary and has_primary_support is False:
            score -= 18.0
            reasons.append("non_primary_penalty")

        claim_tokens = self._focus_tokens(claim.text)
        chunk_tokens = self._focus_tokens(chunk_text)
        overlap = claim_tokens & chunk_tokens
        if overlap:
            score += float(len(overlap))
            reasons.append("token_overlap")

        if claim.claim_type in {"causal", "management_commentary"} and any(
            marker in chunk_lower
            for marker in (
                "due to",
                "because",
                "driven by",
                "primarily due to",
                "resulting from",
                "reflecting",
                "management",
                "expects",
                "guidance",
            )
        ):
            score += 20.0
            reasons.append("semantic_marker_match")

        noise_penalty = self._structural_noise_penalty(claim, chunk_text, chunk_tokens, phrase_hits)
        if noise_penalty:
            score -= noise_penalty
            reasons.append("noise_penalty")

        if not reasons:
            reasons.append("fallback")
        return score, reasons

    def _claim_requires_primary_support(self, claim: Claim) -> bool:
        if getattr(claim, "requires_primary_source", False):
            return True
        if claim.claim_type in {"numeric", "comparative", "causal", "management_commentary"}:
            return True
        if claim.contains_numbers:
            return True
        return False

    def _chunk_has_primary_support(self, chunk: dict) -> Optional[bool]:
        if chunk.get("is_primary") is True:
            return True
        if chunk.get("is_primary") is False:
            source_type = str(chunk.get("source_type") or "").strip().lower()
            return source_type in PRIMARY_SOURCE_TYPES
        source_type = str(chunk.get("source_type") or "").strip().lower()
        if source_type:
            return source_type in PRIMARY_SOURCE_TYPES
        return None

    def _claim_label_phrases(self, claim: Claim) -> List[str]:
        lowered = claim.text.lower()
        phrases = []
        for phrase in FINANCIAL_LABEL_PHRASES:
            if phrase in lowered:
                phrases.append(phrase)
        metric_phrase = (claim.metric or "").replace("_", " ").strip().lower()
        if metric_phrase and metric_phrase not in phrases:
            phrases.append(metric_phrase)
        subject_phrase = (claim.subject or "").replace("_", " ").strip().lower()
        if subject_phrase and len(subject_phrase.split()) >= 2 and subject_phrase not in phrases:
            phrases.append(subject_phrase)
        return phrases

    def _structural_noise_penalty(
        self,
        claim: Claim,
        text: str,
        chunk_tokens: set[str],
        phrase_hits: Sequence[str],
    ) -> float:
        lowered = text.lower()
        penalty = 0.0
        year_count = len({match.group(0) for match in re.finditer(r"\b20\d{2}\b", text)})
        numeric_count = len(NUMBER_PATTERN.findall(text))
        query_focus_overlap = len(self._focus_tokens(claim.text) & chunk_tokens)

        if "please visit" in lowered or "investors." in lowered or "https://" in lowered:
            penalty += 35.0
        if year_count >= 5 and any(
            phrase in lowered
            for phrase in (
                "earnings press release",
                "form 10-k",
                "form 10-q",
                "prepared management remarks",
                "transcript - investors q&a",
            )
        ):
            penalty += 45.0
        if numeric_count >= 18 and not phrase_hits and query_focus_overlap <= 2:
            penalty += 28.0
        if numeric_count >= 10 and any(marker in lowered for marker in ("see accompanying notes", "page ", "unaudited")) and not phrase_hits:
            penalty += 12.0
        return penalty

    def _resolve_primary_flag(self, chunks: List[dict]) -> Optional[bool]:
        flags = [chunk.get("is_primary") for chunk in chunks if chunk.get("is_primary") is not None]
        if not flags:
            return None
        return all(bool(flag) for flag in flags)

    def _source_has_primary_support(self, chunks: Sequence[dict]) -> bool:
        for chunk in chunks:
            if chunk.get("is_primary") is True:
                return True
            source_type = str(chunk.get("source_type") or "").strip().lower()
            if source_type in PRIMARY_SOURCE_TYPES:
                return True
        return False

    def _candidate_has_primary_support(self, candidate: EvidenceCandidate) -> bool:
        if candidate.is_primary is True:
            return True
        source_type = str(candidate.source_type or "").strip().lower()
        return source_type in PRIMARY_SOURCE_TYPES

    def _normalize_numeric_token(self, token: str) -> str:
        return token.lower().replace("$", "").replace(",", "").strip()

    def _extract_period_tokens(self, text: str) -> set[str]:
        return {
            match.group(0).lower().replace(" ", "")
            for match in PERIOD_PATTERN.finditer(text or "")
        }

    def _extract_unit_tokens(self, text: str) -> set[str]:
        lowered = (text or "").lower()
        units = set(re.findall(r"\b(?:billion|million|thousand|bn|mn|b|m|k|usd|dollars?)\b", lowered))
        if "$" in (text or ""):
            units.add("$")
        if "%" in (text or ""):
            units.add("%")
        return units

    def _focus_tokens(self, text: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9%$]+", " ", (text or "").lower()).strip()
        if not normalized:
            return set()
        return {
            token for token in TOKEN_PATTERN.findall(normalized)
            if len(token) > 2 and token not in STOPWORDS
        }
