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

        for source_id in self._ordered_source_ids(claim):
            if source_id == "no_citation" or source_id not in evidence_store:
                continue
            chunks = self.normalize_evidence_chunks(source_id, evidence_store[source_id])
            if not chunks:
                continue
            selected.extend(self._select_candidates_from_source(claim, chunks, limit=limit, seen_keys=seen_keys))

        selected.sort(key=lambda item: item.match_score, reverse=True)
        return selected

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

        claim_tokens = self._focus_tokens(claim.text)
        chunk_tokens = self._focus_tokens(chunk_text)
        overlap = claim_tokens & chunk_tokens
        if overlap:
            score += float(len(overlap))
            reasons.append("token_overlap")

        if claim.claim_type in {"causal", "management_commentary"} and any(
            marker in chunk_lower for marker in ("due to", "because", "driven by", "management", "expects", "guidance")
        ):
            score += 20.0
            reasons.append("semantic_marker_match")

        if not reasons:
            reasons.append("fallback")
        return score, reasons

    def _resolve_primary_flag(self, chunks: List[dict]) -> Optional[bool]:
        flags = [chunk.get("is_primary") for chunk in chunks if chunk.get("is_primary") is not None]
        if not flags:
            return None
        return all(bool(flag) for flag in flags)

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
