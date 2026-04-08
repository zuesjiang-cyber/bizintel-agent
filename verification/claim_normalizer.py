"""
Claim normalization helpers.

The extractor is intentionally coarse. This module turns those coarse sentences into
finance-aware, more atomic claims that are easier to verify and diagnose.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import List, Optional

from agent.schemas import Claim


NUMBER_PATTERN = re.compile(
    r"\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?%?(?![A-Za-z])",
    re.IGNORECASE,
)
PERIOD_PATTERN = re.compile(r"\b(?:q[1-4]\s*20\d{2}|fy\s*20\d{2}|20\d{2}|ttm|ltm)\b", re.IGNORECASE)

METRIC_KEYWORDS = (
    ("operating margin", "operating_margin"),
    ("gross margin", "gross_margin"),
    ("free cash flow", "free_cash_flow"),
    ("cash flow", "cash_flow"),
    ("capital expenditures", "capital_expenditures"),
    ("capital expenditure", "capital_expenditures"),
    ("capex", "capital_expenditures"),
    ("net income", "net_income"),
    ("operating income", "operating_income"),
    ("revenue", "revenue"),
    ("sales", "revenue"),
    ("margin", "margin"),
    ("ratio", "ratio"),
    ("guidance", "guidance"),
    ("debt", "debt"),
    ("inventory", "inventory"),
    ("accounts receivable", "accounts_receivable"),
    ("current assets", "current_assets"),
    ("current liabilities", "current_liabilities"),
    ("products", "products"),
    ("services", "services"),
)

POSITIVE_DIRECTION_TOKENS = (
    "up",
    "increase",
    "increased",
    "grew",
    "growth",
    "rose",
    "expanded",
    "improved",
    "higher",
    "largest",
    "highest",
    "most",
)
NEGATIVE_DIRECTION_TOKENS = (
    "down",
    "decrease",
    "decreased",
    "decline",
    "declined",
    "fell",
    "lower",
    "lowest",
    "least",
    "negative",
)
CAUSAL_MARKERS = ("due to", "because", "driven by", "reflecting", "primarily because", "mainly due to")
MANAGEMENT_MARKERS = ("management said", "management noted", "management expects", "guided", "guidance", "expects")
INFERENCE_MARKERS = ("appears", "suggests", "likely", "may", "might", "implies", "could")
COORDINATING_SPLIT_MARKERS = (" but ", " while ", "; ")
AND_SPLIT_PATTERN = re.compile(r"\s+and\s+", re.IGNORECASE)
GROWTH_TO_PATTERN = re.compile(
    r"^(?P<subject>.+?)\s+(?P<verb>grew|grew by|increased|rose|declined|decreased|fell|expanded|improved)\s+"
    r"(?P<delta>\$?[\d,]+(?:\.\d+)?%?)\s+to\s+(?P<value>\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?%?)"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)


class ClaimNormalizer:
    def normalize_claims(self, claims: List[Claim]) -> List[Claim]:
        normalized: List[Claim] = []
        for claim in claims:
            normalized.extend(self.normalize_claim(claim))
        return normalized

    def normalize_claim(self, claim: Claim) -> List[Claim]:
        atomic_fragments = self._split_atomic_fragments(claim.text)
        if not atomic_fragments:
            return [self._enrich_claim(claim, claim.text, ordinal=0, was_split=False)]

        was_split = len(atomic_fragments) > 1
        return [
            self._enrich_claim(claim, fragment, ordinal=index, was_split=was_split)
            for index, fragment in enumerate(atomic_fragments)
        ]

    def _split_atomic_fragments(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip(" -")
        if not cleaned:
            return []

        special_split = self._split_growth_to_statement(cleaned)
        if special_split:
            return special_split

        fragments = [cleaned]
        for marker in COORDINATING_SPLIT_MARKERS:
            next_fragments: List[str] = []
            for fragment in fragments:
                pieces = [piece.strip(" ,") for piece in fragment.split(marker) if piece.strip(" ,")]
                if len(pieces) >= 2 and all(self._looks_clause_like(piece) for piece in pieces):
                    next_fragments.extend(pieces)
                else:
                    next_fragments.append(fragment)
            fragments = next_fragments

        next_fragments = []
        for fragment in fragments:
            if " and " not in fragment.lower():
                next_fragments.append(fragment)
                continue
            pieces = [piece.strip(" ,") for piece in AND_SPLIT_PATTERN.split(fragment) if piece.strip(" ,")]
            if len(pieces) == 2 and all(self._looks_clause_like(piece) for piece in pieces):
                next_fragments.extend(pieces)
            else:
                next_fragments.append(fragment)
        return next_fragments

    def _split_growth_to_statement(self, text: str) -> List[str]:
        match = GROWTH_TO_PATTERN.match(text)
        if not match:
            return []
        subject = match.group("subject").strip(" ,")
        verb = match.group("verb").strip()
        delta = match.group("delta").strip()
        value = match.group("value").strip()
        tail = match.group("tail").strip()
        first = f"{subject} {verb} {delta}".strip()
        second = f"{subject} was {value}".strip()
        if tail:
            second = f"{second} {tail}".strip()
        return [first, second]

    def _looks_clause_like(self, text: str) -> bool:
        lowered = text.lower()
        if len(text.split()) < 4:
            return False
        if any(marker in lowered for marker in MANAGEMENT_MARKERS):
            return True
        verb_markers = (
            " was ",
            " were ",
            " is ",
            " are ",
            " grew ",
            " increased ",
            " decreased ",
            " declined ",
            " reported ",
            " sold ",
            " generated ",
            " represented ",
            " accounted for ",
            " expects ",
            " guided ",
        )
        return any(marker in f" {lowered} " for marker in verb_markers) or bool(NUMBER_PATTERN.search(text))

    def _enrich_claim(self, parent: Claim, text: str, *, ordinal: int, was_split: bool) -> Claim:
        clean_text = re.sub(r"\s+", " ", text).strip(" -")
        extracted_numbers = NUMBER_PATTERN.findall(clean_text)
        claim_type = self._classify_claim_type(clean_text, extracted_numbers)
        metric = self._extract_metric(clean_text)
        value = extracted_numbers[0] if extracted_numbers else None
        unit = self._extract_unit(value or clean_text)
        period = self._extract_period(clean_text)
        comparison_basis = self._extract_comparison_basis(clean_text)
        directionality = self._extract_directionality(clean_text)
        is_inference = any(marker in clean_text.lower() for marker in INFERENCE_MARKERS)
        requires_primary_source = self._requires_primary_source(clean_text, claim_type, extracted_numbers, is_inference)
        risk_level = self._risk_level(claim_type, extracted_numbers, requires_primary_source, is_inference)
        subject = self._extract_subject(clean_text, metric)

        specificity_score = parent.specificity_score
        if was_split:
            specificity_score = max(specificity_score, 0.4)
        claim_id = parent.claim_id if not was_split and ordinal == 0 else self._child_claim_id(parent.claim_id, clean_text, ordinal)
        contains_numbers = bool(extracted_numbers)

        return replace(
            parent,
            claim_id=claim_id,
            text=clean_text,
            contains_numbers=contains_numbers,
            extracted_numbers=extracted_numbers,
            specificity_score=round(specificity_score, 2),
            claim_type=claim_type,
            risk_level=risk_level,
            subject=subject,
            metric=metric,
            value=value,
            unit=unit,
            period=period,
            comparison_basis=comparison_basis,
            directionality=directionality,
            is_inference=is_inference,
            requires_primary_source=requires_primary_source,
            atomicity=True,
        )

    def _child_claim_id(self, parent_id: str, text: str, ordinal: int) -> str:
        digest = hashlib.md5(f"{parent_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:10]
        return f"{parent_id[:6]}-{digest}"

    def _classify_claim_type(self, text: str, extracted_numbers: List[str]) -> str:
        lowered = text.lower()
        if any(marker in lowered for marker in MANAGEMENT_MARKERS):
            return "management_commentary"
        if any(marker in lowered for marker in CAUSAL_MARKERS):
            return "causal"
        if any(token in lowered for token in ("versus", " vs ", "compared", "highest", "lowest", "most", "least")):
            return "comparative"
        if extracted_numbers or "%" in text:
            return "numeric"
        return "descriptive"

    def _extract_metric(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for keyword, metric in METRIC_KEYWORDS:
            if keyword in lowered:
                return metric
        return None

    def _extract_subject(self, text: str, metric: Optional[str]) -> Optional[str]:
        if metric:
            return metric
        capitalized = re.findall(r"\b[A-Z][A-Za-z&.-]+(?:\s+[A-Z][A-Za-z&.-]+){0,2}\b", text)
        if capitalized:
            return capitalized[0]
        words = text.split()
        if not words:
            return None
        return " ".join(words[: min(4, len(words))])

    def _extract_unit(self, text: str) -> Optional[str]:
        lowered = text.lower()
        if "%" in text or "percent" in lowered or "percentage points" in lowered or "pts" in lowered:
            return "%"
        if "$" in text or "usd" in lowered or "dollar" in lowered:
            return "USD"
        for suffix in ("billion", "million", "thousand", "bn", "mn", "b", "m", "k"):
            if suffix in lowered:
                return suffix
        return None

    def _extract_period(self, text: str) -> Optional[str]:
        match = PERIOD_PATTERN.search(text or "")
        return match.group(0) if match else None

    def _extract_comparison_basis(self, text: str) -> Optional[str]:
        lowered = text.lower()
        if "yoy" in lowered or "year-over-year" in lowered:
            return "yoy"
        if "qoq" in lowered or "quarter-over-quarter" in lowered:
            return "qoq"
        if "versus guidance" in lowered or "vs guidance" in lowered:
            return "vs_guidance"
        if "versus" in lowered or " vs " in lowered or "compared" in lowered:
            return "comparison"
        return None

    def _extract_directionality(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for token in POSITIVE_DIRECTION_TOKENS:
            if token in lowered:
                return "up"
        for token in NEGATIVE_DIRECTION_TOKENS:
            if token in lowered:
                return "down"
        return None

    def _requires_primary_source(
        self,
        text: str,
        claim_type: str,
        extracted_numbers: List[str],
        is_inference: bool,
    ) -> bool:
        lowered = text.lower()
        if claim_type in {"numeric", "comparative", "causal"}:
            return True
        if claim_type == "management_commentary":
            return True
        if extracted_numbers:
            return True
        return is_inference or any(keyword in lowered for _, keyword in (("metric", "revenue"), ("metric", "margin"), ("metric", "guidance")))

    def _risk_level(
        self,
        claim_type: str,
        extracted_numbers: List[str],
        requires_primary_source: bool,
        is_inference: bool,
    ) -> str:
        if claim_type in {"numeric", "comparative", "causal"} or extracted_numbers or requires_primary_source:
            return "high"
        if is_inference or claim_type == "management_commentary":
            return "medium"
        return "low"
