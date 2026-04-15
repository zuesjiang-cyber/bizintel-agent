"""
Finance-oriented hard rules for claim verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.schemas import Claim
from verification.evidence_selector import EvidenceCandidate


NUMERIC_TOKEN_PATTERN = re.compile(
    r"\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|b|m|k))?%?(?![A-Za-z])",
    re.IGNORECASE,
)
PERIOD_PATTERN = re.compile(r"\b(?:q[1-4]\s*20\d{2}|fy\s*20\d{2}|20\d{2}|ttm|ltm)\b", re.IGNORECASE)
UNIT_PATTERN = re.compile(r"\b(?:billion|million|thousand|bn|mn|b|m|k|usd|dollars?|percentage points|pts)\b", re.IGNORECASE)


@dataclass
class RuleCheckResult:
    passed: bool
    stage: str
    failure_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
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
        "decline", "declined", "expanded", "improved", "fell", "rose", "higher", "lower",
        "highest", "lowest", "largest", "smallest", "most", "least",
    }
    POSITIVE_DIRECTION_TOKENS = {
        "up", "increase", "increased", "grew", "growth", "expanded", "improved", "rose",
        "higher", "highest", "largest", "most",
    }
    NEGATIVE_DIRECTION_TOKENS = {
        "down", "decrease", "decreased", "decline", "declined", "fell", "lower", "lowest",
        "smallest", "least",
    }
    LINE_ITEM_CLAIM_SPECS = (
        {
            "keywords": ("capital expenditure", "capital expenditures", "capex"),
            "labels": ("capital expenditures", "capital expenditure"),
            "absolute_value": True,
        },
        {
            "keywords": ("accounts payable",),
            "labels": ("accounts payable",),
        },
        {
            "keywords": ("inventories", "inventory"),
            "labels": ("inventories", "inventory"),
        },
        {
            "keywords": ("accounts receivable", "net ar", "receivables"),
            "labels": ("accounts receivable, net", "accounts receivable", "receivables, net", "trade receivables, net"),
        },
        {
            "keywords": ("pp&e", "property, plant", "property and equipment", "ppne"),
            "labels": (
                "property, plant and equipment, net",
                "property and equipment, net",
                "net property, plant and equipment",
                "property, plant and equipment",
            ),
        },
        {
            "keywords": ("total current liabilities",),
            "labels": ("total current liabilities",),
        },
        {
            "keywords": ("total current assets",),
            "labels": ("total current assets",),
        },
        {
            "keywords": ("total assets",),
            "labels": ("total assets",),
        },
        {
            "keywords": ("total liabilities",),
            "labels": ("total liabilities",),
        },
        {
            "keywords": ("depreciation", "amortization", "d&a"),
            "labels": ("depreciation and amortization",),
            "absolute_value": True,
        },
        {
            "keywords": ("operating income",),
            "labels": ("operating income",),
        },
        {
            "keywords": ("net income",),
            "labels": ("net income",),
        },
        {
            "keywords": ("revenue", "net sales", "sales"),
            "labels": ("total revenue", "net revenue", "net sales", "revenue", "sales"),
        },
        {
            "keywords": ("cost of goods sold", "cogs", "cost of sales", "cost of revenue"),
            "labels": ("cost of goods sold", "cost of sales", "cost of revenue"),
        },
    )

    def check_claim_structure(self, claim: Claim, evidence_store: Dict[str, List[object]]) -> RuleCheckResult:
        if not claim.cited_sources or all(source_id == "no_citation" for source_id in claim.cited_sources):
            return RuleCheckResult(
                passed=False,
                stage="structure",
                failure_reason="missing_citation",
                details={"review_notes": ["Claim does not include a valid citation marker."]},
            )

        known_sources = [source_id for source_id in dict.fromkeys(claim.cited_sources) if source_id in evidence_store]
        if not known_sources:
            return RuleCheckResult(
                passed=False,
                stage="structure",
                failure_reason="bad_source_id",
                details={"review_notes": ["Claim cites sources that are not present in the evidence store."]},
            )
        return RuleCheckResult(
            passed=True,
            stage="structure",
            details={"known_sources": known_sources},
        )

    def evaluate_candidate(self, claim: Claim, candidate: EvidenceCandidate) -> RuleCheckResult:
        details = {
            "numeric_verified": None,
            "period_verified": None,
            "currency_verified": None,
            "unit_verified": None,
            "directionality_verified": None,
            "contradiction_detected": False,
            "primary_source_supported": self._has_primary_source(candidate),
            "review_notes": [],
        }

        evidence_text = candidate.text
        requires_numeric_gate = self.claim_requires_numeric_gate(claim)

        if requires_numeric_gate:
            details["numeric_verified"] = self.verify_numeric_alignment(claim, evidence_text)
            if details["numeric_verified"] is False:
                details["review_notes"].append("Numeric alignment failed.")
                details["period_verified"] = self.verify_period_alignment(claim, evidence_text)
                details["currency_verified"] = self.verify_currency_alignment(claim, evidence_text)
                details["unit_verified"] = self.verify_unit_alignment(claim, evidence_text)
                details["directionality_verified"] = self.verify_directionality_alignment(claim, evidence_text)
                details["contradiction_detected"] = bool(
                    details["directionality_verified"] is False or self.detect_directional_contradiction(claim, evidence_text)
                )
                return RuleCheckResult(
                    passed=False,
                    stage="rules",
                    failure_reason="numeric_mismatch",
                    details=details,
                )

        details["period_verified"] = self.verify_period_alignment(claim, evidence_text)
        if details["period_verified"] is False:
            details["review_notes"].append("Reporting period did not match supporting evidence.")
            return RuleCheckResult(
                passed=False,
                stage="rules",
                failure_reason="period_mismatch",
                details=details,
            )

        details["currency_verified"] = self.verify_currency_alignment(claim, evidence_text)
        if details["currency_verified"] is False:
            details["review_notes"].append("Currency token did not match supporting evidence.")
            return RuleCheckResult(
                passed=False,
                stage="rules",
                failure_reason="currency_mismatch",
                details=details,
            )

        details["unit_verified"] = self.verify_unit_alignment(claim, evidence_text)
        if details["unit_verified"] is False:
            details["review_notes"].append("Unit token did not match supporting evidence.")
            return RuleCheckResult(
                passed=False,
                stage="rules",
                failure_reason="unit_mismatch",
                details=details,
            )

        details["directionality_verified"] = self.verify_directionality_alignment(claim, evidence_text)
        details["contradiction_detected"] = self.detect_directional_contradiction(claim, evidence_text)
        if details["directionality_verified"] is False or details["contradiction_detected"]:
            details["review_notes"].append("Directionality contradicted supporting evidence.")
            return RuleCheckResult(
                passed=False,
                stage="rules",
                failure_reason="direction_mismatch",
                details=details,
            )

        if self.claim_requires_primary_source(claim) and details["primary_source_supported"] is False:
            details["review_notes"].append("High-risk claim lacks primary source support.")
            return RuleCheckResult(
                passed=False,
                stage="rules",
                failure_reason="primary_source_missing",
                details=details,
            )

        return RuleCheckResult(
            passed=True,
            stage="rules",
            details=details,
        )

    def claim_requires_primary_source(self, claim: Claim) -> bool:
        if claim.requires_primary_source:
            return True
        return self.claim_requires_numeric_gate(claim) or claim.claim_type in {"comparative", "causal", "management_commentary"}

    def claim_requires_numeric_gate(self, claim: Claim) -> bool:
        if claim.claim_type in {"numeric", "comparative"}:
            return True
        if claim.contains_numbers:
            return True
        text = claim.text.lower()
        return any(keyword in text for keyword in self.FINANCIAL_KEYWORDS)

    def verify_numeric_alignment(self, claim: Claim, evidence_text: str) -> bool:
        if not evidence_text:
            return False

        evidence_lower = evidence_text.lower()
        derived_supported = self._supports_derived_numeric_claim(claim, evidence_text)
        extracted_numbers = [num for num in claim.extracted_numbers if num.strip()]
        normalized_haystack = self._normalize_numeric_haystack(evidence_text)
        missing_numbers = []
        if extracted_numbers:
            for num in extracted_numbers:
                clean_num = self._normalize_numeric_token(num)
                if clean_num not in normalized_haystack:
                    missing_numbers.append(clean_num)
            if missing_numbers and not derived_supported:
                return False

        for token in self._extract_period_tokens(claim.text):
            if token.lower() not in evidence_lower and not derived_supported:
                return False

        for token in self._extract_currency_tokens(claim.text):
            if token.lower() not in evidence_lower and not derived_supported:
                return False

        claim_directionality = self._extract_directionality_tokens(claim.text)
        evidence_directionality = self._extract_directionality_tokens(evidence_text)
        if claim_directionality and not (claim_directionality & evidence_directionality) and not derived_supported:
            return False

        return True

    def verify_period_alignment(self, claim: Claim, evidence_text: str) -> Optional[bool]:
        claim_periods = self._extract_period_tokens(claim.period or claim.text)
        if not claim_periods:
            return None
        evidence_periods = self._extract_period_tokens(evidence_text)
        if not evidence_periods:
            return None
        return bool(claim_periods & evidence_periods)

    def verify_currency_alignment(self, claim: Claim, evidence_text: str) -> Optional[bool]:
        claim_tokens = self._extract_currency_tokens(claim.text)
        if claim.unit:
            claim_tokens |= self._extract_currency_tokens(claim.unit)
        if not claim_tokens:
            return None
        evidence_tokens = self._extract_currency_tokens(evidence_text)
        if not evidence_tokens:
            return None
        return bool(claim_tokens & evidence_tokens)

    def verify_unit_alignment(self, claim: Claim, evidence_text: str) -> Optional[bool]:
        claim_units = self._extract_unit_tokens(claim.text)
        if claim.unit:
            claim_units |= self._extract_unit_tokens(claim.unit)
        if not claim_units:
            return None
        evidence_units = self._extract_unit_tokens(evidence_text)
        if not evidence_units:
            return True if self._supports_derived_numeric_claim(claim, evidence_text) else None
        if claim_units & evidence_units:
            return True
        if self._supports_derived_numeric_claim(claim, evidence_text):
            return True
        return False

    def verify_directionality_alignment(self, claim: Claim, evidence_text: str) -> Optional[bool]:
        claim_dir = self._extract_directionality_tokens(claim.directionality or claim.text)
        if not claim_dir:
            return None
        evidence_dir = self._extract_directionality_tokens(evidence_text)
        if not evidence_dir:
            return None
        return bool(claim_dir & evidence_dir)

    def detect_directional_contradiction(self, claim: Claim, evidence_text: str) -> bool:
        claim_tokens = self._extract_directionality_tokens(claim.directionality or claim.text)
        evidence_tokens = self._extract_directionality_tokens(evidence_text)
        if not claim_tokens or not evidence_tokens:
            return False
        claim_positive = bool(claim_tokens & self.POSITIVE_DIRECTION_TOKENS)
        claim_negative = bool(claim_tokens & self.NEGATIVE_DIRECTION_TOKENS)
        evidence_positive = bool(evidence_tokens & self.POSITIVE_DIRECTION_TOKENS)
        evidence_negative = bool(evidence_tokens & self.NEGATIVE_DIRECTION_TOKENS)
        return (claim_positive and evidence_negative) or (claim_negative and evidence_positive)

    def _has_primary_source(self, evidence: Optional[EvidenceCandidate]) -> Optional[bool]:
        if evidence is None:
            return None
        if evidence.is_primary is not None:
            return bool(evidence.is_primary)
        source_type = (evidence.source_type or "").strip().lower()
        if source_type:
            return source_type in self.PRIMARY_SOURCE_TYPES
        return None

    def _supports_derived_numeric_claim(self, claim: Claim, evidence_text: str) -> bool:
        lowered = claim.text.lower()
        if any(token in lowered for token in ("highest", "lowest", "largest", "smallest", "most", "least")):
            return self._supports_extreme_ranking_claim(claim, evidence_text)
        if "quick ratio" in lowered:
            return self._supports_quick_ratio_claim(claim, evidence_text)
        if "working capital" in lowered:
            return self._supports_working_capital_claim(claim, evidence_text)
        if any(token in lowered for token in ("ebitda", "unadjusted ebitda")) and any(token in lowered for token in ("capex", "capital expenditures", "capital expenditure")) and any(token in lowered for token in ("less", "minus")):
            return self._supports_ebitda_less_capex_claim(claim, evidence_text)
        if "margin" in lowered:
            return self._supports_margin_formula_claim(claim, evidence_text)
        if self._supports_line_item_numeric_claim(claim, evidence_text):
            return True
        return False

    def _supports_extreme_ranking_claim(self, claim: Claim, evidence_text: str) -> bool:
        claim_values = self._parse_claim_numeric_values(claim)
        if not claim_values:
            return False
        claim_value = abs(claim_values[0])
        evidence_values: List[float] = []
        for match in NUMERIC_TOKEN_PATTERN.finditer(evidence_text):
            parsed = self._parse_numeric_literal(match.group(0))
            if parsed is not None:
                evidence_values.append(abs(parsed))
        if len(evidence_values) < 2:
            return False
        lowered = claim.text.lower()
        if any(token in lowered for token in ("highest", "largest", "most")):
            return abs(max(evidence_values) - claim_value) <= max(1.0, claim_value * 0.02)
        if any(token in lowered for token in ("lowest", "smallest", "least")):
            return abs(min(evidence_values) - claim_value) <= max(1.0, claim_value * 0.02)
        return False

    def _supports_line_item_numeric_claim(self, claim: Claim, evidence_text: str) -> bool:
        claim_values = self._parse_claim_numeric_values(claim)
        if not claim_values:
            return False
        lowered = claim.text.lower()
        claim_value = claim_values[0]
        for spec in self.LINE_ITEM_CLAIM_SPECS:
            if not any(keyword in lowered for keyword in spec["keywords"]):
                continue
            evidence_value = self._extract_labeled_numeric_value(evidence_text, list(spec["labels"]))
            if evidence_value is None:
                continue
            if spec.get("absolute_value"):
                claim_value_cmp = abs(claim_value)
                evidence_value_cmp = abs(evidence_value)
            else:
                claim_value_cmp = claim_value
                evidence_value_cmp = evidence_value
            tolerance = max(1.0, abs(evidence_value_cmp) * 0.02)
            if abs(claim_value_cmp - evidence_value_cmp) <= tolerance:
                return True
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

    def _supports_ebitda_less_capex_claim(self, claim: Claim, evidence_text: str) -> bool:
        ebitda = self._extract_labeled_numeric_value(
            evidence_text,
            ["unadjusted ebitda", "adjusted ebitda", "ebitda"],
        )
        capex = self._extract_labeled_numeric_value(
            evidence_text,
            ["capital expenditures", "capital expenditure"],
        )
        if ebitda is None or capex is None:
            return False
        claim_value = self._extract_first_claim_numeric_value(claim)
        if claim_value is None:
            return False
        computed = float(ebitda) - abs(float(capex))
        return abs(computed - claim_value) <= max(1.0, abs(computed) * 0.02)

    def _supports_margin_formula_claim(self, claim: Claim, evidence_text: str) -> bool:
        claim_percent = self._extract_claim_percentage_value(claim)
        if claim_percent is None:
            return False
        numerator_labels = self._margin_numerator_labels(claim.text)
        if not numerator_labels:
            return False
        numerator = self._extract_labeled_numeric_value(evidence_text, numerator_labels)
        revenue = self._extract_labeled_numeric_value(
            evidence_text,
            ["total revenue", "net revenue", "net sales", "revenue", "sales"],
        )
        if numerator is None or revenue in (None, 0):
            return False
        computed_percent = (float(numerator) / float(revenue)) * 100.0
        return abs(computed_percent - claim_percent) <= 0.2

    def _margin_numerator_labels(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        if "depreciation and amortization" in lowered or "d&a" in lowered:
            return ["depreciation and amortization"]
        if "operating margin" in lowered or "operating income" in lowered:
            return ["operating income"]
        if "unadjusted ebitda" in lowered:
            return ["unadjusted ebitda", "adjusted ebitda", "ebitda"]
        if "ebitda" in lowered:
            return ["ebitda", "adjusted ebitda", "unadjusted ebitda"]
        if "net profit margin" in lowered or "net income" in lowered:
            return ["net income"]
        return []

    def _extract_first_claim_numeric_value(self, claim: Claim) -> Optional[float]:
        values = self._parse_claim_numeric_values(claim)
        return values[0] if values else None

    def _extract_claim_percentage_value(self, claim: Claim) -> Optional[float]:
        for raw in claim.extracted_numbers:
            if "%" not in raw:
                continue
            parsed = self._parse_numeric_literal(raw.replace("%", ""))
            if parsed is not None:
                return parsed
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", claim.text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _normalize_numeric_token(self, token: str) -> str:
        return token.lower().replace("$", "").replace(",", "").strip()

    def _normalize_numeric_haystack(self, text: str) -> set[str]:
        return {
            self._normalize_numeric_token(match.group(0))
            for match in NUMERIC_TOKEN_PATTERN.finditer(text or "")
            if match.group(0).strip()
        }

    def _extract_period_tokens(self, text: str) -> set[str]:
        return {
            match.group(0)
            for match in PERIOD_PATTERN.finditer(text or "")
        }

    def _extract_currency_tokens(self, text: str) -> set[str]:
        tokens = set()
        if "$" in (text or ""):
            tokens.add("$")
        for token in re.findall(r"\b(?:usd|dollars?)\b", text or "", re.IGNORECASE):
            tokens.add(token.lower())
        return tokens

    def _extract_unit_tokens(self, text: str) -> set[str]:
        units = {match.group(0).lower() for match in UNIT_PATTERN.finditer(text or "")}
        lowered = (text or "").lower()
        if "$" in (text or "") or "usd" in lowered or "dollar" in lowered:
            units.update({"$", "usd"})
        if "%" in (text or ""):
            units.update({"%", "percent"})
        if "percentage points" in lowered or "pts" in lowered:
            units.update({"pts", "percentage points"})
        return units

    def _extract_directionality_tokens(self, text: str) -> set[str]:
        lowered = (text or "").lower()
        return {token for token in self.DIRECTIONALITY_TOKENS if token in lowered}

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
