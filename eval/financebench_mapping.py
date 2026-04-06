"""
Helpers for evaluating FinanceBench-style gold answers inside BizIntel's benchmark harness.
"""

from __future__ import annotations

import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "based",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.\-]+", " ", (text or "").lower())).strip()


def _strip_citations(text: str | None) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\[Chunk:\s*[^\]]+\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[Source:\s*[^\]]+\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"`+", " ", cleaned)
    return cleaned


def _semantic_tokens(text: str | None) -> list[str]:
    normalized = _normalize(_strip_citations(text))
    if not normalized:
        return []
    tokens = []
    for token in normalized.split():
        if len(token) <= 2:
            continue
        if token in STOPWORDS:
            continue
        if re.fullmatch(r"[\d.$%\-]+", token):
            continue
        tokens.append(token)
    return tokens


def _token_f1(expected_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not expected_tokens or not candidate_tokens:
        return 0.0
    expected_set = set(expected_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(expected_set & candidate_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_set)
    recall = overlap / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _token_recall(expected_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not expected_tokens or not candidate_tokens:
        return 0.0
    expected_set = set(expected_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(expected_set & candidate_set)
    if overlap == 0:
        return 0.0
    return overlap / len(expected_set)


def _candidate_segments(text: str | None) -> list[str]:
    cleaned = _strip_citations(text)
    pieces = re.split(r"[\n\r]+|(?<=[.!?])\s+", cleaned)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def score_gold_answer_mapping(markdown: str, answer_row: dict) -> dict:
    mapping = answer_row.get("evaluation_mapping") or {}
    if not mapping:
        return {
            "gold_answer_mode": "unmapped",
            "gold_answer_hit": 0.0,
            "gold_numeric_hit": 0.0,
            "gold_citation_hit": 0.0,
            "gold_semantic_similarity": 0.0,
            "gold_semantic_hit": 0.0,
        }

    markdown_normalized = _normalize(markdown)
    expected_answer = _normalize(mapping.get("expected_answer"))
    gold_answer_hit = 1.0 if expected_answer and expected_answer in markdown_normalized else 0.0

    numeric_tokens = mapping.get("normalized_answer_tokens") or []
    if numeric_tokens:
        gold_numeric_hit = 1.0 if all(token.lower() in markdown.lower() for token in numeric_tokens) else 0.0
    else:
        gold_numeric_hit = 1.0

    required_doc_id = mapping.get("required_doc_id")
    gold_citation_hit = 1.0 if required_doc_id and f"[Source: {required_doc_id}]" in markdown else 0.0

    expected_semantic_tokens = _semantic_tokens(
        " ".join(
            part
            for part in [
                mapping.get("expected_answer"),
                mapping.get("expected_justification"),
            ]
            if part
        )
    )
    segments = _candidate_segments(markdown)
    if not expected_semantic_tokens:
        semantic_similarity = 1.0 if gold_answer_hit or gold_numeric_hit else 0.0
    else:
        semantic_similarity = 0.0
        for segment in segments:
            candidate_tokens = _semantic_tokens(segment)
            semantic_similarity = max(
                semantic_similarity,
                _token_f1(expected_semantic_tokens, candidate_tokens),
                _token_recall(expected_semantic_tokens, candidate_tokens),
            )
    if gold_answer_hit == 1.0:
        semantic_similarity = 1.0
    gold_semantic_hit = 1.0 if semantic_similarity >= 0.35 else 0.0

    return {
        "gold_answer_mode": mapping.get("mode", "unmapped"),
        "gold_answer_hit": gold_answer_hit,
        "gold_numeric_hit": gold_numeric_hit,
        "gold_citation_hit": gold_citation_hit,
        "gold_semantic_similarity": semantic_similarity,
        "gold_semantic_hit": gold_semantic_hit,
    }
