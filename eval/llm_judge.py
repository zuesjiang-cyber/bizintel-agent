import json
import logging
from typing import Any, Dict, List, Optional

from agent.config import settings
from agent.llm_utils import build_openai_client, generate_text_response, should_use_stub_llm

logger = logging.getLogger(__name__)

HARD_BLOCKING_ERROR_TAGS = {
    "missing_citation",
    "source_not_found",
    "fabricated_citation",
    "numeric_mismatch",
    "unit_mismatch",
    "period_mismatch",
    "direction_mismatch",
    "contradiction",
}
GRAY_ZONE_FAILURE_REASONS = {"low_entailment", "no_supporting_chunk", "primary_source_missing", None}
CLAIM_LABELS = {"strong_support", "weak_support", "unsupported"}
ANSWER_STATUS_LABELS = {"answered", "partial", "abstained"}


class LLMJudge:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        client: Any = None,
        model: Optional[str] = None,
    ) -> None:
        self.enabled = settings.llm_judge_enabled if enabled is None else enabled
        self._client = client
        self.model = model or settings.openai_model

    def is_available(self) -> bool:
        return bool(self.enabled) and not should_use_stub_llm(settings.llm_mode, settings.openai_api_key)

    def _get_client(self):
        if not self.is_available():
            return None
        if self._client is None:
            try:
                self._client = build_openai_client(settings.openai_api_key, settings.openai_api_base)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                logger.warning("LLM judge unavailable because client setup failed: %s", exc)
                self.enabled = False
                return None
        return self._client

    def _call_json(self, *, system_prompt: str, user_prompt: str) -> Optional[dict]:
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = generate_text_response(
                client,
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=250,
                temperature=0.0,
                max_retries=settings.llm_request_max_retries,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - depends on live API
            logger.warning("LLM judge request failed: %s", exc)
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM judge returned non-JSON payload: %s", raw[:300])
            return None

    def _collect_claim_evidence(self, claim_diagnostic: dict, evidence_store: Dict[str, List[Any]]) -> List[str]:
        snippets: List[str] = []
        seen = set()
        for text in claim_diagnostic.get("supporting_evidence") or []:
            cleaned = str(text).strip()
            if cleaned and cleaned not in seen:
                snippets.append(cleaned[:900])
                seen.add(cleaned)

        for source_id in claim_diagnostic.get("supporting_source_ids") or claim_diagnostic.get("cited_sources") or []:
            for chunk in (evidence_store.get(source_id) or [])[:2]:
                if isinstance(chunk, dict):
                    text = str(chunk.get("text", "")).strip()
                else:
                    text = str(chunk).strip()
                if not text or text in seen:
                    continue
                snippets.append(f"[Source: {source_id}] {text[:900]}")
                seen.add(text)
                if len(snippets) >= 4:
                    return snippets
        return snippets

    def should_review_claim(self, claim_diagnostic: dict) -> bool:
        if not self.is_available():
            return False
        if set(claim_diagnostic.get("error_tags", [])) & HARD_BLOCKING_ERROR_TAGS:
            return False
        if claim_diagnostic.get("support_label") == "strong_support":
            return False
        if claim_diagnostic.get("failure_reason") not in GRAY_ZONE_FAILURE_REASONS:
            return False
        if not claim_diagnostic.get("cited_sources") or claim_diagnostic.get("cited_sources") == ["no_citation"]:
            return False
        if claim_diagnostic.get("nli_score") is not None and float(claim_diagnostic.get("nli_score", 0.0)) > 0.75:
            return False
        return True

    def adjudicate_claim_support(
        self,
        *,
        question_text: str,
        claim_diagnostic: dict,
        evidence_store: Dict[str, List[Any]],
    ) -> Optional[dict]:
        if not self.should_review_claim(claim_diagnostic):
            return None
        evidence_snippets = self._collect_claim_evidence(claim_diagnostic, evidence_store)
        if not evidence_snippets:
            return None

        system_prompt = (
            "You are a conservative financial evidence judge. "
            "Never treat weak evidence as strong. "
            "Do not override hard numeric or citation failures."
        )
        user_prompt = json.dumps(
            {
                "task": "Classify whether the cited evidence strongly supports, weakly supports, or does not support the claim.",
                "question": question_text,
                "claim": claim_diagnostic.get("claim_text", ""),
                "rule_support_label": claim_diagnostic.get("support_label"),
                "rule_failure_reason": claim_diagnostic.get("failure_reason"),
                "rule_error_tags": claim_diagnostic.get("error_tags", []),
                "evidence_snippets": evidence_snippets,
                "allowed_labels": ["strong_support", "weak_support", "unsupported"],
                "instructions": [
                    "Return strong_support only when the evidence directly supports the claim.",
                    "Return weak_support when the evidence is related but not decisive.",
                    "Return unsupported when the evidence does not really support the claim.",
                    "Prefer conservative labels.",
                ],
                "output_schema": {"support_label": "label", "reason": "short explanation"},
            },
            ensure_ascii=False,
        )
        parsed = self._call_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not parsed:
            return None
        label = parsed.get("support_label")
        if label not in CLAIM_LABELS:
            return None
        return {"support_label": label, "reason": str(parsed.get("reason", "")).strip()}

    def should_review_answer_status(self, *, rule_status: str, metrics: dict, scorable_markdown: str) -> bool:
        if not self.is_available():
            return False
        if rule_status == "abstained":
            return bool(metrics.get("total_claims", 0) > 0)
        if rule_status == "partial":
            return bool(
                metrics.get("strong_support_rate", 0.0) >= 0.75
                and metrics.get("required_slot_coverage", 0.0) >= 0.75
                and "insufficient evidence" not in scorable_markdown.lower()
            )
        return False

    def adjudicate_answer_status(
        self,
        *,
        question_text: str,
        scorable_markdown: str,
        metrics: dict,
        rule_status: str,
    ) -> Optional[dict]:
        if not self.should_review_answer_status(rule_status=rule_status, metrics=metrics, scorable_markdown=scorable_markdown):
            return None
        system_prompt = (
            "You are a conservative financial QA judge. "
            "Classify whether the answer is answered, partial, or abstained. "
            "Use the answer text and metrics. Prefer partial over answered when there are major gaps."
        )
        user_prompt = json.dumps(
            {
                "question": question_text,
                "rule_status": rule_status,
                "metrics": {
                    "total_claims": metrics.get("total_claims", 0),
                    "strong_support_rate": metrics.get("strong_support_rate", 0.0),
                    "required_slot_coverage": metrics.get("required_slot_coverage", 0.0),
                    "fabricated_citation_rate": metrics.get("fabricated_citation_rate", 0.0),
                    "contradiction_rate": metrics.get("contradiction_rate", 0.0),
                },
                "answer_excerpt": scorable_markdown[:3000],
                "allowed_labels": ["answered", "partial", "abstained"],
                "instructions": [
                    "answered means the question is directly answered and usable.",
                    "partial means there is some usable answer but important gaps remain.",
                    "abstained means the answer mostly says insufficient evidence or gives no usable conclusion.",
                    "Do not mark answered when the answer is mostly caveats.",
                ],
                "output_schema": {"answer_status": "label", "reason": "short explanation"},
            },
            ensure_ascii=False,
        )
        parsed = self._call_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not parsed:
            return None
        label = parsed.get("answer_status")
        if label not in ANSWER_STATUS_LABELS:
            return None
        return {"answer_status": label, "reason": str(parsed.get("reason", "")).strip()}
