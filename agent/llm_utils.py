import json
import logging
import re
import time
from typing import Iterable, List, Sequence

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
except ImportError:  # pragma: no cover - optional in demo-only mode
    APIConnectionError = None
    APIStatusError = None
    APITimeoutError = None
    OpenAI = None

from agent.schemas import AnalysisStep, RetrievedChunk

logger = logging.getLogger(__name__)


def normalize_api_key(api_key: str) -> str:
    return (api_key or "").strip()


def normalize_openai_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def should_use_stub_llm(llm_mode: str, api_key: str) -> bool:
    normalized = (llm_mode or "auto").strip().lower()
    if normalized == "stub":
        return True
    if normalized == "live":
        return False
    cleaned = normalize_api_key(api_key)
    if not cleaned:
        return True
    if not cleaned.isascii():
        return True
    if any(char.isspace() for char in cleaned):
        return True
    return False


def build_openai_client(api_key: str, base_url: str):
    cleaned = normalize_api_key(api_key)
    if not cleaned:
        raise RuntimeError("Missing API key for OpenAI-compatible client.")
    if OpenAI is None:
        raise RuntimeError("openai package is not installed; run make install to add runtime dependencies.")

    return OpenAI(
        api_key=cleaned,
        base_url=normalize_openai_base_url(base_url),
        max_retries=0,
    )


def _summarize_error_body(body: object, *, max_chars: int = 300) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        text = body
    else:
        try:
            text = json.dumps(body, ensure_ascii=True, sort_keys=True)
        except TypeError:
            text = repr(body)
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _request_url_for_error(err: Exception) -> str:
    request = getattr(err, "request", None)
    if request is None:
        response = getattr(err, "response", None)
        request = getattr(response, "request", None)
    url = getattr(request, "url", None)
    return str(url) if url else "<unknown>"


def _log_llm_request_error(err: Exception, *, attempt: int, total_attempts: int) -> None:
    if APIStatusError is not None and isinstance(err, APIStatusError):
        response = getattr(err, "response", None)
        status_code = getattr(response, "status_code", None)
        body_summary = _summarize_error_body(getattr(err, "body", None)) or "<empty>"
        logger.warning(
            "LLM request failed (attempt %s/%s): status=%s error_type=%s url=%s body=%s",
            attempt,
            total_attempts,
            status_code,
            type(err).__name__,
            _request_url_for_error(err),
            body_summary,
        )
        return

    if APITimeoutError is not None and isinstance(err, APITimeoutError):
        logger.warning(
            "LLM request timed out (attempt %s/%s): error_type=%s url=%s",
            attempt,
            total_attempts,
            type(err).__name__,
            _request_url_for_error(err),
        )
        return

    if APIConnectionError is not None and isinstance(err, APIConnectionError):
        logger.warning(
            "LLM request connection error (attempt %s/%s): error_type=%s url=%s detail=%s",
            attempt,
            total_attempts,
            type(err).__name__,
            _request_url_for_error(err),
            str(err),
        )
        return

    logger.warning(
        "LLM request failed (attempt %s/%s): error_type=%s detail=%s",
        attempt,
        total_attempts,
        type(err).__name__,
        str(err),
    )


def _retry_delay_seconds(attempt: int) -> float:
    return min(0.5 * (2 ** (attempt - 1)), 8.0)


def extract_text_from_completion(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_blocks: List[str] = []
        for block in content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_blocks.append(block.text)
        return "\n".join(text_blocks).strip()
    return ""


def generate_text_response(
    client,
    *,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> str:
    system_chars = len(system_prompt or "")
    user_chars = len(user_prompt or "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    logger.info(
        "LLM request start: model=%s system_chars=%s user_chars=%s total_chars=%s max_tokens=%s temperature=%s",
        model,
        system_chars,
        user_chars,
        system_chars + user_chars,
        max_tokens,
        temperature,
    )

    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.chat.completions.create(**payload)
            return extract_text_from_completion(response)
        except Exception as err:
            _log_llm_request_error(err, attempt=attempt, total_attempts=total_attempts)
            if attempt == total_attempts:
                raise
            delay = _retry_delay_seconds(attempt)
            logger.info("Retrying LLM request in %.2f seconds", delay)
            time.sleep(delay)

    raise RuntimeError("LLM request loop exited without returning a response.")


def build_stub_section_analysis(
    step: AnalysisStep,
    user_query: str,
    chunks: List[RetrievedChunk],
    previous_findings: str = "",
) -> str:
    if not chunks:
        return (
            f"No strong evidence was retrieved for `{step.name}` while analyzing "
            f"\"{user_query}\". Additional sources are needed before making a confident claim."
        )

    lines: List[str] = []
    seen_sentences = set()
    scored_candidates = []

    for chunk_rank, chunk in enumerate(chunks):
        for sentence_rank, sentence in enumerate(_extract_evidence_sentences(chunk.text)):
            score = _score_sentence(sentence, chunk.source_id, step, user_query)
            scored_candidates.append((score, -chunk_rank, -sentence_rank, sentence, chunk.source_id))

    for _, _, _, sentence, source_id in sorted(scored_candidates, reverse=True):
        normalized = sentence.lower()
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        lines.append(f"- {_ensure_terminal_punctuation(sentence)} [Chunk: {chunk.chunk_id}] [Source: {source_id}]")
        if len(lines) >= 3:
            break

    if not lines:
        snippet = _clean_snippet(chunks[0].text)
        lines.append(f"- {_ensure_terminal_punctuation(snippet)} [Chunk: {chunks[0].chunk_id}] [Source: {chunks[0].source_id}]")

    return "\n".join(lines)


def build_stub_executive_summary(section_texts: Iterable[str]) -> str:
    cleaned_sections = [text.strip() for text in section_texts if text and text.strip()]
    if not cleaned_sections:
        return "No analysis content was generated."

    evidence_lines: List[str] = []
    for text in cleaned_sections:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and (
                ("[Source:" in stripped and "[Chunk:" in stripped)
                or "insufficient evidence" in stripped.lower()
            ):
                evidence_lines.append(stripped[2:])
            if len(evidence_lines) >= 3:
                break
        if len(evidence_lines) >= 3:
            break

    if not evidence_lines:
        return "Insufficient evidence in the source pack to support an executive summary."

    bullets = []
    for line in evidence_lines[:3]:
        cleaned = _ensure_terminal_punctuation(line)
        bullets.append(f"- {cleaned}")
    return "\n".join(bullets)


def _extract_evidence_sentences(text: str) -> Sequence[str]:
    cleaned = _clean_snippet(text, max_chars=600)
    if not cleaned:
        return []

    profile_sentences = _profile_blob_to_sentences(cleaned)
    if len(profile_sentences) >= 3:
        return profile_sentences

    candidates = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        normalized = _normalize_sentence(sentence)
        if len(normalized) < 20:
            continue
        candidates.append(normalized)

    if candidates:
        return candidates

    return [cleaned]


def _profile_blob_to_sentences(text: str) -> List[str]:
    field_keys = [
        "Company",
        "Founded",
        "Headquarters",
        "Founders",
        "Employees",
        "Total Funding",
        "Latest Round",
        "Latest Valuation",
        "Industry",
        "Key Products",
        "Target Customers",
        "Core Revenue Model",
        "Core Customers",
        "Key Monitorables",
        "Memo Angle",
    ]
    key_pattern = "|".join(re.escape(key) for key in field_keys)
    parts = re.findall(rf"({key_pattern}):\s*(.*?)(?=\s+(?:{key_pattern}):|$)", text)
    sentences: List[str] = []
    for raw_key, raw_value in parts:
        key = raw_key.strip().lower()
        value = " ".join(raw_value.strip().split())
        if not value:
            continue
        if key == "company":
            sentences.append(f"The company is {value}")
        elif key == "founded":
            sentences.append(f"It was founded in {value}")
        elif key == "headquarters":
            sentences.append(f"It is headquartered in {value}")
        elif key == "founders":
            sentences.append(f"The founders are {value}")
        elif key == "total funding":
            sentences.append(f"It has raised {value} in total funding")
        elif key == "latest round":
            sentences.append(f"The latest funding round was {value}")
        elif key == "latest valuation":
            sentences.append(f"The latest valuation is {value}")
        elif key == "industry":
            sentences.append(f"It operates in {value}")
        elif key == "key products":
            sentences.append(f"Key products include {value}")
        elif key == "target customers":
            sentences.append(f"Target customers include {value}")
        elif key == "core revenue model":
            sentences.append(f"The core revenue model is {value}")
        elif key == "core customers":
            sentences.append(f"Core customers include {value}")
        elif key == "key monitorables":
            sentences.append(f"Key monitorables include {value}")
        elif key == "memo angle":
            sentences.append(f"The memo angle is {value}")
        else:
            sentences.append(f"{raw_key.strip()}: {value}")
    return sentences


def _score_sentence(sentence: str, source_id: str, step: AnalysisStep, user_query: str) -> int:
    sentence_tokens = set(re.findall(r"\b[a-z0-9]+\b", sentence.lower()))
    query_tokens = set(re.findall(r"\b[a-z0-9]+\b", user_query.lower()))
    step_tokens = set(re.findall(r"\b[a-z0-9]+\b", f"{step.name} {step.description}".lower()))
    priority_tokens = set(_step_priority_tokens(step.name))

    score = len(sentence_tokens & query_tokens) * 2
    score += len(sentence_tokens & step_tokens) * 3
    score += len(sentence_tokens & priority_tokens) * 4
    score += _source_priority_bonus(source_id, step.name)

    if any(char.isdigit() for char in sentence):
        score += 1
    if _is_generic_company_sentence(sentence):
        score -= 4
        if step.name in {"company_overview", "business_model"}:
            score += 2

    return score


def _step_priority_tokens(step_name: str) -> List[str]:
    mapping = {
        "company_overview": ["founded", "headquartered", "founders", "company"],
        "business_model": ["products", "customers", "platform", "revenue"],
        "financial_analysis": ["funding", "valuation", "revenue", "payment"],
        "financial_quality": ["revenue", "margin", "cash", "quality"],
        "competitive_landscape": ["industry", "customers", "products", "competitors"],
        "investment_takeaway": ["valuation", "funding", "customers", "products"],
        "risks_and_outlook": ["industry", "regulatory", "valuation", "growth"],
    }
    return mapping.get(step_name, [])


def _source_priority_bonus(source_id: str, step_name: str) -> int:
    source_id = source_id.lower()
    mapping = {
        "company_overview": {"profile": 4, "about": 3},
        "business_model": {"revenue_quality": 5, "profile": 3, "about": 2, "competitors": 1},
        "financial_analysis": {"revenue": 5, "funding": 5, "valuation": 4, "monitorables": 3, "profile": 1},
        "financial_quality": {"revenue_quality": 6, "revenue": 4, "risks": 2, "profile": 1},
        "competitive_landscape": {"competitors": 6, "industry": 3, "revenue_quality": 1},
        "investment_takeaway": {"valuation": 6, "monitorables": 5, "funding": 4, "revenue_quality": 3, "risks": 3},
        "risks_and_outlook": {"risks": 6, "industry": 4, "valuation": 2},
    }
    total = 0
    for key, bonus in mapping.get(step_name, {}).items():
        if key in source_id:
            total += bonus
    return total


def _is_generic_company_sentence(sentence: str) -> bool:
    normalized = sentence.lower()
    generic_patterns = [
        "financial infrastructure platform",
        "millions of companies use stripe",
        "increase the gdp of the internet",
        "the company is stripe",
        "source is most useful for company description",
    ]
    return any(pattern in normalized for pattern in generic_patterns)


def _normalize_sentence(sentence: str) -> str:
    stripped = sentence.strip(" -")
    stripped = re.sub(r"^(Observation|Implication|Risk|Watch items?|Bottom line):\s*", "", stripped, flags=re.IGNORECASE)
    return stripped


def _clean_snippet(text: str, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[Chunk:\s*[^\]]+\]", "", text).strip()
    text = re.sub(r"\[Source:\s*[^\]]+\]", "", text).strip()
    text = re.sub(r"^Source:\s*https?://\S+\\nDate:\s*.*?\\nTitle:\s*.*?\\n\\n", "", text).strip()
    text = re.sub(r"^Source:\s*https?://\S+\s+Date:\s+.*?\s+Title:\s+.*?\s+", "", text).strip()
    if len(text) <= max_chars:
        return text

    cutoff = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{cutoff}..."


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.endswith((".", "!", "?")):
        return stripped
    return f"{stripped}."
