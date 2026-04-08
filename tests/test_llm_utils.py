import json
import logging
from types import SimpleNamespace

import httpx
from openai import APIStatusError

from agent import llm_utils
from agent.llm_utils import (
    build_openai_client,
    build_stub_executive_summary,
    build_stub_section_analysis,
    extract_text_from_completion,
    generate_text_response,
    normalize_api_key,
    normalize_openai_base_url,
    should_use_stub_llm,
    strip_hidden_reasoning,
)
from agent.schemas import AnalysisStep, RetrievedChunk


def test_should_use_stub_llm_for_missing_key():
    assert should_use_stub_llm("auto", "") is True


def test_should_use_stub_llm_for_non_ascii_key():
    assert should_use_stub_llm("auto", "你从 MiniMax 控制台重新复制的一整串 ASCII key") is True


def test_should_use_stub_llm_for_ascii_key_without_spaces():
    assert should_use_stub_llm("auto", "sk-test-ascii-key") is False


def test_normalize_api_key_strips_whitespace():
    assert normalize_api_key("  test-key  ") == "test-key"


def test_normalize_openai_base_url_keeps_v1_path():
    assert normalize_openai_base_url("https://callflow.top/v1/") == "https://callflow.top/v1"


def test_build_openai_client_passes_api_key_and_base_url(monkeypatch):
    class DummyOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_utils, "OpenAI", DummyOpenAIClient)

    client = build_openai_client(" test-key ", "https://callflow.top/v1/")

    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["base_url"] == "https://callflow.top/v1"
    assert client.kwargs["max_retries"] == 0
    assert client.kwargs["timeout"] == llm_utils.settings.llm_request_timeout_seconds


def test_extract_text_from_completion_reads_string_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="hello world")),
        ]
    )

    assert extract_text_from_completion(response) == "hello world"


def test_extract_text_from_completion_parses_sse_chunk_string_and_strips_think():
    events = [
        {"choices": [{"delta": {"content": "<think>hidden</think>"}}]},
        {"choices": [{"delta": {"content": '{"answers":['}}]},
        {"choices": [{"delta": {"content": '{"question_id":"q1"}'}}]},
        {"choices": [{"delta": {"content": "]}"}}]},
    ]
    response = "\n\n".join(f"data: {json.dumps(event)}" for event in events)

    assert extract_text_from_completion(response) == '{"answers":[{"question_id":"q1"}]}'


def test_extract_text_from_completion_strips_think_blocks_from_string_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="<think>internal reasoning</think>\nOK")),
        ]
    )

    assert extract_text_from_completion(response) == "OK"


def test_extract_text_from_completion_reads_text_blocks():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        SimpleNamespace(type="reasoning", text="hidden"),
                        SimpleNamespace(type="text", text="hello"),
                        SimpleNamespace(type="text", text="world"),
                    ]
                )
            )
        ]
    )

    assert extract_text_from_completion(response) == "hello\nworld"


def test_strip_hidden_reasoning_removes_standalone_think_tags():
    assert strip_hidden_reasoning("<think>\nplan\n</think>\nFinal answer") == "Final answer"


def test_generate_text_response_uses_chat_completions_api(caplog):
    captured = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="<think>hidden</think>\nok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))
    with caplog.at_level(logging.INFO):
        result = generate_text_response(
            client,
            model="MiniMax-M2.7",
            system_prompt="You are helpful.",
            user_prompt="Hi",
            max_tokens=123,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

    assert result == "ok"
    assert captured["model"] == "MiniMax-M2.7"
    assert captured["max_tokens"] == 123
    assert captured["temperature"] == 0.3
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert captured["messages"][1] == {"role": "user", "content": "Hi"}
    assert "LLM request start" in caplog.text
    assert "total_chars=18" in caplog.text


def test_generate_text_response_logs_status_and_body_before_retry(monkeypatch, caplog):
    request = httpx.Request("POST", "https://callflow.top/v1/chat/completions")
    response = httpx.Response(429, request=request)
    error = APIStatusError(
        "rate limited",
        response=response,
        body={"error": {"message": "too many requests", "type": "rate_limit"}},
    )
    calls = {"count": 0}

    class DummyCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm_utils.time, "sleep", lambda _: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    with caplog.at_level(logging.INFO):
        result = generate_text_response(
            client,
            model="MiniMax-M2.7",
            user_prompt="Hi",
            max_retries=1,
        )

    assert result == "ok"
    assert calls["count"] == 2
    assert "status=429" in caplog.text
    assert "too many requests" in caplog.text
    assert "https://callflow.top/v1/chat/completions" in caplog.text


def test_build_stub_executive_summary_skips_uncited_inference_lines():
    summary = build_stub_executive_summary(
        [
            "- This inference has no citation.\n- Cloudflare is a connectivity cloud company [Chunk: c1] [Source: cloudflare_10k]",
            "- Insufficient evidence to verify monetization mix.",
        ]
    )

    assert "This inference has no citation" not in summary
    assert summary.startswith("- ")
    assert "[Source: cloudflare_10k]" in summary


def test_build_stub_executive_summary_returns_insufficient_evidence_without_cited_lines():
    summary = build_stub_executive_summary(
        [
            "- This inference has no citation.",
            "Another uncited paragraph.",
        ]
    )

    assert summary == "Insufficient evidence in the source pack to support an executive summary."


def test_build_stub_section_analysis_preserves_chunk_source_pairing():
    content = build_stub_section_analysis(
        AnalysisStep(name="revenue", description="Revenue evidence", required=True),
        "Assess revenue evidence",
        [
            RetrievedChunk(
                chunk_id="c1",
                text="Revenue was $10 million.",
                source_id="s1",
                page=None,
                score=1.0,
                bm25_rank=0,
                dense_rank=0,
                rerank_score=1.0,
            ),
            RetrievedChunk(
                chunk_id="c2",
                text="Management outlook was stable.",
                source_id="s2",
                page=None,
                score=0.5,
                bm25_rank=1,
                dense_rank=1,
                rerank_score=0.5,
            ),
        ],
    )

    assert "[Chunk: c1] [Source: s1]" in content
