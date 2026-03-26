import logging
from types import SimpleNamespace

import httpx
from openai import APIStatusError

from agent import llm_utils
from agent.llm_utils import (
    build_openai_client,
    build_stub_executive_summary,
    extract_text_from_completion,
    generate_text_response,
    normalize_api_key,
    normalize_openai_base_url,
    should_use_stub_llm,
)


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


def test_extract_text_from_completion_reads_string_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="hello world")),
        ]
    )

    assert extract_text_from_completion(response) == "hello world"


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


def test_generate_text_response_uses_chat_completions_api(caplog):
    captured = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))
    with caplog.at_level(logging.INFO):
        result = generate_text_response(
            client,
            model="MiniMax-M2.7",
            system_prompt="You are helpful.",
            user_prompt="Hi",
            max_tokens=123,
            temperature=0.3,
        )

    assert result == "ok"
    assert captured["model"] == "MiniMax-M2.7"
    assert captured["max_tokens"] == 123
    assert captured["temperature"] == 0.3
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
