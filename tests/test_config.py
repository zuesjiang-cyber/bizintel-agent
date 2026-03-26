from agent.config import Settings


def test_minimax_env_aliases_are_supported(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "MiniMax-M2.7")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "test-key"
    assert settings.openai_api_base == "https://api.minimaxi.com/anthropic"
    assert settings.openai_model == "MiniMax-M2.7"


def test_workflow_timeout_env_aliases_are_supported(monkeypatch):
    monkeypatch.setenv("WORKFLOW_TIMEOUT_SECONDS", "420")

    settings = Settings(_env_file=None)

    assert settings.default_timeout_seconds == 420
