"""Smoke tests for titan.config."""

from __future__ import annotations

import pytest

from titan.config import Settings, get_settings, reload_settings


def test_settings_loads_with_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear every key the test cares about and disable .env loading so
    # a developer's local credentials don't leak into the assertions.
    for key in (
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEY_FALLBACK",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
        "SAMBANOVA_API_KEY",
        "GITHUB_MODELS_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "QDRANT_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(settings, Settings)
    assert settings.has_any_llm is False
    assert settings.has_gemini is False
    assert settings.langfuse_configured is False


def test_provider_chain_filters_to_configured() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        GOOGLE_API_KEY="test-key",
        GROQ_API_KEY="groq-key",
    )
    chain = settings.provider_chain
    assert "gemini" in chain
    assert "groq" in chain
    assert "cerebras" not in chain


def test_placeholder_keys_are_treated_as_absent() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        CEREBRAS_API_KEY="<your-cerebras-key>",
    )
    assert settings.has_cerebras is False


def test_get_settings_is_cached() -> None:
    reload_settings()
    a = get_settings()
    b = get_settings()
    assert a is b
