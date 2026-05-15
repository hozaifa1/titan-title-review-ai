"""Centralized application settings.

Single ``Settings`` object loaded from environment variables (and ``.env``)
via ``pydantic-settings``. Modules should accept a ``Settings`` instance
through dependency injection rather than reading ``os.environ`` directly.

A module-level :func:`get_settings` returns a cached instance so callers
that have not been refactored yet can still access the same values without
a global mutable state.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Titan pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM / Gemini
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_api_key_fallback: str | None = Field(default=None, alias="GOOGLE_API_KEY_FALLBACK")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # Alternative LLM providers (used as fallback when Gemini is unavailable/rate-limited)
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="deepseek/deepseek-chat-v3.1:free", alias="OPENROUTER_MODEL")
    # Cerebras (60K tokens/min free, ~1,700 req/day) — OpenAI-compatible. Sign up: https://cloud.cerebras.ai/
    cerebras_api_key: str | None = Field(default=None, alias="CEREBRAS_API_KEY")
    cerebras_model: str = Field(default="llama3.3-70b", alias="CEREBRAS_MODEL")
    # SambaNova (DeepSeek R1 + Llama, fast) — OpenAI-compatible. Sign up: https://cloud.sambanova.ai/
    sambanova_api_key: str | None = Field(default=None, alias="SAMBANOVA_API_KEY")
    sambanova_model: str = Field(default="Meta-Llama-3.3-70B-Instruct", alias="SAMBANOVA_MODEL")
    # GitHub Models (free with any GitHub PAT) — 150 req/day for high-tier models. https://github.com/settings/tokens
    github_models_token: str | None = Field(default=None, alias="GITHUB_MODELS_TOKEN")
    github_models_model: str = Field(default="meta/Meta-Llama-3.3-70B-Instruct", alias="GITHUB_MODELS_MODEL")
    # Ordered preference. Cerebras leads because it has the highest throughput allowance.
    llm_provider_order: str = Field(
        default="cerebras,sambanova,groq,gemini,github,openrouter",
        alias="TITAN_LLM_PROVIDER_ORDER",
    )
    # Global concurrency cap across all providers. Prevents bursting past per-minute limits
    # when 8 ALTA sections × N documents fan out in parallel.
    llm_max_concurrency: int = Field(default=3, alias="TITAN_LLM_MAX_CONCURRENCY")

    # Vector store
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Observability
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")

    # Optional providers
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    docling_cache_dir: Path = Field(default=Path("./cache/docling"), alias="DOCLING_CACHE_DIR")

    # Behavioural toggles
    use_local_models: bool = Field(default=False, alias="TITAN_LOCAL_MODELS")
    log_level: str = Field(default="INFO", alias="TITAN_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="TITAN_LOG_JSON")

    # Retry / external-call policy
    max_retries: int = Field(default=3, alias="TITAN_MAX_RETRIES")
    retry_max_wait_seconds: int = Field(default=8, alias="TITAN_RETRY_MAX_WAIT")

    # Filesystem defaults
    sqlite_path: Path = Field(default=Path("data/titan.db"), alias="TITAN_SQLITE_PATH")
    rules_dir: Path = Field(default=Path("rules"), alias="TITAN_RULES_DIR")

    @property
    def gemini_key(self) -> str | None:
        """Resolve a single Gemini key, preferring ``GOOGLE_API_KEY``."""
        return self.google_api_key or self.gemini_api_key

    @property
    def gemini_keys(self) -> list[str]:
        """All Gemini keys in fallback order (primary then fallback)."""
        keys = [k for k in (self.google_api_key, self.google_api_key_fallback, self.gemini_api_key) if k]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                unique.append(key)
        return unique

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_cerebras(self) -> bool:
        # Treat literal placeholder strings as absent so a copy-pasted .env
        # template doesn't accidentally route traffic at a 404 endpoint.
        return bool(self.cerebras_api_key and self.cerebras_api_key.strip() and not self.cerebras_api_key.startswith("<"))

    @property
    def has_sambanova(self) -> bool:
        return bool(self.sambanova_api_key and self.sambanova_api_key.strip() and not self.sambanova_api_key.startswith("<"))

    @property
    def has_github_models(self) -> bool:
        return bool(self.github_models_token and self.github_models_token.strip() and not self.github_models_token.startswith("<"))

    @property
    def has_any_llm(self) -> bool:
        return (
            self.has_gemini
            or self.has_groq
            or self.has_openrouter
            or self.has_cerebras
            or self.has_sambanova
            or self.has_github_models
        )

    @property
    def provider_chain(self) -> list[str]:
        """Ordered list of providers to try, filtered to those with keys configured."""
        wanted = [p.strip().lower() for p in self.llm_provider_order.split(",") if p.strip()]
        availability = {
            "gemini": self.has_gemini,
            "groq": self.has_groq,
            "openrouter": self.has_openrouter,
            "cerebras": self.has_cerebras,
            "sambanova": self.has_sambanova,
            "github": self.has_github_models,
        }
        return [p for p in wanted if availability.get(p, False)]

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def reload_settings() -> Settings:
    """Force a fresh Settings load (useful in tests after env mutation)."""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["Settings", "get_settings", "reload_settings"]
