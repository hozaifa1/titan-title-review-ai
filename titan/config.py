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
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

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
    def has_gemini(self) -> bool:
        return bool(self.gemini_key)

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
