"""Unified LLM client with multi-provider fallback.

Wraps Gemini (primary), Groq (fast Llama 3.3 70B fallback), and OpenRouter
(catch-all). Each ``generate`` call walks the provider chain configured in
``Settings.provider_chain``: if the primary returns a quota/rate-limit error
the next provider is tried automatically.

This is the only module that should import provider-specific SDKs. Every
other titan module talks to ``LLMClient`` through ``generate_text`` or
``generate_json``.

Rationale for the fallback chain in May 2026:
  * Gemini Flash free tier = 1,500 req/day, easily exhausted in a single eval.
  * Groq's Llama 3.3 70B = ~14,400 req/day, ~30 req/min, very fast.
  * OpenRouter's free DeepSeek V3 = unlimited-ish, slower but a true backstop.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from titan.config import Settings, get_settings
from titan.errors import LLMUnavailableError
from titan.telemetry import get_logger

log = get_logger(__name__)


_RATE_LIMIT_PATTERNS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "resource_exhausted",
    "insufficient_quota",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(pattern in text for pattern in _RATE_LIMIT_PATTERNS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a Retry-After value (seconds) from an httpx error if present.

    Both Groq and OpenAI-compatible servers return ``Retry-After`` on 429s.
    The value is either an integer seconds count or an HTTP-date; we only
    handle the integer/float form.
    """

    response = getattr(exc, "response", None)
    if response is None:
        return None
    retry_after = response.headers.get("retry-after") if hasattr(response, "headers") else None
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LLMResult:
    """Outcome of a single ``LLMClient.generate`` call."""

    text: str
    provider: str
    model: str


class LLMClient:
    """Provider-agnostic LLM client with automatic fallback.

    Use ``generate_text`` for free-form text or ``generate_json`` when you need
    a parsed dict. Both walk the provider chain (Gemini → Groq → OpenRouter)
    and only raise ``LLMUnavailableError`` if every configured provider fails.
    """

    # Class-level semaphore so concurrent callers share one budget across the
    # whole process. Lazily created on first use to honour the configured cap.
    _global_sem: asyncio.Semaphore | None = None
    _global_sem_size: int = 0

    @classmethod
    def _acquire_semaphore(cls, size: int) -> asyncio.Semaphore:
        if cls._global_sem is None or cls._global_sem_size != size:
            cls._global_sem = asyncio.Semaphore(size)
            cls._global_sem_size = size
        return cls._global_sem

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sem = self._acquire_semaphore(self.settings.llm_max_concurrency)

    @property
    def has_any(self) -> bool:
        return self.settings.has_any_llm

    async def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        response_json: bool = False,
        max_attempts_per_provider: int = 2,
    ) -> LLMResult:
        """Return raw text from the first provider that responds successfully.

        Calls are gated by ``_global_sem`` so concurrent batch operations
        (8-section drafts × N docs) don't burst past per-minute caps. On
        ``429`` we honour the provider's ``Retry-After`` header where
        possible — usually the difference between recoverable and dead.
        """

        providers = self.settings.provider_chain
        if not providers:
            raise LLMUnavailableError("No LLM provider configured")

        async with self._sem:
            return await self._generate_text_inner(
                prompt, temperature, response_json, max_attempts_per_provider, providers
            )

    async def _generate_text_inner(
        self,
        prompt: str,
        temperature: float,
        response_json: bool,
        max_attempts_per_provider: int,
        providers: list[str],
    ) -> LLMResult:
        last_exc: Exception | None = None
        for provider in providers:
            for attempt in range(1, max_attempts_per_provider + 1):
                try:
                    if provider == "gemini":
                        return await self._call_gemini(prompt, temperature, response_json)
                    if provider == "groq":
                        return await self._call_groq(prompt, temperature, response_json)
                    if provider == "cerebras":
                        return await self._call_cerebras(prompt, temperature, response_json)
                    if provider == "sambanova":
                        return await self._call_sambanova(prompt, temperature, response_json)
                    if provider == "github":
                        return await self._call_github_models(prompt, temperature, response_json)
                    if provider == "openrouter":
                        return await self._call_openrouter(prompt, temperature, response_json)
                except Exception as exc:  # pragma: no cover - exercised via integration tests
                    last_exc = exc
                    rate_limited = _is_rate_limit_error(exc)
                    retry_after = _retry_after_seconds(exc)
                    log.warning(
                        "llm.provider_failed",
                        provider=provider,
                        attempt=attempt,
                        rate_limited=rate_limited,
                        retry_after=retry_after,
                        error=str(exc)[:200],
                    )
                    if rate_limited:
                        # If the server told us when to retry AND it's within a
                        # sane budget, wait it out instead of failing the chain.
                        if retry_after is not None and retry_after <= 8 and attempt < max_attempts_per_provider:
                            await asyncio.sleep(retry_after + 0.25)
                            continue
                        break  # move to next provider
                    if attempt < max_attempts_per_provider:
                        await asyncio.sleep(1.5 * attempt)
                        continue
            log.warning("llm.provider_exhausted", provider=provider)

        raise LLMUnavailableError(
            f"All providers in chain {providers} failed; last error: {last_exc}"
        ) from last_exc

    async def generate_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_attempts_per_provider: int = 2,
    ) -> tuple[dict[str, Any], LLMResult]:
        """Return a parsed JSON object plus the raw provider/model used.

        The prompt is sent with provider-native JSON-mode hints when available
        and the response text is tolerantly extracted (handles fenced blocks
        and prefix text from models that ignore the JSON-only instruction).
        """

        result = await self.generate_text(
            prompt, temperature=temperature, response_json=True,
            max_attempts_per_provider=max_attempts_per_provider,
        )
        data = _coerce_json_object(result.text)
        return data, result

    async def _call_gemini(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        keys = self.settings.gemini_keys
        if not keys:
            raise LLMUnavailableError("Gemini key missing")
        import google.generativeai as genai  # type: ignore[import-untyped]

        last_exc: Exception | None = None
        config: dict[str, Any] = {"temperature": temperature}
        if response_json:
            config["response_mime_type"] = "application/json"

        for key in keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(self.settings.gemini_model)
                response = await asyncio.to_thread(
                    model.generate_content, prompt, generation_config=config
                )
                text = _gemini_response_text(response)
                if not text:
                    raise RuntimeError("Gemini returned empty text")
                return LLMResult(text=text, provider="gemini", model=self.settings.gemini_model)
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc):
                    raise
                log.info("llm.gemini_key_exhausted_trying_fallback", error=str(exc)[:140])
                continue

        assert last_exc is not None
        raise last_exc

    async def _call_groq(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        api_key = self.settings.groq_api_key
        if not api_key:
            raise LLMUnavailableError("Groq key missing")
        model = self.settings.groq_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Groq returned empty content")
        return LLMResult(text=text, provider="groq", model=model)

    async def _call_cerebras(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        """Cerebras inference (OpenAI-compatible). ~60K tokens/min on free tier."""

        api_key = self.settings.cerebras_api_key
        # Reject empty / placeholder keys so a misconfigured chain doesn't burn
        # attempts hitting a 404 endpoint with an Authorization header that
        # Cerebras will silently 404 rather than 401.
        if not api_key or not api_key.strip() or api_key.startswith("<") or len(api_key.strip()) < 16:
            raise LLMUnavailableError("Cerebras key missing")
        model = self.settings.cerebras_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Cerebras returned empty content")
        return LLMResult(text=text, provider="cerebras", model=model)

    async def _call_sambanova(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        """SambaNova inference (OpenAI-compatible). Fast, DeepSeek R1 + Llama 3.3."""

        api_key = self.settings.sambanova_api_key
        if not api_key:
            raise LLMUnavailableError("SambaNova key missing")
        model = self.settings.sambanova_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("SambaNova returned empty content")
        return LLMResult(text=text, provider="sambanova", model=model)

    async def _call_github_models(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        """GitHub Models inference — free for any GitHub PAT, OpenAI-compatible."""

        api_key = self.settings.github_models_token
        if not api_key:
            raise LLMUnavailableError("GitHub Models token missing")
        model = self.settings.github_models_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://models.inference.ai.azure.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("GitHub Models returned empty content")
        return LLMResult(text=text, provider="github", model=model)

    async def _call_openrouter(self, prompt: str, temperature: float, response_json: bool) -> LLMResult:
        api_key = self.settings.openrouter_api_key
        if not api_key:
            raise LLMUnavailableError("OpenRouter key missing")
        model = self.settings.openrouter_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/specter-titan",
                    "X-Title": "Specter Titan Title Review",
                },
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("OpenRouter returned empty content")
        return LLMResult(text=text, provider="openrouter", model=model)


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", []) or []
        return "".join(str(getattr(part, "text", "")) for part in parts)
    return ""


def _coerce_json_object(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction tolerant of Markdown fences and stray prose."""

    if not raw:
        raise ValueError("Empty LLM response")
    # Strip fenced ```json blocks
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else raw
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        # Last-resort: pull the first {...} block
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise
        loaded = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError("LLM JSON response was not a JSON object")
    return loaded


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Module-level cached client. Re-reads settings on first call only."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_llm_client() -> None:
    """Force the cached client to be re-instantiated (useful in tests)."""
    global _default_client
    _default_client = None


__all__ = [
    "LLMClient",
    "LLMResult",
    "get_llm_client",
    "reset_llm_client",
]
