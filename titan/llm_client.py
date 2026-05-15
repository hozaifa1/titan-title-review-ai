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


def _semaphore_loop_alive(sem: asyncio.Semaphore) -> bool:
    """Best-effort check that the semaphore's bound loop is still open."""
    loop = getattr(sem, "_loop", None)
    if loop is None:  # not yet bound; safe to keep
        return True
    try:
        return not loop.is_closed()
    except Exception:
        return False


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

    Concurrency budgets and per-call timeouts are enforced strictly. Every
    provider call is wrapped in ``asyncio.wait_for(per_call_timeout)`` and the
    total call (chain walk) is wrapped in ``asyncio.wait_for(total_timeout)``
    so a stuck provider can't hang the pipeline forever.
    """

    # Per-event-loop semaphore cache. Keyed by id(loop) because a Semaphore is
    # implicitly bound to the loop where it's first awaited; reusing one across
    # loops would deadlock indefinitely on closed loops (a real bug we hit).
    _loop_sems: dict[int, asyncio.Semaphore] = {}
    # Providers that have failed-hard (rate-limit / auth / config) during this
    # process lifetime. We skip them on subsequent calls so a single 429 doesn't
    # cause every future request to waste 15s re-confirming the provider is dead.
    # This is critical: ``asyncio.to_thread`` orphans threads on timeout (Python
    # threads can't be cancelled), so repeatedly hitting a stalled provider
    # drains the executor pool and stalls the whole pipeline.
    _dead_providers: set[str] = set()

    @classmethod
    def mark_provider_dead(cls, provider: str) -> None:
        cls._dead_providers.add(provider)

    @classmethod
    def reset_dead_providers(cls) -> None:
        cls._dead_providers.clear()

    @classmethod
    def _acquire_semaphore(cls, size: int) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet — create a fresh semaphore that will be bound on
            # first await; do NOT cache it because we can't key it.
            return asyncio.Semaphore(size)
        cached = cls._loop_sems.get(id(loop))
        # Drop entries whose loop has been closed so we don't leak.
        if cached is None or loop.is_closed():
            cached = asyncio.Semaphore(size)
            cls._loop_sems[id(loop)] = cached
            # Best-effort cleanup of closed loops
            cls._loop_sems = {
                key: sem
                for key, sem in cls._loop_sems.items()
                if key == id(loop) or _semaphore_loop_alive(sem)
            }
        return cached

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sem: asyncio.Semaphore | None = None

    def _ensure_sem(self) -> asyncio.Semaphore:
        # Re-resolve every call so a fresh asyncio loop gets its own semaphore.
        self._sem = self._acquire_semaphore(self.settings.llm_max_concurrency)
        return self._sem

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

        Concurrency is gated by ``_loop_sems`` so batch operations
        (8-section drafts × N docs) don't burst past per-minute caps. The
        total walk is wrapped in an ``asyncio.wait_for`` budget — if every
        provider stalls we surface ``LLMUnavailableError`` rather than hang.
        On ``429`` we honour the provider's ``Retry-After`` header where
        possible — usually the difference between recoverable and dead.
        """

        providers = self.settings.provider_chain
        if not providers:
            raise LLMUnavailableError("No LLM provider configured")

        # Check the content-hash cache before spending any quota. This makes
        # iterative eval runs effectively free — same prompt returns cached
        # output without burning daily-tier limits on any provider.
        if self.settings.llm_cache_enabled:
            from titan import llm_cache

            cached = llm_cache.get(prompt, temperature, response_json)
            if cached is not None:
                return cached

        sem = self._ensure_sem()
        total_budget = self.settings.llm_total_timeout_seconds

        async def _gated() -> LLMResult:
            async with sem:
                return await self._generate_text_inner(
                    prompt, temperature, response_json, max_attempts_per_provider, providers
                )

        try:
            result = await asyncio.wait_for(_gated(), timeout=total_budget)
        except asyncio.TimeoutError as exc:
            raise LLMUnavailableError(
                f"LLM call exceeded total budget {total_budget}s across providers {providers}"
            ) from exc

        if self.settings.llm_cache_enabled:
            from titan import llm_cache

            llm_cache.put(prompt, temperature, response_json, result)
        return result

    async def _generate_text_inner(
        self,
        prompt: str,
        temperature: float,
        response_json: bool,
        max_attempts_per_provider: int,
        providers: list[str],
    ) -> LLMResult:
        last_exc: Exception | None = None
        per_call = self.settings.llm_per_call_timeout_seconds
        dispatch: dict[str, Any] = {
            "gemini": self._call_gemini,
            "groq": self._call_groq,
            "cerebras": self._call_cerebras,
            "sambanova": self._call_sambanova,
            "github": self._call_github_models,
            "openrouter": self._call_openrouter,
        }
        for provider in providers:
            handler = dispatch.get(provider)
            if handler is None:
                continue
            if provider in type(self)._dead_providers:
                # Already marked dead this session — skip without spending a request.
                continue
            for attempt in range(1, max_attempts_per_provider + 1):
                try:
                    # Hard per-call ceiling. Without this a stuck provider
                    # (Gemini SDK worker thread, slow streaming response, etc.)
                    # blocks the pipeline forever — the original stall bug.
                    return await asyncio.wait_for(
                        handler(prompt, temperature, response_json),
                        timeout=per_call,
                    )
                except asyncio.TimeoutError:
                    last_exc = TimeoutError(f"{provider} exceeded {per_call}s per-call timeout")
                    log.warning(
                        "llm.provider_timeout",
                        provider=provider,
                        attempt=attempt,
                        per_call=per_call,
                    )
                    # Stalls leak threads on Gemini specifically; mark dead so we
                    # don't keep draining the executor pool with orphaned work.
                    type(self).mark_provider_dead(provider)
                    break
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
                        # If the wait is recoverable AND short, sit it out;
                        # otherwise the provider is dead for this session.
                        if retry_after is not None and retry_after <= 8 and attempt < max_attempts_per_provider:
                            await asyncio.sleep(retry_after + 0.25)
                            continue
                        type(self).mark_provider_dead(provider)
                        break
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
        """Direct REST call to Gemini.

        Avoids the ``google.generativeai`` SDK because that SDK requires
        ``asyncio.to_thread``, and a stalled thread cannot be cancelled by
        ``asyncio.wait_for`` — Python threads aren't externally
        interruptible, so timeouts leak threads and eventually exhaust the
        executor pool, hanging the whole pipeline. Using httpx gives us
        proper cooperative cancellation on timeout.
        """

        keys = self.settings.gemini_keys
        if not keys:
            raise LLMUnavailableError("Gemini key missing")

        model = self.settings.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        generation_config: dict[str, Any] = {"temperature": temperature}
        if response_json:
            generation_config["response_mime_type"] = "application/json"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        last_exc: Exception | None = None
        # Slightly under per-call timeout so httpx surfaces an error before
        # the outer ``asyncio.wait_for`` fires and orphans the request.
        http_timeout = max(5.0, float(self.settings.llm_per_call_timeout_seconds) - 1.0)

        async with httpx.AsyncClient(timeout=http_timeout) as client:
            for key in keys:
                try:
                    resp = await client.post(
                        url,
                        params={"key": key},
                        json=body,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    candidates = payload.get("candidates") or []
                    if not candidates:
                        raise RuntimeError("Gemini returned no candidates")
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    text = "".join(part.get("text", "") for part in parts)
                    if not text:
                        raise RuntimeError("Gemini returned empty text")
                    return LLMResult(text=text, provider="gemini", model=model)
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
    """Force the cached client to be re-instantiated and clear all dead-provider state."""
    global _default_client
    _default_client = None
    LLMClient.reset_dead_providers()


__all__ = [
    "LLMClient",
    "LLMResult",
    "get_llm_client",
    "reset_llm_client",
]
