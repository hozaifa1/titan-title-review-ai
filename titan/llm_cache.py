"""On-disk content-hash cache for LLM responses.

Re-running the same prompt with the same temperature should never hit the
provider again — this matters enormously when iterating on an eval over
the same documents, where without caching we burn through every free
tier's daily quota inside the first run.

Keyed by SHA-256 of ``(prompt, temperature, response_json)`` so a single
change to prompt content invalidates the entry. Hot path: ``get`` returns
the cached ``LLMResult`` (provider="cache") or ``None``. Cold path: caller
makes a real LLM call and stores the result via ``put``.

The cache lives under ``data/.llm_cache/`` (gitignored). A ``cache:`` provider
in the ``LLMResult`` tells trace consumers that no real call was made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from titan.llm_client import LLMResult


def _cache_dir() -> Path:
    path = Path("data") / ".llm_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(prompt: str, temperature: float, response_json: bool, namespace: str = "default") -> str:
    payload = json.dumps(
        {"p": prompt, "t": temperature, "j": response_json, "n": namespace},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(prompt: str, temperature: float, response_json: bool, namespace: str = "default") -> "LLMResult | None":
    """Return the cached ``LLMResult`` for this prompt or ``None``."""

    from titan.llm_client import LLMResult  # local import to avoid cycle

    cache_path = _cache_dir() / f"{_key(prompt, temperature, response_json, namespace)}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return LLMResult(
            text=data["text"],
            provider=f"cache:{data.get('original_provider', 'unknown')}",
            model=data.get("original_model", "cache"),
        )
    except Exception:
        # Bad cache file — silently treat as miss; next put() will overwrite.
        return None


def put(
    prompt: str,
    temperature: float,
    response_json: bool,
    result: "LLMResult",
    namespace: str = "default",
) -> None:
    """Persist an ``LLMResult`` keyed by prompt+config."""

    cache_path = _cache_dir() / f"{_key(prompt, temperature, response_json, namespace)}.json"
    payload = {
        "text": result.text,
        "original_provider": result.provider,
        "original_model": result.model,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def clear() -> int:
    """Delete every cached entry. Returns the number of files removed."""

    count = 0
    for entry in _cache_dir().glob("*.json"):
        entry.unlink(missing_ok=True)
        count += 1
    return count


__all__ = ["get", "put", "clear"]
