"""Structured logging via ``structlog`` with a context-local ``trace_id``.

Use :func:`configure_logging` once at process startup (the CLI does this).
Then call :func:`get_logger` from any module:

    log = get_logger(__name__)
    log.info("starting ingest", path=str(path))

A ``trace_id`` is automatically attached to every log entry once you enter
:func:`trace_context` (or manually call :func:`bind_trace_id`). This is the
correlation key for a single pipeline run.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

from titan.config import Settings, get_settings

_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "titan_trace_id", default=None
)
_CONFIGURED: bool = False


def _trace_id_processor(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    trace_id = _TRACE_ID.get()
    if trace_id and "trace_id" not in event_dict:
        event_dict["trace_id"] = trace_id
    return event_dict


def configure_logging(settings: Settings | None = None) -> None:
    """Configure ``structlog`` + stdlib ``logging`` once."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = settings or get_settings()
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    renderer: structlog.types.Processor
    if cfg.log_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _trace_id_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def new_trace_id() -> str:
    """Generate a fresh hex trace id."""
    return uuid.uuid4().hex[:16]


def bind_trace_id(trace_id: str | None = None) -> str:
    """Attach ``trace_id`` to the current context. Returns the id used."""
    value = trace_id or new_trace_id()
    _TRACE_ID.set(value)
    return value


def clear_trace_id() -> None:
    _TRACE_ID.set(None)


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id for the duration of a ``with`` block."""
    token = _TRACE_ID.set(trace_id or new_trace_id())
    try:
        yield _TRACE_ID.get() or ""
    finally:
        _TRACE_ID.reset(token)


__all__ = [
    "bind_trace_id",
    "clear_trace_id",
    "configure_logging",
    "get_logger",
    "new_trace_id",
    "trace_context",
]
