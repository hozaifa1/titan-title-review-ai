"""Telemetry sanity tests — logging config + trace IDs."""

from __future__ import annotations

from titan.config import reload_settings
from titan.telemetry import bind_trace_id, configure_logging, get_logger


def test_configure_logging_returns_a_usable_logger() -> None:
    settings = reload_settings()
    configure_logging(settings)
    log = get_logger("titan.tests")
    # We just want to confirm a structlog logger comes back and accepts kwargs.
    log.info("test_event", k="v")


def test_bind_trace_id_returns_a_string() -> None:
    trace_id = bind_trace_id()
    assert isinstance(trace_id, str)
    assert len(trace_id) >= 8


def test_bind_trace_id_accepts_explicit_value() -> None:
    trace_id = bind_trace_id("known-trace-id")
    assert trace_id == "known-trace-id"
