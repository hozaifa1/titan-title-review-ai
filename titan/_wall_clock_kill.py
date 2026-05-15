"""Hard wall-clock kill for processes that have non-cancellable threads.

``asyncio.wait_for`` only cancels coroutines. When the coroutine is blocked
on ``asyncio.to_thread(...)`` the underlying Python thread keeps running and
``asyncio.run`` waits for it on shutdown, so even after the timeout fires
the process never exits.

The functions here use a background ``threading.Timer`` that calls
``os._exit`` — bypassing all Python-level cleanup and thread joins. Use
this in test/smoke-test wrappers where you want a *true* upper bound on
wall-clock time. Do not use in production code paths; this kills the
process without giving callers a chance to clean up.
"""

from __future__ import annotations

import os
import threading


def arm_kill_timer(seconds: float, exit_code: int = 2) -> threading.Timer:
    """Start a background timer that hard-kills the process after ``seconds``.

    Returns the timer so callers can ``cancel()`` it if they finish in time.
    """

    def _kill() -> None:
        # Print before exiting so we have a clear breadcrumb in CI logs.
        try:
            print(f"!!! WALL-CLOCK KILL after {seconds:.0f}s !!!", flush=True)
        finally:
            os._exit(exit_code)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()
    return timer


__all__ = ["arm_kill_timer"]
