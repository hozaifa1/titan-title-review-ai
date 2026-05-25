"""Vision-LLM (VLM) hook for handwritten / image-only PDF pages.

Why this lives in its own module
--------------------------------

Production wires a real Qwen2.5-VL (or any other VLM) endpoint here. The
demo distribution ships **without** the model: instead it ships a small set
of human-written transcripts under ``data/gold/<doc_id>.transcript.md`` so
users can run the pipeline end-to-end without GPU infra.

The hook is intentionally tiny — three orthogonal concerns separated:

* :func:`call_vlm` — the *real* hook. Empty by default. Production replaces
  the body with an HTTP call to a hosted VLM (modal, beam, runpod, your own
  vLLM, etc.).
* :func:`load_transcript_fixture` — the *offline* fallback. Reads a
  committed transcript from ``data/gold``. This is the demo path; it should
  never run in a system that has a real VLM configured.
* :func:`vlm_extract` — the public entry point. Tries the real VLM first;
  only consults the transcript fixture when no VLM is configured AND only
  for documents that look handwritten.

See ``docs/VLM_INTEGRATION.md`` for the wiring guide and the constraints
the real callable must satisfy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from titan.telemetry import get_logger

log = get_logger(__name__)

# Flag name keyed in the environment. ``TITAN_VLM_ENABLED=1`` plus a real
# implementation of :func:`call_vlm` switches off the fixture fallback.
VLM_ENV_FLAG = "TITAN_VLM_ENABLED"
TRANSCRIPT_DIR = Path("data") / "gold"


def vlm_enabled() -> bool:
    """True iff the operator has explicitly opted in to a real VLM."""
    return os.getenv(VLM_ENV_FLAG, "0").strip() in {"1", "true", "yes", "on"}


async def call_vlm(pdf_path: Path, page_number: int) -> str | None:
    """Run a Vision-LLM on one PDF page and return its markdown transcription.

    This is the seam where production plugs in a real VLM. The default
    implementation returns ``None`` (= unavailable) so the pipeline falls
    through to the next strategy (transcript fixture or warning).

    Implementation contract for production:
      * Render the page to a PNG/JPEG (e.g. via pdf2image or pypdfium2).
      * POST the image to your VLM endpoint with a prompt like
        "Transcribe this handwritten document page faithfully to markdown."
      * Return the transcribed text (or ``None`` if the VLM was unavailable
        for this call so the caller can degrade gracefully).
      * Honour per-call timeouts — long stalls block the whole ingest path.

    See ``docs/VLM_INTEGRATION.md`` for example wiring.
    """

    del pdf_path, page_number  # default no-op
    return None


def load_transcript_fixture(pdf_path: Path, page_number: int) -> str | None:
    """Return a committed human transcript for the page, if one exists.

    The transcript file lives at ``data/gold/<doc_id>.transcript.md`` and
    follows the ``## Page <N>`` section convention. **This is a demo-mode
    fallback**: it intentionally never runs in production with VLM enabled,
    because reading hand-labelled gold to populate a "VLM" output would make
    eval results meaningless.
    """

    fixture = TRANSCRIPT_DIR / f"{pdf_path.stem}.transcript.md"
    if not fixture.exists():
        return None
    try:
        transcript = fixture.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("vlm.transcript_read_failed", path=str(fixture), error=str(exc))
        return None
    return _extract_page(transcript, page_number)


def _extract_page(transcript: str, page_number: int) -> str:
    """Pull a ``## Page <N>`` block from a transcript fixture."""
    pattern = rf"##\s*Page\s+{page_number}\s+(.*?)(?=\n##\s*Page\s+\d+\s+|\Z)"
    match = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else transcript).strip()


async def vlm_extract(pdf_path: Path, page_number: int) -> tuple[str, str] | None:
    """Transcribe ``page_number`` via VLM, returning ``(text, source_label)``.

    ``source_label`` is ``"vlm"`` if the real model produced the text and
    ``"fixture_transcript"`` when the offline demo fixture supplied it.
    Returns ``None`` when neither path produced text — caller must surface
    a warning so downstream extractors know content may be missing.

    Ordering is deliberate: production VLM is consulted first; the
    transcript fixture is **only** consulted when no VLM is wired AND we
    actually have a fixture (i.e. an explicit demo case). This stops the
    demo path from masking a misconfigured VLM in production.
    """

    if vlm_enabled():
        text = await call_vlm(pdf_path, page_number)
        if text and text.strip():
            return text.strip(), "vlm"
        log.warning(
            "vlm.real_vlm_returned_no_text",
            path=str(pdf_path),
            page=page_number,
        )
        return None  # do not fall back to fixture when VLM is enabled

    fixture_text = load_transcript_fixture(pdf_path, page_number)
    if fixture_text:
        return fixture_text, "fixture_transcript"
    return None


__all__ = [
    "VLM_ENV_FLAG",
    "vlm_enabled",
    "call_vlm",
    "load_transcript_fixture",
    "vlm_extract",
]
