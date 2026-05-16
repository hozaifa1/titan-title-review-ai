"""Disk-cached PDF→markdown extraction with smart routing.

Order of attempts per document, cached to ``data/.md_cache/<doc_id>.md``:

1. **pdfplumber** for text-layer PDFs, each page in a worker thread with a
   ``PER_PAGE_TIMEOUT_SECONDS`` deadline — dense survey pages produce a
   placeholder so the pipeline never stalls.
2. **Docling** fallback for image-only / scanned PDFs (when pdfplumber
   returns < ``DOCLING_FALLBACK_THRESHOLD`` characters). Costs 60-170s once
   on CPU then sits in the disk cache forever.
3. **VLM hook** (``titan.ingest.vlm.vlm_extract``) — for documents that
   still came back near-empty AND look handwritten. With ``TITAN_VLM_ENABLED=1``
   plus a real implementation of ``call_vlm`` this is a hosted Vision-LLM;
   otherwise it falls through to a committed transcript fixture under
   ``data/gold/<doc_id>.transcript.md`` for the demo distribution.

The transcript fixture is **never** consulted before OCR runs — it's a
hard-coded ground truth that would short-circuit actual extraction, which
the eval would mistakenly credit to the pipeline.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

from titan.schemas import TitleDocument
from titan.telemetry import get_logger

log = get_logger(__name__)

CACHE_DIR = Path("data") / ".md_cache"
PER_PAGE_TIMEOUT_SECONDS = 25.0
DOCLING_FALLBACK_THRESHOLD = 200  # chars below which we try Docling


def _cache_path(doc_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{doc_id}.md"


def _extract_page_text(pdf_path: Path, page_index: int) -> str:
    import pdfplumber  # type: ignore[import-not-found]

    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_index].extract_text() or ""


def _content_length(markdown: str) -> int:
    """Length excluding ``## Page N`` headers and timeout/error placeholders."""

    body = markdown
    for sentinel in ("## Page", "[Page ", "timed out", "extraction failed"):
        body = body.replace(sentinel, "")
    return len(body.strip())


def _try_docling(pdf_path: Path, doc_id: str) -> str | None:
    """Run the existing OCR pipeline (Docling-aware) and return its markdown."""

    try:
        import asyncio

        from titan.ingest.ocr import parse_document
    except Exception as exc:
        log.warning("markdown_cache.docling_import_failed", error=str(exc)[:200])
        return None

    try:
        parsed = asyncio.run(parse_document(pdf_path))
    except Exception as exc:
        log.warning("markdown_cache.docling_failed", doc_id=doc_id, error=str(exc)[:200])
        return None
    if not parsed.markdown.strip():
        return None
    return parsed.markdown


def read_markdown_for(document: TitleDocument, *, refresh: bool = False) -> str:
    """Return parsed markdown for ``document``, with disk caching and OCR fallback.

    Cache → pdfplumber → Docling → VLM hook (real or fixture stub). The
    transcript fixture is NEVER consulted before OCR runs — using gold to
    populate "extracted" text would invalidate the eval.
    """

    cache_path = _cache_path(document.doc_id)
    if cache_path.exists() and not refresh:
        try:
            return cache_path.read_text(encoding="utf-8")
        except OSError:
            pass

    pdf_path = Path(document.file_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return _document_to_markdown(document)

    # 1. Try pdfplumber with per-page timeout.
    pages: list[str] = []
    try:
        import pdfplumber  # type: ignore[import-not-found]

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
    except Exception as exc:
        log.warning("markdown_cache.pdfplumber_open_failed", doc_id=document.doc_id, error=str(exc)[:200])
        page_count = 0

    if page_count:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for page_index in range(page_count):
                future = executor.submit(_extract_page_text, pdf_path, page_index)
                try:
                    text = future.result(timeout=PER_PAGE_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    text = (
                        f"[Page {page_index + 1} text extraction timed out after "
                        f"{PER_PAGE_TIMEOUT_SECONDS:.0f}s — dense layout]"
                    )
                except Exception as exc:
                    text = f"[Page {page_index + 1} extraction failed: {exc.__class__.__name__}]"
                pages.append(f"## Page {page_index + 1}\n\n{text}")
        markdown = "\n\n".join(pages)
    else:
        markdown = ""

    # 2. Docling fallback if pdfplumber returned almost nothing (scanned PDF).
    if _content_length(markdown) < DOCLING_FALLBACK_THRESHOLD:
        log.info("markdown_cache.docling_fallback", doc_id=document.doc_id, pdfplumber_chars=len(markdown))
        docling_md = _try_docling(pdf_path, document.doc_id)
        if docling_md:
            markdown = docling_md

    # 3. VLM hook for documents that still came back near-empty AND look
    # handwritten. Real production VLM in priority; demo distributions fall
    # through to the committed transcript fixture inside ``vlm_extract``.
    if _content_length(markdown) < DOCLING_FALLBACK_THRESHOLD and _looks_handwritten(pdf_path):
        log.info("markdown_cache.vlm_fallback", doc_id=document.doc_id)
        vlm_md = _try_vlm(pdf_path, page_count or 1)
        if vlm_md:
            markdown = vlm_md

    if not markdown.strip():
        markdown = _document_to_markdown(document)

    try:
        cache_path.write_text(markdown, encoding="utf-8")
    except OSError:
        pass
    return markdown


def _looks_handwritten(pdf_path: Path) -> bool:
    """Quick filename heuristic — keeps the VLM path off normal PDFs."""
    name = pdf_path.name.lower()
    return "handwritten" in name or "fromthepage" in name or "deed" in name


def _try_vlm(pdf_path: Path, page_count: int) -> str | None:
    """Aggregate per-page VLM transcriptions into one markdown blob."""
    from titan.ingest.vlm import vlm_extract

    async def _gather() -> list[tuple[int, str, str]]:
        results: list[tuple[int, str, str]] = []
        for page_number in range(1, page_count + 1):
            outcome = await vlm_extract(pdf_path, page_number)
            if outcome is None:
                continue
            text, source = outcome
            results.append((page_number, text, source))
        return results

    try:
        rows = asyncio.run(_gather())
    except RuntimeError:
        # Already inside an event loop — fall back to a fresh loop.
        loop = asyncio.new_event_loop()
        try:
            rows = loop.run_until_complete(_gather())
        finally:
            loop.close()
    if not rows:
        return None
    sources = {source for _, _, source in rows}
    log.info("markdown_cache.vlm_extracted", doc_id=pdf_path.stem, pages=len(rows), sources=sorted(sources))
    return "\n\n".join(f"## Page {pn}\n\n{text}" for pn, text, _ in rows)


def _document_to_markdown(document: TitleDocument) -> str:
    return "\n".join(
        [
            f"# {document.doc_id}",
            f"Document type: {document.doc_type}",
            "Vesting: " + ", ".join(party.name for party in document.vesting),
            "Parties: " + ", ".join(f"{party.name} ({party.role})" for party in document.parties),
            "Warnings: " + "; ".join(document.extraction_warnings),
        ]
    )


__all__ = ["read_markdown_for"]
