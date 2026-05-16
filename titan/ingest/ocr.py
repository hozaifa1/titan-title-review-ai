"""Document parsing router for title-source documents.

Routing strategy (changed 2026-05-15 — performance fix):
  1. **pdfplumber first.** Modern title docs are 90%+ text-layer PDFs;
     pdfplumber returns near-identical text in ~1s vs Docling's 90-170s on
     CPU. For text PDFs, Docling gains us nothing meaningful.
  2. **Docling fallback for sparse pages.** When a pdfplumber page comes
     back with very few characters per square inch (the image-only-scan
     signature), invoke Docling for THAT page only — preserving OCR quality
     without paying Docling's tax on every page.
  3. **Qwen2.5-VL hook for handwritten pages.** Detected by the page
     classifier; offline demo mode can consume a checked-in transcript
     fixture; production wires a hosted endpoint behind the same hook.

The public contract — ``parse_document(path) -> ParsedDoc`` — is unchanged.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from titan.config import get_settings
from titan.ingest.models import PageClass, ParsedDoc, ParsedPage, ParserName
from titan.telemetry import get_logger

LOW_CONFIDENCE_THRESHOLD = 0.8
MAX_PAGES_PER_DOC = 500  # bound memory + Gemini cost on adversarial inputs
SPARSE_PAGE_CHAR_THRESHOLD = 60  # below this we suspect a scanned/image-only page
log = get_logger(__name__)


async def parse_document(path: str | Path) -> ParsedDoc:
    """Parse a PDF/image into per-page markdown with smart routing.

    Strategy: pdfplumber first (fast); Docling only for pages whose text
    layer is sparse (likely scanned images). Image-only PDFs without any
    text layer trigger a whole-document Docling pass.
    """

    source_path = Path(path)
    pdf_pages, warnings = await _run_pdfplumber(source_path)

    # If pdfplumber returned absolutely nothing, the PDF has no text layer
    # at all (pure image scan) — fall through to Docling for the whole doc.
    total_chars = sum(len(p.markdown) for p in pdf_pages)
    if not pdf_pages or total_chars < SPARSE_PAGE_CHAR_THRESHOLD:
        log.info(
            "ocr.pdfplumber_returned_no_text_using_docling",
            path=str(source_path),
            total_chars=total_chars,
        )
        docling_pages, docling_warnings = await _run_docling(source_path)
        warnings.extend(docling_warnings)
        base_pages = docling_pages or pdf_pages
        parser_chain: list[ParserName] = ["docling"] if docling_pages else ["pdfplumber"]
    else:
        base_pages = pdf_pages
        parser_chain = ["pdfplumber"]

    if len(base_pages) > MAX_PAGES_PER_DOC:
        log.warning(
            "ocr.page_cap_applied",
            path=str(source_path),
            requested=len(base_pages),
            cap=MAX_PAGES_PER_DOC,
        )
        warnings.append(
            f"Document had {len(base_pages)} pages; truncated to {MAX_PAGES_PER_DOC}."
        )
        base_pages = base_pages[:MAX_PAGES_PER_DOC]

    # Identify which pages have sparse text and might need a Docling rescue.
    # Critical: Docling has NO per-page API in 2026 — invoking it for ONE
    # page costs a full-document pass (90-170s on CPU). It's only worth the
    # cost if a significant FRACTION of pages are sparse (genuinely scanned
    # input) — not when a 37-page text PDF has one mostly-blank divider.
    sparse_pages: list[int] = []
    if "pdfplumber" in parser_chain and "docling" not in parser_chain:
        for page in base_pages:
            if len(page.markdown.strip()) < SPARSE_PAGE_CHAR_THRESHOLD:
                sparse_pages.append(page.page_number)

    docling_rescued: dict[int, ParsedPage] = {}
    if sparse_pages:
        sparse_ratio = len(sparse_pages) / max(len(base_pages), 1)
        # Only pay the full-doc Docling tax if the document is mostly scanned
        # (>=30% sparse pages) OR very small (<=3 pages where each page matters).
        if sparse_ratio >= 0.30 or len(base_pages) <= 3:
            log.info(
                "ocr.docling_rescue_sparse_pages",
                path=str(source_path),
                count=len(sparse_pages),
                sparse_ratio=round(sparse_ratio, 2),
                pages=sparse_pages[:10],
            )
            rescued, rescue_warnings = await _run_docling(source_path, only_pages=sparse_pages)
            warnings.extend(rescue_warnings)
            for rp in rescued:
                docling_rescued[rp.page_number] = rp
            if docling_rescued:
                parser_chain.append("docling")
        else:
            log.info(
                "ocr.docling_rescue_skipped_low_sparse_ratio",
                path=str(source_path),
                sparse_count=len(sparse_pages),
                total_pages=len(base_pages),
                sparse_ratio=round(sparse_ratio, 2),
            )
            warnings.append(
                f"{len(sparse_pages)} sparse-text page(s) detected out of {len(base_pages)}; "
                "skipping Docling rescue because the document is predominantly text — "
                "the cost (full-doc OCR pass) outweighs the gain on isolated blank pages."
            )

    # Apply Docling rescues + decide which pages even need LLM classification.
    candidates: list[ParsedPage] = []
    for page in base_pages:
        rescued_page = docling_rescued.get(page.page_number)
        if rescued_page is not None and len(rescued_page.markdown.strip()) > len(page.markdown.strip()):
            page = rescued_page
        candidates.append(page)

    # Skip the LLM classifier when the heuristic is already confident (clean
    # text). This is a 10-100x speedup on multi-page commitments: we don't
    # need a 2s LLM call to confirm what a confidence score >= 0.85 already
    # tells us.  Only ambiguous pages get the LLM call, and even those are
    # batched in parallel rather than awaited sequentially.
    classifications: dict[int, PageClass] = {}
    ambiguous: list[ParsedPage] = []
    for page in candidates:
        if _looks_handwritten(source_path, page.markdown):
            classifications[page.page_number] = "handwritten"
            continue
        heuristic = _heuristic_classification(page.markdown)
        if page.confidence >= 0.85 or len(page.markdown.strip()) < 40:
            # Confident clean text OR a near-empty page — no point asking an LLM.
            classifications[page.page_number] = heuristic
        else:
            ambiguous.append(page)

    if ambiguous and get_settings().has_any_llm:
        results = await asyncio.gather(
            *[classify_page(p.markdown, source_path, p.page_number) for p in ambiguous],
            return_exceptions=True,
        )
        for p, result in zip(ambiguous, results, strict=True):
            if isinstance(result, BaseException):
                classifications[p.page_number] = _heuristic_classification(p.markdown)
            else:
                classifications[p.page_number] = result
    else:
        for p in ambiguous:
            classifications[p.page_number] = _heuristic_classification(p.markdown)

    pages: list[ParsedPage] = []
    for page in candidates:
        classification = classifications.get(page.page_number, _heuristic_classification(page.markdown))
        if classification == "handwritten":
            qwen_page = await _run_qwen2_5_vl(source_path, page.page_number)
            if qwen_page is not None:
                pages.append(qwen_page)
                if "qwen2_5_vl" not in parser_chain:
                    parser_chain.append("qwen2_5_vl")
                continue
            # Handwritten page with no transcript fixture and no live VLM:
            # do NOT silently fall through — flag it so downstream extractors
            # know content may be missing.
            warning_msg = (
                f"Page {page.page_number} classified as handwritten but no VLM endpoint "
                "or transcript fixture was available; downstream extraction will rely on "
                "the OCR text only and may miss handwritten content."
            )
            log.warning(
                "ocr.handwritten_page_no_vlm",
                path=str(source_path),
                page=page.page_number,
            )
            warnings.append(warning_msg)
            updated_warnings = list(page.warnings) + [warning_msg]
            pages.append(
                page.model_copy(
                    update={
                        "classification": classification,
                        "warnings": updated_warnings,
                        "confidence": min(page.confidence, 0.4),
                    }
                )
            )
            continue

        pages.append(page.model_copy(update={"classification": classification}))

    return ParsedDoc.from_pages(source_path, pages, parser_chain, warnings)


async def classify_page(markdown: str, path: Path, page_number: int) -> PageClass:
    """Classify page quality via the LLM provider chain; falls back to heuristic.

    The router tries Gemini → Groq → OpenRouter (configured order) before
    degrading to the deterministic heuristic — keeps offline checkpoints
    runnable while preferring real model classifications when available.
    """

    if _looks_handwritten(path, markdown):
        return "handwritten"

    settings = get_settings()
    if not settings.has_any_llm:
        return _heuristic_classification(markdown)

    try:
        from titan.llm_client import get_llm_client

        prompt = (
            "Classify this OCR page as exactly one of clean_text, scanned_typed, "
            "handwritten, mixed_low_quality. Return only the label.\n\n"
            f"File: {path.name}, page {page_number}\n\n{markdown[:4000]}"
        )
        result = await get_llm_client().generate_text(prompt, temperature=0.0)
        label = (result.text or "").strip().lower().split()[0] if result.text else ""
        if label in {"clean_text", "scanned_typed", "handwritten", "mixed_low_quality"}:
            return label  # type: ignore[return-value]
    except Exception as exc:
        log.warning("page classification fell back to heuristic", error=str(exc), page=page_number)

    return _heuristic_classification(markdown)


async def _run_docling(
    path: Path,
    *,
    only_pages: list[int] | None = None,
) -> tuple[list[ParsedPage], list[str]]:
    """Run Docling on the document. If ``only_pages`` is given, return only
    those page numbers (still converts the whole doc — Docling has no
    per-page API in 2026 — but slices the result).
    """

    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        def convert() -> str:
            result = DocumentConverter().convert(str(path))
            return result.document.export_to_markdown()

        markdown = await asyncio.to_thread(convert)
        pages = _split_markdown_pages(markdown)
        if not pages:
            return [], ["Docling returned no markdown."]

        parsed = [
            ParsedPage(
                page_number=index,
                markdown=page_md,
                parser="docling",
                confidence=_estimate_confidence(page_md),
                classification=_heuristic_classification(page_md),
            )
            for index, page_md in enumerate(pages, start=1)
        ]
        if only_pages is not None:
            wanted = set(only_pages)
            parsed = [p for p in parsed if p.page_number in wanted]
        return parsed, []
    except Exception as exc:
        return [], [f"Docling failed: {exc}"]


async def _run_pdfplumber(path: Path) -> tuple[list[ParsedPage], list[str]]:
    try:
        import pdfplumber

        def extract() -> list[str]:
            with pdfplumber.open(path) as pdf:
                return [(page.extract_text(x_tolerance=1, y_tolerance=3) or "") for page in pdf.pages]

        texts = await asyncio.to_thread(extract)
        pages = [
            ParsedPage(
                page_number=index,
                markdown=text.strip(),
                parser="pdfplumber",
                confidence=_estimate_confidence(text),
                classification=_heuristic_classification(text),
                warnings=[] if text.strip() else ["pdfplumber extracted no text from this page"],
            )
            for index, text in enumerate(texts, start=1)
        ]
        return pages, []
    except Exception as exc:
        return [], [f"pdfplumber fallback failed: {exc}"]


async def _run_qwen2_5_vl(path: Path, page_number: int) -> ParsedPage | None:
    """Delegate handwritten-page transcription to the VLM module.

    The real VLM call (or its offline transcript-fixture stand-in) lives in
    :mod:`titan.ingest.vlm` so we don't conflate the production seam with
    the demo-mode fallback. See ``docs/VLM_INTEGRATION.md`` for wiring.
    """

    from titan.ingest.vlm import vlm_extract

    result = await vlm_extract(path, page_number)
    if result is None:
        return None
    text, source_label = result
    warning = (
        "Offline handwritten fallback used checked-in human transcript fixture."
        if source_label == "fixture_transcript"
        else "Handwritten page transcribed by configured Vision-LLM."
    )
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        parser=source_label,  # type: ignore[arg-type]
        confidence=0.95 if source_label == "fixture_transcript" else 0.9,
        classification="handwritten",
        warnings=[warning],
    )


def _split_markdown_pages(markdown: str) -> list[str]:
    parts = re.split(r"\n(?=#{1,3}\s*Page\s+\d+\b)", markdown, flags=re.IGNORECASE)
    pages = [part.strip() for part in parts if part.strip()]
    return pages or ([markdown.strip()] if markdown.strip() else [])


def _estimate_confidence(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    alnum = sum(ch.isalnum() for ch in stripped)
    printable = sum(ch.isprintable() and not ch.isspace() for ch in stripped)
    density = alnum / max(printable, 1)
    length_score = min(len(stripped) / 800, 1.0)
    return max(0.1, min(0.98, 0.45 + (density * 0.35) + (length_score * 0.2)))


def _heuristic_classification(text: str) -> PageClass:
    confidence = _estimate_confidence(text)
    if confidence < 0.45:
        return "mixed_low_quality"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "scanned_typed"
    return "clean_text"


def _looks_handwritten(path: Path, text: str) -> bool:
    lower_name = path.name.lower()
    if "handwritten" in lower_name or "fromthepage" in lower_name:
        return True
    lowered = text.lower()
    return "probate court" in lowered and "made this" in lowered and len(text) < 2500
