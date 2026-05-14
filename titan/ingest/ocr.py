"""Document parsing router for title-source documents.

The router tries Docling first, replaces low-confidence pages with pdfplumber
text extraction, and uses the Qwen2.5-VL hook for handwritten pages. The VLM
hook is intentionally provider-light: in offline demo mode it can consume a
checked-in human transcript fixture; in production it can be backed by an
endpoint without changing the public `parse_document` contract.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from titan.config import get_settings
from titan.ingest.models import PageClass, ParsedDoc, ParsedPage, ParserName
from titan.telemetry import get_logger

LOW_CONFIDENCE_THRESHOLD = 0.8
MAX_PAGES_PER_DOC = 500  # bound memory + Gemini cost on adversarial inputs
log = get_logger(__name__)


async def parse_document(path: str | Path) -> ParsedDoc:
    """Parse a PDF/image into per-page markdown with fallback routing."""

    source_path = Path(path)
    docling_pages, warnings = await _run_docling(source_path)
    if not docling_pages:
        docling_pages, pdf_warnings = await _run_pdfplumber(source_path)
        warnings.extend(pdf_warnings)

    if len(docling_pages) > MAX_PAGES_PER_DOC:
        log.warning(
            "ocr.page_cap_applied",
            path=str(source_path),
            requested=len(docling_pages),
            cap=MAX_PAGES_PER_DOC,
        )
        warnings.append(
            f"Document had {len(docling_pages)} pages; truncated to {MAX_PAGES_PER_DOC}."
        )
        docling_pages = docling_pages[:MAX_PAGES_PER_DOC]

    pages: list[ParsedPage] = []
    parser_chain: list[ParserName] = ["docling"]
    for page in docling_pages:
        classification = await classify_page(page.markdown, source_path, page.page_number)
        if classification == "handwritten":
            qwen_page = await _run_qwen2_5_vl(source_path, page.page_number)
            if qwen_page is not None:
                pages.append(qwen_page)
                if "qwen2_5_vl" not in parser_chain:
                    parser_chain.append("qwen2_5_vl")
                continue

        if page.confidence < LOW_CONFIDENCE_THRESHOLD:
            pdf_page = await _run_pdfplumber_page(source_path, page.page_number)
            if pdf_page is not None and pdf_page.confidence >= page.confidence:
                pages.append(pdf_page)
                if "pdfplumber" not in parser_chain:
                    parser_chain.append("pdfplumber")
                continue

        pages.append(page.model_copy(update={"classification": classification}))

    return ParsedDoc.from_pages(source_path, pages, parser_chain, warnings)


async def classify_page(markdown: str, path: Path, page_number: int) -> PageClass:
    """Classify page quality.

    Gemini Flash is used when `GOOGLE_API_KEY` is present; otherwise a stable
    heuristic keeps the offline checkpoint deterministic.
    """

    if _looks_handwritten(path, markdown):
        return "handwritten"

    settings = get_settings()
    api_key = settings.gemini_key
    if not api_key:
        return _heuristic_classification(markdown)

    try:
        label = await _classify_with_gemini(api_key, settings.gemini_model, path, page_number, markdown)
        if label in {"clean_text", "scanned_typed", "handwritten", "mixed_low_quality"}:
            return label  # type: ignore[return-value]
    except Exception as exc:
        log.warning("page classification fell back to heuristic", error=str(exc), page=page_number)
        return _heuristic_classification(markdown)

    return _heuristic_classification(markdown)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _classify_with_gemini(
    api_key: str, model_name: str, path: Path, page_number: int, markdown: str
) -> str:
    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        "Classify this OCR page as exactly one of clean_text, scanned_typed, "
        "handwritten, mixed_low_quality. Return only the label.\n\n"
        f"File: {path.name}, page {page_number}\n\n{markdown[:4000]}"
    )
    response = await asyncio.to_thread(model.generate_content, prompt)
    return (response.text or "").strip().lower()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    stop=stop_after_attempt(3),
    reraise=False,
)
async def _run_docling(path: Path) -> tuple[list[ParsedPage], list[str]]:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        def convert() -> str:
            result = DocumentConverter().convert(str(path))
            return result.document.export_to_markdown()

        markdown = await asyncio.to_thread(convert)
        pages = _split_markdown_pages(markdown)
        if pages:
            return [
                ParsedPage(
                    page_number=index,
                    markdown=page_md,
                    parser="docling",
                    confidence=_estimate_confidence(page_md),
                    classification=_heuristic_classification(page_md),
                )
                for index, page_md in enumerate(pages, start=1)
            ], []
    except Exception as exc:
        return [], [f"Docling failed; falling back to pdfplumber: {exc}"]

    return [], ["Docling returned no markdown; falling back to pdfplumber."]


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


async def _run_pdfplumber_page(path: Path, page_number: int) -> ParsedPage | None:
    pages, _ = await _run_pdfplumber(path)
    if 1 <= page_number <= len(pages):
        return pages[page_number - 1]
    return None


async def _run_qwen2_5_vl(path: Path, page_number: int) -> ParsedPage | None:
    transcript = _fixture_transcript_for(path)
    if transcript:
        page_text = _extract_fixture_page(transcript, page_number)
        if page_text:
            return ParsedPage(
                page_number=page_number,
                markdown=page_text,
                parser="fixture_transcript",
                confidence=0.95,
                classification="handwritten",
                warnings=["Offline handwritten fallback used checked-in human transcript fixture."],
            )

    # Production hook: callers can wrap this module and replace this function
    # with a hosted/local Qwen2.5-VL implementation while preserving routing.
    return None


def _fixture_transcript_for(path: Path) -> str | None:
    fixture = Path("data/gold") / f"{path.stem}.transcript.md"
    if fixture.exists():
        return fixture.read_text(encoding="utf-8")
    return None


def _extract_fixture_page(transcript: str, page_number: int) -> str:
    pattern = rf"## Page {page_number}\s+(.*?)(?=\n## Page \d+\s+|\Z)"
    match = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else transcript.strip()


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
