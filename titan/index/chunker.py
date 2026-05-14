"""Contextual chunking for title-source documents."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass

from titan.index.models import Chunk
from titan.schemas import Provenance, TitleDocument

DEFAULT_CHUNK_TOKENS = 600
DEFAULT_OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class ChunkerConfig:
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    use_gemini_context: bool = True


async def chunk_title_document(
    title_document: TitleDocument,
    markdown: str,
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    """Split a parsed title document into contextual chunks.

    When a Gemini API key is present, each chunk receives the contextual
    one-sentence prefix described in the architecture plan. The offline fallback
    derives a stable sentence from the nearest Schedule/page heading so tests and
    reviewer checkpoints run without external calls.
    """

    cfg = config or ChunkerConfig()
    windows = _token_windows(markdown, cfg.chunk_tokens, cfg.overlap_tokens)
    chunks: list[Chunk] = []
    for index, (text, start, end) in enumerate(windows):
        context = await _context_sentence(markdown, text, title_document, cfg.use_gemini_context)
        chunk_id = _chunk_id(title_document.doc_id, index, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=title_document.doc_id,
                doc_type=title_document.doc_type,
                text=text,
                contextual_text=f"{context}\n\n{text}",
                provenance=Provenance(
                    doc_id=title_document.doc_id,
                    page=_page_for_offset(markdown, start),
                    char_span=(start, end),
                    snippet=text[:200],
                ),
                metadata={"context": context},
            )
        )

    chunks.extend(_structured_field_chunks(title_document, len(chunks)))
    return chunks


def _token_windows(text: str, chunk_tokens: int, overlap_tokens: int) -> list[tuple[str, int, int]]:
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []

    step = max(1, chunk_tokens - overlap_tokens)
    windows: list[tuple[str, int, int]] = []
    for token_start in range(0, len(tokens), step):
        selected = tokens[token_start : token_start + chunk_tokens]
        if not selected:
            break
        start = selected[0].start()
        end = selected[-1].end()
        windows.append((text[start:end].strip(), start, end))
        if token_start + chunk_tokens >= len(tokens):
            break
    return windows


async def _context_sentence(full_doc: str, chunk: str, title_document: TitleDocument, use_gemini: bool) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if use_gemini and api_key:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = (
                "You are creating contextual retrieval chunks for a title-review system. "
                "Given the full document and one chunk, write one concise sentence that "
                "situates the chunk by document section and subject. Return only the sentence.\n\n"
                f"Full document:\n{full_doc[:120000]}\n\nChunk:\n{chunk[:8000]}"
            )
            response = await asyncio.to_thread(model.generate_content, prompt)
            text = (response.text or "").strip()
            if text:
                return _single_sentence(text)
        except Exception:
            pass

    return _heuristic_context(chunk, title_document)


def _heuristic_context(chunk: str, title_document: TitleDocument) -> str:
    schedule = re.search(r"\b(Schedule\s+[A-Z](?:[-\s]+Section\s+[IVX]+)?)\b", chunk, re.IGNORECASE)
    if schedule:
        section = schedule.group(1).replace("\n", " ")
        return f"This chunk comes from {section} of {title_document.doc_id}, a {title_document.doc_type}."
    page = re.search(r"\bPage\s+\d+\b", chunk, re.IGNORECASE)
    if page:
        return f"This chunk comes from {page.group(0)} of {title_document.doc_id}, a {title_document.doc_type}."
    return f"This chunk comes from {title_document.doc_id}, a {title_document.doc_type}."


def _structured_field_chunks(title_document: TitleDocument, start_index: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for offset, party in enumerate(title_document.vesting):
        text = f"Schedule A vested owner: {party.name}. Role: {party.role}."
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(title_document.doc_id, start_index + offset, text),
                doc_id=title_document.doc_id,
                doc_type=title_document.doc_type,
                text=text,
                contextual_text=f"This chunk summarizes Schedule A vesting for {title_document.doc_id}.\n\n{text}",
                provenance=Provenance(doc_id=title_document.doc_id, page=1, char_span=None, snippet=text),
                metadata={"field": "vesting"},
            )
        )
    return chunks


def _page_for_offset(text: str, offset: int) -> int:
    page = 1
    for match in re.finditer(r"\bPage\s+(\d+)\b|^##\s*Page\s+(\d+)", text[:offset], re.IGNORECASE | re.MULTILINE):
        number = match.group(1) or match.group(2)
        if number:
            page = int(number)
    return page


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:chunk:{index:04d}:{digest}"


def _single_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1) if match else cleaned[:240]
