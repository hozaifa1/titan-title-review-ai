"""Contextual chunking for title-source documents."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass

from titan.config import get_settings
from titan.index.models import Chunk
from titan.schemas import Provenance, TitleDocument
from titan.telemetry import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_TOKENS = 600
DEFAULT_OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class ChunkerConfig:
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    # Heuristic context (Schedule/Page header + doc_type) is good enough for
    # BM25+dense retrieval — eliminating the LLM context-sentence call per
    # chunk cuts ~80% of provider calls on free tiers. Override via env to
    # re-enable when running with abundant quota.
    use_gemini_context: bool = False


MAX_CHUNKS_PER_DOC = 400  # cap to bound memory + Gemini cost on huge docs
_CONTEXT_CONCURRENCY = 4


async def chunk_title_document(
    title_document: TitleDocument,
    markdown: str,
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    """Split a parsed title document into contextual chunks.

    Per-chunk Gemini context calls run in parallel under a small semaphore so a
    long document does not serialize on N x ~1s LLM round-trips. The offline
    fallback derives a stable sentence from the nearest Schedule/page heading so
    tests and reviewer checkpoints run without external calls.
    """

    cfg = config or ChunkerConfig()
    windows = _token_windows(markdown, cfg.chunk_tokens, cfg.overlap_tokens)
    if len(windows) > MAX_CHUNKS_PER_DOC:
        log.warning("chunker.window_cap_applied", requested=len(windows), cap=MAX_CHUNKS_PER_DOC)
        windows = windows[:MAX_CHUNKS_PER_DOC]

    sem = asyncio.Semaphore(_CONTEXT_CONCURRENCY)

    async def _one_context(text: str) -> str:
        async with sem:
            return await _context_sentence(markdown, text, title_document, cfg.use_gemini_context)

    contexts = await asyncio.gather(*[_one_context(text) for text, _, _ in windows])

    chunks: list[Chunk] = []
    for index, ((text, start, end), context) in enumerate(zip(windows, contexts, strict=True)):
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


_CONTEXT_WINDOW_CHARS = 4000  # ~1K tokens of context around the chunk


def _doc_window_around(full_doc: str, chunk: str) -> str:
    """Return a small slice of the document centred on where ``chunk`` appears.

    The chunker calls this for every chunk in a doc. Sending the full doc
    each time burned ~600K tokens on a 37-page commitment — enough to
    exhaust any free tier in a single eval. The chunk's neighbourhood is
    almost always enough context for a one-sentence situator: section
    headers above, neighbouring schedule references, etc.
    """

    needle = chunk[:200].strip()
    if needle:
        index = full_doc.find(needle)
        if index >= 0:
            start = max(0, index - _CONTEXT_WINDOW_CHARS // 2)
            end = min(len(full_doc), index + len(chunk) + _CONTEXT_WINDOW_CHARS // 2)
            return full_doc[start:end]
    # Fallback: just take the head of the doc (typically has the title block).
    return full_doc[:_CONTEXT_WINDOW_CHARS]


async def _context_sentence(full_doc: str, chunk: str, title_document: TitleDocument, use_gemini: bool) -> str:
    """Generate a one-sentence chunk context via the LLM client, with heuristic fallback.

    ``use_gemini`` is retained for backwards compatibility but now toggles the
    full multi-provider chain. If every provider fails or no key is configured
    we degrade to the heuristic context line.

    Token cost: a 4K-char window around the chunk, NOT the full doc — cuts
    chunker tokens ~30x and prevents free-tier exhaustion on long commitments.
    """

    settings = get_settings()
    if use_gemini and settings.has_any_llm:
        try:
            from titan.llm_client import get_llm_client

            window = _doc_window_around(full_doc, chunk)
            prompt = (
                "You are creating contextual retrieval chunks for a title-review system. "
                "Given the surrounding document context and one chunk, write one concise "
                "sentence that situates the chunk by document section and subject. "
                "Return only the sentence.\n\n"
                f"Document type: {title_document.doc_type}\n"
                f"Surrounding context:\n{window}\n\nChunk:\n{chunk[:6000]}"
            )
            result = await get_llm_client().generate_text(prompt, temperature=0.0)
            if result.text:
                return _single_sentence(result.text)
        except Exception as exc:
            log.warning("contextual chunk generation fell back to heuristic", error=str(exc))

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
        if party.role == "owner":
            text = f"Schedule A vested owner: {party.name}. Role: owner."
            context = f"This chunk summarizes Schedule A vesting for {title_document.doc_id}."
        else:
            text = f"Schedule A vesting party: {party.name}. Role: {party.role}."
            context = f"This chunk summarizes a Schedule A party for {title_document.doc_id}."
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(title_document.doc_id, start_index + offset, text),
                doc_id=title_document.doc_id,
                doc_type=title_document.doc_type,
                text=text,
                contextual_text=f"{context}\n\n{text}",
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
