"""Step-by-step diagnostic of the draft pipeline. Prints between every stage."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv()

def tick(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def main(doc_path: str) -> None:
    tick(f"start {doc_path}")
    from titan.schemas import TitleDocument
    td = TitleDocument.model_validate_json(Path(doc_path).read_text(encoding="utf-8"))
    tick(f"loaded TitleDocument {td.doc_id}")

    from titan.ingest.markdown_cache import read_markdown_for
    tick("reading markdown (cached if available)")
    markdown = read_markdown_for(td)
    tick(f"markdown ready: {len(markdown)} chars")

    from titan.index.chunker import chunk_title_document
    tick("starting chunker")
    chunks = await chunk_title_document(td, markdown)
    tick(f"chunker done: {len(chunks)} chunks")

    from titan.index.embed import embed_chunks
    tick("starting embed")
    embedded, bm25, embedder = embed_chunks(chunks)
    tick(f"embed done: {len(embedded)} chunks, backend={embedder.backend}")

    from titan.index.qdrant_store import HybridChunkStore
    from titan.config import get_settings
    s = get_settings()
    store = HybridChunkStore(qdrant_url=s.qdrant_url, qdrant_api_key=s.qdrant_api_key or None)
    tick("upsert to qdrant")
    store.upsert(embedded, bm25)
    tick(f"qdrant upserted, mirrored={store.qdrant_mirrored}")

    from titan.retrieve.hybrid import HybridRetriever
    retriever = HybridRetriever(store, embedder, bm25)

    from titan.draft.orchestrator import DraftOrchestrator
    tick("starting orchestrator generate")
    summary = await DraftOrchestrator(retriever).generate(td)
    tick(f"orchestrator done: matter_id={summary.matter_id}, sections={[summary.s1_vesting_and_estate.summary[0].text[:80]]}")

    out_path = Path("data/out") / f"{summary.matter_id}.trace.json"
    out_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    tick(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data/gold/orlando_kobe_apartments_alta_survey.title_document.json"))
