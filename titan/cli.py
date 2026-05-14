"""Titan command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from titan.index.chunker import chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.ingest.extract import extract_title_document
from titan.ingest.ocr import parse_document
from titan.persist.sqlite import persist_parsed_doc, persist_title_document
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import TitleDocument


DEFAULT_DEMO_DOCS = [
    "data/raw/mortgage/osmre_mortgage_deed_of_trust.pdf",
    "data/raw/deed/fromthepage_1875_handwritten_deed.pdf",
    "data/raw/commitment/wayne_county_commitment_0.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="titan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse and extract one document.")
    ingest.add_argument("path")
    ingest.add_argument("--out-dir", default="data/out")
    ingest.add_argument("--sqlite", default="data/titan.db")

    demo = subparsers.add_parser("demo-ingest", help="Run the Hour 3-6 checkpoint on three sample docs.")
    demo.add_argument("--out-dir", default="data/out")
    demo.add_argument("--sqlite", default="data/titan.db")
    demo.add_argument("paths", nargs="*")

    index_query = subparsers.add_parser("index-query", help="Build a local hybrid index and run a checkpoint query.")
    index_query.add_argument("--docs-dir", default="data/out")
    index_query.add_argument("--query", default="Who is the vested owner?")
    index_query.add_argument("--top-k", type=int, default=5)
    index_query.add_argument("--qdrant-url", default=None)

    args = parser.parse_args()
    if args.command == "ingest":
        asyncio.run(_run_one(Path(args.path), Path(args.out_dir), Path(args.sqlite)))
    elif args.command == "demo-ingest":
        paths = [Path(path) for path in (args.paths or DEFAULT_DEMO_DOCS)]
        asyncio.run(_run_many(paths, Path(args.out_dir), Path(args.sqlite)))
    elif args.command == "index-query":
        asyncio.run(_run_index_query(Path(args.docs_dir), args.query, args.top_k, args.qdrant_url))


async def _run_many(paths: list[Path], out_dir: Path, sqlite_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        title_document = await _run_one(path, out_dir, sqlite_path)
        results.append(_checkpoint_result(title_document, out_dir))
    print(json.dumps(results, indent=2))


async def _run_one(path: Path, out_dir: Path, sqlite_path: Path) -> TitleDocument:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_doc = await parse_document(path)
    title_document = await extract_title_document(parsed_doc)
    persist_parsed_doc(parsed_doc, sqlite_path)
    persist_title_document(title_document, sqlite_path)

    output_path = out_dir / f"{title_document.doc_id}.title_document.json"
    output_path.write_text(title_document.model_dump_json(indent=2), encoding="utf-8")
    return title_document


def _checkpoint_result(title_document: TitleDocument, out_dir: Path) -> dict[str, object]:
    output_path = out_dir / f"{title_document.doc_id}.title_document.json"
    gold_path = Path("data/gold") / f"{title_document.doc_id}.title_document.json"
    result: dict[str, object] = {
        "doc_id": title_document.doc_id,
        "output": str(output_path),
        "doc_type": title_document.doc_type,
        "has_gold": gold_path.exists(),
    }
    if gold_path.exists():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        produced = json.loads(output_path.read_text(encoding="utf-8"))
        result["reasonable_diff"] = _reasonable_diff(gold, produced)
    return result


def _reasonable_diff(gold: dict, produced: dict) -> dict[str, object]:
    fields = ["doc_id", "doc_type", "page_count", "parties", "chain_of_title", "open_liens"]
    changed = [field for field in fields if gold.get(field) != produced.get(field)]
    return {
        "matched_fields": [field for field in fields if field not in changed],
        "changed_fields": changed,
        "passes": len(changed) <= 1,
    }


async def _run_index_query(docs_dir: Path, query: str, top_k: int, qdrant_url: str | None) -> None:
    documents = [
        TitleDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.glob("*.title_document.json"))
    ]
    if not documents:
        raise SystemExit(f"No title document JSON files found in {docs_dir}")

    chunks = []
    for document in documents:
        markdown = _read_markdown_for(document)
        chunks.extend(await chunk_title_document(document, markdown))

    embedded, bm25, dense_embedder = embed_chunks(chunks)
    store = HybridChunkStore(qdrant_url=qdrant_url)
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)
    hits = await retriever.retrieve(query, top_k=top_k)

    print(
        json.dumps(
            {
                "query": query,
                "chunk_count": len(chunks),
                "dense_backend": dense_embedder.backend,
                "collection": store.collection_name,
                "qdrant_mirrored": store.qdrant_mirrored,
                "hits": [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "source": hit.source,
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "doc_type": hit.chunk.doc_type,
                        "provenance": hit.chunk.provenance.model_dump(mode="json"),
                        "text": hit.chunk.text[:1200],
                    }
                    for hit in hits
                ],
            },
            indent=2,
        )
    )


def _read_markdown_for(document: TitleDocument) -> str:
    path = Path(document.file_path)
    if path.exists() and path.suffix.lower() == ".pdf":
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                pages = []
                for index, page in enumerate(pdf.pages, start=1):
                    pages.append(f"## Page {index}\n\n{page.extract_text() or ''}")
                return "\n\n".join(pages)
        except Exception:
            pass

    return _document_to_markdown(document)


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


if __name__ == "__main__":
    main()
