"""Titan command-line entrypoints."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from titan.config import get_settings
from titan.draft.orchestrator import DraftOrchestrator
from titan.telemetry import bind_trace_id, configure_logging, get_logger
from titan.eval.run import render_markdown_table, run_eval
from titan.index.chunker import chunk_title_document
from titan.index.embed import DenseEmbedder, embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.ingest.extract import extract_title_document
from titan.ingest.ocr import parse_document
from titan.learn.diff import diff_summaries
from titan.learn.distill import RuleStore, distill_rules_for_section
from titan.learn.memory import EditMemory
from titan.persist.sqlite import (
    load_edit_events,
    persist_edit_events,
    persist_parsed_doc,
    persist_title_document,
)
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import TitleDocument, TitleReviewSummary


DEFAULT_DEMO_DOCS = [
    "data/raw/mortgage/osmre_mortgage_deed_of_trust.pdf",
    "data/raw/deed/fromthepage_1875_handwritten_deed.pdf",
    "data/raw/commitment/wayne_county_commitment_0.pdf",
]


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    trace_id = bind_trace_id()
    log = get_logger("titan.cli")
    log.info("titan.cli.start", trace_id=trace_id, has_gemini=settings.has_gemini)

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

    draft = subparsers.add_parser("draft", help="Generate a cited TitleReviewSummary for one PDF or TitleDocument JSON.")
    draft.add_argument("path")
    draft.add_argument("--out-dir", default="data/out")
    draft.add_argument("--sqlite", default="data/titan.db")
    draft.add_argument("--qdrant-url", default=None)
    draft.add_argument("--rules-dir", default="rules")
    draft.add_argument("--with-learning", action="store_true", help="Inject distilled rules and past edits into the prompt.")
    draft.add_argument("--suffix", default="TitleReviewSummary")

    learn_capture = subparsers.add_parser(
        "learn-capture",
        help="Diff a baseline draft JSON against an operator-edited draft JSON; persist EditEvents.",
    )
    learn_capture.add_argument("baseline")
    learn_capture.add_argument("edited")
    learn_capture.add_argument("--sqlite", default="data/titan.db")
    learn_capture.add_argument("--qdrant-url", default=None)
    learn_capture.add_argument("--operator-id", default=None)
    learn_capture.add_argument("--operator-note", default=None)

    learn_distill = subparsers.add_parser(
        "learn-distill",
        help="Run the LLM-as-judge rule distillation for one section (or all).",
    )
    learn_distill.add_argument("--sqlite", default="data/titan.db")
    learn_distill.add_argument("--rules-dir", default="rules")
    learn_distill.add_argument("--section", default=None, help="Section field_name (e.g. s4_open_encumbrances_and_liens). Default: all sections with edits.")
    learn_distill.add_argument("--window", type=int, default=20)

    eval_run = subparsers.add_parser(
        "eval-run",
        help="Run the held-out paired eval (pre and post learning); writes eval/results_pre.json and eval/results_post.json.",
    )
    eval_run.add_argument("--sqlite", default="data/titan.db")
    eval_run.add_argument("--rules-dir", default="rules")
    eval_run.add_argument("--qdrant-url", default=None)
    eval_run.add_argument("--output-dir", default="eval")

    args = parser.parse_args()
    if args.command == "ingest":
        asyncio.run(_run_one(Path(args.path), Path(args.out_dir), Path(args.sqlite)))
    elif args.command == "demo-ingest":
        paths = [Path(path) for path in (args.paths or DEFAULT_DEMO_DOCS)]
        asyncio.run(_run_many(paths, Path(args.out_dir), Path(args.sqlite)))
    elif args.command == "index-query":
        asyncio.run(_run_index_query(Path(args.docs_dir), args.query, args.top_k, args.qdrant_url))
    elif args.command == "draft":
        asyncio.run(
            _run_draft(
                Path(args.path),
                Path(args.out_dir),
                Path(args.sqlite),
                args.qdrant_url,
                Path(args.rules_dir),
                args.with_learning,
                args.suffix,
            )
        )
    elif args.command == "learn-capture":
        _run_learn_capture(
            Path(args.baseline),
            Path(args.edited),
            Path(args.sqlite),
            args.qdrant_url,
            args.operator_id,
            args.operator_note,
        )
    elif args.command == "learn-distill":
        asyncio.run(
            _run_learn_distill(
                Path(args.sqlite),
                Path(args.rules_dir),
                args.section,
                args.window,
            )
        )
    elif args.command == "eval-run":
        asyncio.run(
            _run_eval_cmd(
                Path(args.sqlite),
                Path(args.rules_dir),
                args.qdrant_url,
                Path(args.output_dir),
            )
        )


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
    settings = get_settings()
    store = HybridChunkStore(
        qdrant_url=qdrant_url or settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
    )
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


async def _run_draft(
    path: Path,
    out_dir: Path,
    sqlite_path: Path,
    qdrant_url: str | None,
    rules_dir: Path,
    with_learning: bool,
    suffix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        title_document = TitleDocument.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        title_document = await _run_one(path, out_dir, sqlite_path)

    markdown = _read_markdown_for(title_document)
    chunks = await chunk_title_document(title_document, markdown)
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    settings = get_settings()
    qdrant_target = qdrant_url or settings.qdrant_url
    store = HybridChunkStore(qdrant_url=qdrant_target, qdrant_api_key=settings.qdrant_api_key or None)
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)

    edit_memory: EditMemory | None = None
    rule_store: RuleStore | None = None
    if with_learning:
        rule_store = RuleStore(rules_dir)
        edit_memory = EditMemory(
            qdrant_url=qdrant_target,
            qdrant_api_key=settings.qdrant_api_key or None,
            embedder=dense_embedder,
        )
        events = load_edit_events(sqlite_path)
        if events:
            edit_memory.add_many(events)

    summary = await DraftOrchestrator(
        retriever,
        edit_memory=edit_memory,
        rule_store=rule_store,
    ).generate(title_document)

    output_path = out_dir / f"{summary.matter_id}.{suffix}.json"
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(path),
                "output": str(output_path),
                "matter_id": summary.matter_id,
                "sections": len([name for name in type(summary).model_fields if name.startswith("s")]),
                "model": summary.model,
                "rules_version": summary.rules_version,
                "with_learning": with_learning,
                "qdrant_mirrored": store.qdrant_mirrored,
                "edit_memory_active": bool(edit_memory and edit_memory.qdrant_active),
                "edit_memory_size": len(edit_memory) if edit_memory else 0,
            },
            indent=2,
        )
    )
    _flush_langfuse()


def _run_learn_capture(
    baseline_path: Path,
    edited_path: Path,
    sqlite_path: Path,
    qdrant_url: str | None,
    operator_id: str | None,
    operator_note: str | None,
) -> None:
    baseline = TitleReviewSummary.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    edited = TitleReviewSummary.model_validate_json(edited_path.read_text(encoding="utf-8"))
    events = diff_summaries(baseline, edited, operator_id=operator_id, operator_note=operator_note)
    if not events:
        print(json.dumps({"events": 0, "message": "No edits detected."}, indent=2))
        return
    persist_edit_events(events, sqlite_path)
    embedder = DenseEmbedder()
    settings = get_settings()
    memory = EditMemory(
        qdrant_url=qdrant_url or settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
        embedder=embedder,
    )
    memory.add_many(events)
    print(
        json.dumps(
            {
                "events": len(events),
                "qdrant_active": memory.qdrant_active,
                "by_section": _events_by_section(events),
                "by_type": _events_by_type(events),
            },
            indent=2,
        )
    )


async def _run_learn_distill(
    sqlite_path: Path,
    rules_dir: Path,
    section: str | None,
    window: int,
) -> None:
    rule_store = RuleStore(rules_dir)
    sections = [section] if section else _sections_with_edits(sqlite_path)
    if not sections:
        print(json.dumps({"sections": 0, "message": "No edits found in SQLite."}, indent=2))
        return

    output: list[dict[str, object]] = []
    for sec in sections:
        events = load_edit_events(sqlite_path, section_name=sec)
        if not events:
            continue
        result = await distill_rules_for_section(sec, events, rule_store, window=window)
        output.append(
            {
                "section": sec,
                "version": result.rule_set.version,
                "used_gemini": result.used_gemini,
                "rules": len(result.rule_set.rules),
                "path": str(rule_store.path_for(sec)),
            }
        )
    print(json.dumps({"distilled": output, "rules_version": rule_store.aggregated_version_tag()}, indent=2))


async def _run_eval_cmd(
    sqlite_path: Path,
    rules_dir: Path,
    qdrant_url: str | None,
    output_dir: Path,
) -> None:
    report = await run_eval(
        sqlite_path=sqlite_path,
        rules_dir=rules_dir,
        qdrant_url=qdrant_url,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "pre": {
                    "aggregate": report.pre.aggregate,
                    "cases": [case.doc_id for case in report.pre.cases],
                    "rules_version": report.pre.rules_version,
                    "edit_memory_size": report.pre.edit_memory_size,
                },
                "post": {
                    "aggregate": report.post.aggregate,
                    "cases": [case.doc_id for case in report.post.cases],
                    "rules_version": report.post.rules_version,
                    "edit_memory_size": report.post.edit_memory_size,
                },
                "improvement": report.improvement,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    print("\n" + render_markdown_table(report))


def _sections_with_edits(sqlite_path: Path) -> list[str]:
    events = load_edit_events(sqlite_path)
    # dict.fromkeys preserves first-seen order while deduplicating in O(n).
    return list(dict.fromkeys(event.section_name for event in events))


def _events_by_section(events: list[Any]) -> dict[str, int]:
    return dict(Counter(event.section_name for event in events))


def _events_by_type(events: list[Any]) -> dict[str, int]:
    return dict(Counter(event.edit_type for event in events))


def _flush_langfuse() -> None:
    try:
        from langfuse import get_client  # type: ignore[import-not-found]

        get_client().flush()
    except Exception:
        pass


def _read_markdown_for(document: TitleDocument) -> str:
    """Delegate to the disk-cached, per-page-timeout reader.

    Centralising here so the CLI and eval paths share one implementation
    that doesn't stall on pathological survey pages.
    """

    from titan.ingest.markdown_cache import read_markdown_for

    return read_markdown_for(document)


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
