"""Titan command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from titan.ingest.extract import extract_title_document
from titan.ingest.ocr import parse_document
from titan.persist.sqlite import persist_parsed_doc, persist_title_document
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

    args = parser.parse_args()
    if args.command == "ingest":
        asyncio.run(_run_one(Path(args.path), Path(args.out_dir), Path(args.sqlite)))
    elif args.command == "demo-ingest":
        paths = [Path(path) for path in (args.paths or DEFAULT_DEMO_DOCS)]
        asyncio.run(_run_many(paths, Path(args.out_dir), Path(args.sqlite)))


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


if __name__ == "__main__":
    main()
