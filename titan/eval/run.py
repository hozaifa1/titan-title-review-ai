"""Paired-condition eval runner for Titan.

Runs the eight-section draft generator over the held-out eval set twice:

* ``pre`` — no edit memory, no distilled rules.
* ``post`` — operator-edit memory loaded from SQLite + distilled
  ``rules/*.yaml`` injected into the prompt.

For each condition, four metrics are computed per case and averaged across the
set: field-level edit distance, retrieval recall@5, faithfulness, and answer
relevancy. The aggregated report is dumped to
``eval/results_pre.json`` / ``eval/results_post.json``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from titan.config import get_settings
from titan.draft.orchestrator import DraftOrchestrator
from titan.eval.build_set import EvalCase, load_eval_set
from titan.eval.metrics import (
    SectionEditDistance,
    answer_relevancy,
    faithfulness,
    field_edit_distance,
    retrieval_recall_at_k,
)
from titan.index.chunker import chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.learn.distill import RuleStore
from titan.learn.memory import EditMemory
from titan.persist.sqlite import load_edit_events
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import TitleDocument

Condition = Literal["pre", "post"]

EVAL_QUERY = (
    "Summarize the title review in eight ALTA sections: vesting, legal "
    "description, chain of title, open liens, easements and restrictions, "
    "Schedule B-I requirements, Schedule B-II exceptions, and taxes and survey."
)


@dataclass
class CaseResult:
    doc_id: str
    field_edit_distance: float
    retrieval_recall_at_5: float
    faithfulness: float
    answer_relevancy: float
    per_section_edit_distance: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass
class EvalRunResult:
    condition: Condition
    cases: list[CaseResult]
    aggregate: dict[str, float]
    rules_version: str | None
    edit_memory_size: int


@dataclass
class EvalReport:
    pre: EvalRunResult
    post: EvalRunResult
    improvement: dict[str, float]


async def run_eval(
    cases: list[EvalCase] | None = None,
    sqlite_path: Path = Path("data/titan.db"),
    rules_dir: Path = Path("rules"),
    qdrant_url: str | None = None,
    output_dir: Path = Path("eval"),
) -> EvalReport:
    """Run the paired eval and persist ``results_pre.json`` / ``results_post.json``."""

    eval_cases = cases or load_eval_set()
    if not eval_cases:
        raise RuntimeError(
            "No eval cases found. Add gold files to data/gold/<doc_id>.TitleReviewSummary.gold.json."
        )

    pre_result = await _run_condition(
        "pre",
        eval_cases,
        with_learning=False,
        sqlite_path=sqlite_path,
        rules_dir=rules_dir,
        qdrant_url=qdrant_url,
    )
    post_result = await _run_condition(
        "post",
        eval_cases,
        with_learning=True,
        sqlite_path=sqlite_path,
        rules_dir=rules_dir,
        qdrant_url=qdrant_url,
    )
    improvement = _compute_improvement(pre_result.aggregate, post_result.aggregate)
    report = EvalReport(pre=pre_result, post=post_result, improvement=improvement)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_pre.json").write_text(
        _dump(pre_result, improvement=None), encoding="utf-8"
    )
    (output_dir / "results_post.json").write_text(
        _dump(post_result, improvement=improvement), encoding="utf-8"
    )
    return report


async def _run_condition(
    condition: Condition,
    cases: list[EvalCase],
    with_learning: bool,
    sqlite_path: Path,
    rules_dir: Path,
    qdrant_url: str | None,
) -> EvalRunResult:
    rule_store: RuleStore | None = None
    edit_memory: EditMemory | None = None
    rules_version: str | None = None
    edit_memory_size = 0

    if with_learning:
        rule_store = RuleStore(rules_dir)
        rules_version = rule_store.aggregated_version_tag()
        try:
            events = load_edit_events(sqlite_path)
        except Exception:
            events = []
        settings = get_settings()
        edit_memory = EditMemory(
            qdrant_url=qdrant_url or settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key or None,
        )
        if events:
            edit_memory.add_many(events)
            edit_memory_size = len(events)

    case_results: list[CaseResult] = []
    for case in cases:
        case_results.append(
            await _run_case(
                case,
                edit_memory=edit_memory,
                rule_store=rule_store,
                qdrant_url=qdrant_url,
            )
        )

    aggregate = _aggregate(case_results)
    return EvalRunResult(
        condition=condition,
        cases=case_results,
        aggregate=aggregate,
        rules_version=rules_version,
        edit_memory_size=edit_memory_size,
    )


async def _run_case(
    case: EvalCase,
    edit_memory: EditMemory | None,
    rule_store: RuleStore | None,
    qdrant_url: str | None,
) -> CaseResult:
    title_document = case.title_document
    markdown = _document_markdown(title_document)
    chunks = await chunk_title_document(title_document, markdown)
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    settings = get_settings()
    store = HybridChunkStore(
        qdrant_url=qdrant_url or settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
    )
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)

    if edit_memory is not None:
        edit_memory.embedder = dense_embedder

    summary = await DraftOrchestrator(
        retriever,
        edit_memory=edit_memory,
        rule_store=rule_store,
    ).generate(title_document)

    hits = await retriever.retrieve(EVAL_QUERY, top_k=5)
    avg_distance, per_section = field_edit_distance(summary, case.gold_summary)
    recall = retrieval_recall_at_k(case.gold_summary, hits, k=5)
    faith = faithfulness(summary, hits, embedder=dense_embedder)
    relevancy = answer_relevancy(summary, EVAL_QUERY, embedder=dense_embedder)

    return CaseResult(
        doc_id=case.doc_id,
        field_edit_distance=avg_distance,
        retrieval_recall_at_5=recall,
        faithfulness=faith,
        answer_relevancy=relevancy,
        per_section_edit_distance=[_section_distance_dict(item) for item in per_section],
    )


def _section_distance_dict(item: SectionEditDistance) -> dict[str, float | int | str]:
    return {
        "section": item.section,
        "distance": round(item.distance, 4),
        "gold_length": item.gold_length,
        "produced_length": item.produced_length,
    }


def _aggregate(cases: list[CaseResult]) -> dict[str, float]:
    if not cases:
        return {
            "field_edit_distance": 0.0,
            "retrieval_recall_at_5": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }
    n = len(cases)
    return {
        "field_edit_distance": round(sum(c.field_edit_distance for c in cases) / n, 4),
        "retrieval_recall_at_5": round(sum(c.retrieval_recall_at_5 for c in cases) / n, 4),
        "faithfulness": round(sum(c.faithfulness for c in cases) / n, 4),
        "answer_relevancy": round(sum(c.answer_relevancy for c in cases) / n, 4),
    }


def _compute_improvement(
    pre: dict[str, float], post: dict[str, float]
) -> dict[str, float]:
    edit_pre = pre["field_edit_distance"] or 1e-9
    return {
        "field_edit_distance_delta": round(post["field_edit_distance"] - pre["field_edit_distance"], 4),
        "field_edit_distance_pct_reduction": round(
            (pre["field_edit_distance"] - post["field_edit_distance"]) / edit_pre, 4
        ),
        "faithfulness_delta": round(post["faithfulness"] - pre["faithfulness"], 4),
        "answer_relevancy_delta": round(post["answer_relevancy"] - pre["answer_relevancy"], 4),
        "retrieval_recall_at_5_delta": round(
            post["retrieval_recall_at_5"] - pre["retrieval_recall_at_5"], 4
        ),
    }


def _dump(result: EvalRunResult, improvement: dict[str, float] | None) -> str:
    payload: dict[str, object] = {
        "condition": result.condition,
        "rules_version": result.rules_version,
        "edit_memory_size": result.edit_memory_size,
        "aggregate": result.aggregate,
        "cases": [asdict(case) for case in result.cases],
    }
    if improvement is not None:
        payload["improvement_vs_pre"] = improvement
    return json.dumps(payload, indent=2)


def _document_markdown(title_document: TitleDocument) -> str:
    pdf_path = Path(title_document.file_path)
    if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
        try:
            import pdfplumber  # type: ignore[import-not-found]

            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for index, page in enumerate(pdf.pages, start=1):
                    pages.append(f"## Page {index}\n\n{page.extract_text() or ''}")
                return "\n\n".join(pages)
        except Exception:
            pass
    return "\n".join(
        [
            f"# {title_document.doc_id}",
            f"Document type: {title_document.doc_type}",
            "Vesting: " + ", ".join(party.name for party in title_document.vesting),
            "Parties: "
            + ", ".join(
                f"{party.name} ({party.role})" for party in title_document.parties
            ),
            "Warnings: " + "; ".join(title_document.extraction_warnings),
        ]
    )


def render_markdown_table(report: EvalReport) -> str:
    """Render a README-friendly markdown comparison table."""

    rows = [
        ("Faithfulness (higher = better)", "faithfulness", "+"),
        ("Answer relevancy (higher = better)", "answer_relevancy", "+"),
        ("Field-level edit distance (lower = better)", "field_edit_distance", "-"),
        ("Retrieval recall@5", "retrieval_recall_at_5", "+"),
    ]
    lines = [
        "| Metric | Condition A (no learning) | Condition B (with learning) | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, direction in rows:
        pre_value = report.pre.aggregate[key]
        post_value = report.post.aggregate[key]
        if direction == "-":
            pct = report.improvement["field_edit_distance_pct_reduction"]
            delta_str = f"{post_value - pre_value:+.3f} ({pct * 100:.1f}% lower)"
        else:
            delta_str = f"{post_value - pre_value:+.3f}"
        lines.append(f"| {label} | {pre_value:.3f} | {post_value:.3f} | {delta_str} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    asyncio.run(run_eval())


if __name__ == "__main__":  # pragma: no cover
    main()
