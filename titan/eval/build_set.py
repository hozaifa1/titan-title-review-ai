"""Build the held-out evaluation set for Titan.

Pairs three sample documents with their hand-written gold ``TitleReviewSummary``
and the structured ``TitleDocument`` extraction that drives retrieval. Each
gold summary lives in ``data/gold/<doc_id>.TitleReviewSummary.gold.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from titan.schemas import TitleDocument, TitleReviewSummary

DEFAULT_GOLD_DIR = Path("data/gold")
DEFAULT_DOC_DIR = Path("data/out")
DEFAULT_DOC_IDS: tuple[str, ...] = (
    "wayne_county_commitment_0",
    "osmre_mortgage_deed_of_trust",
    "fromthepage_1875_handwritten_deed",
)


@dataclass(frozen=True)
class EvalCase:
    """One held-out evaluation case."""

    doc_id: str
    title_document: TitleDocument
    gold_summary: TitleReviewSummary


def load_eval_set(
    doc_ids: tuple[str, ...] = DEFAULT_DOC_IDS,
    gold_dir: Path = DEFAULT_GOLD_DIR,
    doc_dir: Path = DEFAULT_DOC_DIR,
) -> list[EvalCase]:
    """Load all eval cases.

    A case is included only if both its ``title_document.json`` and gold
    ``TitleReviewSummary.gold.json`` exist on disk.
    """

    cases: list[EvalCase] = []
    for doc_id in doc_ids:
        doc_path = _resolve_title_document_path(doc_id, gold_dir, doc_dir)
        gold_path = gold_dir / f"{doc_id}.TitleReviewSummary.gold.json"
        if not doc_path or not gold_path.exists():
            continue
        title_document = TitleDocument.model_validate_json(
            doc_path.read_text(encoding="utf-8")
        )
        gold_summary = TitleReviewSummary.model_validate_json(
            gold_path.read_text(encoding="utf-8")
        )
        cases.append(
            EvalCase(
                doc_id=doc_id,
                title_document=title_document,
                gold_summary=gold_summary,
            )
        )
    return cases


def _resolve_title_document_path(
    doc_id: str, gold_dir: Path, doc_dir: Path
) -> Path | None:
    candidates = [
        gold_dir / f"{doc_id}.title_document.json",
        doc_dir / f"{doc_id}.title_document.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
