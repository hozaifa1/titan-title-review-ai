"""SQLite persistence round-trips."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from titan.ingest.models import ParsedDoc, ParsedPage
from titan.persist.sqlite import (
    load_edit_events,
    persist_edit_events,
    persist_parsed_doc,
    persist_title_document,
)
from titan.schemas import EditEvent, TitleDocument


def _sqlite_path(tmp_path: Path) -> Path:
    return tmp_path / "titan.db"


def test_parsed_doc_persists_and_is_idempotent(tmp_path: Path) -> None:
    db = _sqlite_path(tmp_path)
    parsed = ParsedDoc(
        doc_id="d1",
        file_path="d1.pdf",
        markdown="hello",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="hello",
                parser="pdfplumber",
                confidence=0.9,
                classification="clean_text",
            )
        ],
        parser_chain=["pdfplumber"],
    )
    record_a = persist_parsed_doc(parsed, db)
    record_b = persist_parsed_doc(parsed, db)
    assert record_a.id == record_b.id
    assert record_b.markdown == "hello"
    assert record_b.parser_chain == ["pdfplumber"]


def test_title_document_persists(tmp_path: Path) -> None:
    db = _sqlite_path(tmp_path)
    doc = TitleDocument(
        doc_id="d1",
        doc_type="title_commitment",
        file_path="d1.pdf",
        page_count=1,
        parsed_at=date(2026, 5, 14),
    )
    record = persist_title_document(doc, db)
    assert record.doc_id == "d1"
    assert record.payload["doc_type"] == "title_commitment"


def test_edit_events_round_trip(tmp_path: Path) -> None:
    db = _sqlite_path(tmp_path)
    events = [
        EditEvent(
            matter_id="m1",
            section_name="s1_vesting_and_estate",
            field_path="summary[0].text",
            before="X",
            after="Y",
        ),
        EditEvent(
            matter_id="m1",
            section_name="s4_open_encumbrances_and_liens",
            field_path="summary[0].text",
            before="A",
            after="B",
        ),
    ]
    persist_edit_events(events, db)

    loaded_all = load_edit_events(db)
    assert len(loaded_all) == 2

    loaded_s4 = load_edit_events(db, section_name="s4_open_encumbrances_and_liens")
    assert len(loaded_s4) == 1
    assert loaded_s4[0].after == "B"


def test_load_edit_events_filters_by_matter(tmp_path: Path) -> None:
    db = _sqlite_path(tmp_path)
    persist_edit_events(
        [
            EditEvent(
                matter_id="m1",
                section_name="s1_vesting_and_estate",
                field_path="x",
                before="a",
                after="b",
            ),
            EditEvent(
                matter_id="m2",
                section_name="s1_vesting_and_estate",
                field_path="x",
                before="a",
                after="b",
            ),
        ],
        db,
    )
    only_m1 = load_edit_events(db, matter_id="m1")
    assert len(only_m1) == 1
    assert only_m1[0].matter_id == "m1"
