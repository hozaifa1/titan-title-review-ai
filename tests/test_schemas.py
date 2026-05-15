"""Pydantic schema invariants for the title domain."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from titan.schemas import (
    Citation,
    CitedSentence,
    EditEvent,
    PartyParty,
    Provenance,
    TitleDocument,
    TitleReviewSection,
    TitleReviewSummary,
)


def test_provenance_is_minimally_typed() -> None:
    prov = Provenance(doc_id="d1", page=1, char_span=(0, 10))
    assert prov.doc_id == "d1"
    assert prov.page == 1
    assert prov.char_span == (0, 10)


def test_cited_sentence_normalises_string_input() -> None:
    section = TitleReviewSection.model_validate(
        {
            "section_name": "Vesting",
            "summary": ["Bare string finding without citations"],
            "bullet_findings": [],
        }
    )
    assert isinstance(section.summary[0], CitedSentence)
    assert section.summary[0].citations == []
    assert section.summary[0].confidence == "medium"


def test_flag_field_coerces_scalar_to_list() -> None:
    section = TitleReviewSection.model_validate(
        {
            "section_name": "Vesting",
            "summary": [],
            "flags": "green",
        }
    )
    assert section.flags == ["green"]


def test_edit_event_auto_assigns_id() -> None:
    event = EditEvent(
        matter_id="m1",
        section_name="s1_vesting_and_estate",
        field_path="summary[0].text",
        before="X",
        after="Y",
    )
    assert event.edit_id.startswith("e_")
    assert event.edit_type == "wording"
    assert event.source_draft_version == "draft-v1"


def test_party_role_constrained() -> None:
    PartyParty(name="Sam", role="grantor")
    with pytest.raises(ValidationError):
        PartyParty(name="Sam", role="not-a-real-role")  # type: ignore[arg-type]


def test_title_document_accepts_minimal_payload() -> None:
    doc = TitleDocument(
        doc_id="d1",
        doc_type="title_commitment",
        file_path="d1.pdf",
        page_count=1,
        parsed_at="2026-05-14",
    )
    assert doc.doc_id == "d1"
    assert doc.parties == []


def test_summary_round_trips_through_json() -> None:
    summary = TitleReviewSummary(
        matter_id="m1",
        generated_at=date(2026, 5, 14),
        generator_version="draft-v1",
        model="test",
        s1_vesting_and_estate=TitleReviewSection(section_name="s1", summary=[]),
        s2_legal_description=TitleReviewSection(section_name="s2", summary=[]),
        s3_chain_of_title=TitleReviewSection(section_name="s3", summary=[]),
        s4_open_encumbrances_and_liens=TitleReviewSection(section_name="s4", summary=[]),
        s5_easements_and_restrictions=TitleReviewSection(section_name="s5", summary=[]),
        s6_requirements_schedule_b_i=TitleReviewSection(section_name="s6", summary=[]),
        s7_exceptions_schedule_b_ii=TitleReviewSection(section_name="s7", summary=[]),
        s8_taxes_and_survey_matters=TitleReviewSection(section_name="s8", summary=[]),
        overall_risk="clear_to_close",
        overall_summary=[
            CitedSentence(
                text="Vesting confirmed.",
                citations=[
                    Citation(doc_id="d1", page=1, char_span=(0, 10), snippet="x")
                ],
                confidence="high",
            )
        ],
    )
    blob = summary.model_dump_json()
    parsed = TitleReviewSummary.model_validate_json(blob)
    assert parsed.matter_id == "m1"
    assert parsed.overall_summary[0].text == "Vesting confirmed."
