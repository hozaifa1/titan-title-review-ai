"""Tests for the edit-capture and learning loop."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from titan.draft.orchestrator import DraftOrchestrator, SECTION_SPECS
from titan.index.chunker import ChunkerConfig, chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.learn.diff import diff_summaries
from titan.learn.distill import RuleStore, distill_rules_for_section
from titan.learn.memory import EditMemory
from titan.persist.sqlite import load_edit_events, persist_edit_events
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import (
    Citation,
    CitedSentence,
    EditEvent,
    FieldWithProvenance,
    PartyParty,
    Provenance,
    Rule,
    RuleSet,
    TitleDocument,
    TitleReviewSection,
    TitleReviewSummary,
)


def _provenance() -> Provenance:
    return Provenance(doc_id="m1", page=1, char_span=(0, 10), snippet="x")


def _citation() -> Citation:
    return Citation(doc_id="m1", page=1, char_span=(0, 10), snippet="x")


def _section(section_name: str, summary_text: str, *, gaps: list[str] | None = None) -> TitleReviewSection:
    return TitleReviewSection(
        section_name=section_name,
        summary=[CitedSentence(text=summary_text, citations=[_citation()], confidence="medium")],
        bullet_findings=[],
        gaps=gaps or [],
        flags=["green" if not gaps else "yellow"],
    )


def _make_summary(text_factory) -> TitleReviewSummary:
    return TitleReviewSummary(
        matter_id="m1",
        generated_at=date(2026, 5, 14),
        generator_version="draft-v1",
        model="test-model",
        s1_vesting_and_estate=_section("Vesting and Estate", text_factory("s1")),
        s2_legal_description=_section("Legal Description", text_factory("s2")),
        s3_chain_of_title=_section("Chain of Title", text_factory("s3")),
        s4_open_encumbrances_and_liens=_section("Open Encumbrances and Liens", text_factory("s4")),
        s5_easements_and_restrictions=_section("Easements and Restrictions", text_factory("s5")),
        s6_requirements_schedule_b_i=_section("Requirements", text_factory("s6")),
        s7_exceptions_schedule_b_ii=_section("Exceptions", text_factory("s7")),
        s8_taxes_and_survey_matters=_section("Taxes", text_factory("s8")),
        overall_risk="curable_issues",
        overall_summary=[CitedSentence(text="overall", citations=[_citation()], confidence="medium")],
        open_questions_for_client=[],
    )


def test_diff_summaries_detects_wording_and_citation_fixes() -> None:
    baseline = _make_summary(lambda k: f"{k} original wording")
    edited = _make_summary(lambda k: f"{k} original wording")
    edited.s3_chain_of_title.summary[0] = CitedSentence(
        text="Chain of title includes 1 deed recorded at Book 12, Page 34.",
        citations=[_citation()],
        confidence="medium",
    )
    edited.s4_open_encumbrances_and_liens.gaps.append("Confirm payoff of HOA lien.")
    edited.s8_taxes_and_survey_matters.flags = ["yellow"]

    events = diff_summaries(baseline, edited, operator_id="op1", operator_note="house style")

    by_section = {event.section_name for event in events}
    assert "s3_chain_of_title" in by_section
    assert "s4_open_encumbrances_and_liens" in by_section
    assert "s8_taxes_and_survey_matters" in by_section
    assert all(event.matter_id == "m1" for event in events)
    chain_event = next(event for event in events if event.section_name == "s3_chain_of_title")
    assert chain_event.edit_type in {"citation_fix", "fact_correction"}
    flag_event = next(event for event in events if event.section_name == "s8_taxes_and_survey_matters")
    assert flag_event.edit_type == "risk_rating"


def test_edit_memory_search_filters_by_section() -> None:
    memory = EditMemory()
    memory.add_many(
        [
            EditEvent(
                matter_id="m1",
                section_name="s3_chain_of_title",
                field_path="summary[0].text",
                before="Generic chain summary",
                after="Always cite Book 12, Page 34 for warranty deed.",
                edit_type="citation_fix",
            ),
            EditEvent(
                matter_id="m1",
                section_name="s4_open_encumbrances_and_liens",
                field_path="summary[0].text",
                before="Open liens unknown.",
                after="One open mortgage with current balance of $250,000.",
                edit_type="fact_correction",
            ),
        ]
    )

    hits = memory.search("chain of title book page", section="s3_chain_of_title", top_k=3)
    assert len(hits) == 1
    assert hits[0].section_name == "s3_chain_of_title"

    # cross-section query returns 0 from the filtered section
    no_hits = memory.search("query irrelevant", section="s7_exceptions_schedule_b_ii", top_k=3)
    assert no_hits == []


@pytest.mark.asyncio
async def test_distill_offline_produces_versioned_yaml(tmp_path: Path) -> None:
    rule_store = RuleStore(tmp_path)
    events = [
        EditEvent(
            matter_id="m1",
            section_name="s4_open_encumbrances_and_liens",
            field_path="summary[0].text",
            before="Mortgage exists.",
            after="Open mortgage recorded at Book 12, Page 34, balance $250,000.",
            edit_type="citation_fix",
        ),
        EditEvent(
            matter_id="m1",
            section_name="s4_open_encumbrances_and_liens",
            field_path="flags[0]",
            before="green",
            after="yellow",
            edit_type="risk_rating",
        ),
    ]

    result = await distill_rules_for_section(
        "s4_open_encumbrances_and_liens", events, rule_store, use_gemini=False
    )

    yaml_path = rule_store.path_for("s4_open_encumbrances_and_liens")
    assert yaml_path.exists()
    assert result.rule_set.version == 1
    assert result.rule_set.rules
    # Re-distillation bumps version
    result2 = await distill_rules_for_section(
        "s4_open_encumbrances_and_liens", events, rule_store, use_gemini=False
    )
    assert result2.rule_set.version == 2

    loaded = rule_store.load("s4_open_encumbrances_and_liens")
    assert loaded is not None
    assert loaded.version == 2


def test_persist_edit_events_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "titan.db"
    events = [
        EditEvent(
            matter_id="m1",
            section_name="s4_open_encumbrances_and_liens",
            field_path="summary[0].text",
            before="before",
            after="after",
            edit_type="wording",
        )
    ]
    persist_edit_events(events, db)
    loaded = load_edit_events(db, section_name="s4_open_encumbrances_and_liens")
    assert len(loaded) == 1
    assert loaded[0].matter_id == "m1"
    assert loaded[0].before == "before"
    assert loaded[0].after == "after"


@pytest.mark.asyncio
async def test_orchestrator_injects_rules_into_offline_draft(tmp_path: Path) -> None:
    # Build a tiny doc + retriever like the existing draft test
    source = Provenance(doc_id="m1", page=1, char_span=(0, 30), snippet="vested owner")
    document = TitleDocument(
        doc_id="m1",
        doc_type="title_commitment",
        file_path="sample.pdf",
        page_count=1,
        parsed_at="2026-05-14",
        proposed_insured=FieldWithProvenance(value="Sam Buyer", confidence=0.9, source=source),
        vesting=[PartyParty(name="Sam Seller", role="owner")],
    )
    chunks = await chunk_title_document(
        document,
        "Vested owner Sam Seller. Open mortgage recorded Book 12 Page 34.",
        ChunkerConfig(chunk_tokens=40, overlap_tokens=5, use_gemini_context=False),
    )
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    store = HybridChunkStore()
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)

    rule_store = RuleStore(tmp_path)
    rule_set = RuleSet(
        section="s4_open_encumbrances_and_liens",
        version=1,
        rules=[
            Rule(
                id="r1",
                text="Always reference Book and Page when describing recorded liens.",
                trigger_edit_types=["citation_fix"],
                confidence=0.9,
            )
        ],
    )
    rule_store.save(rule_set)

    memory = EditMemory(embedder=dense_embedder)
    memory.add_many(
        [
            EditEvent(
                matter_id="m1",
                section_name="s4_open_encumbrances_and_liens",
                field_path="summary[0].text",
                before="The source extraction identifies 0 open lien or encumbrance record(s).",
                after="Open mortgage recorded at Book 12, Page 34.",
                edit_type="citation_fix",
            )
        ]
    )

    orchestrator = DraftOrchestrator(
        retriever,
        use_gemini=False,
        edit_memory=memory,
        rule_store=rule_store,
    )
    summary = await orchestrator.generate(document)

    assert summary.rules_version == "rules-v1"
    section = summary.s4_open_encumbrances_and_liens
    # When a prior operator edit for summary[0].text is in edit_memory, the
    # offline fallback adopts the operator's reviewed wording verbatim.
    summary_text = section.summary[0].text
    assert "Book 12, Page 34" in summary_text
