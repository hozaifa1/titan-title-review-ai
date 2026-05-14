from __future__ import annotations

import pytest

from titan.draft.orchestrator import DraftOrchestrator
from titan.index.chunker import ChunkerConfig, chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import (
    ChainOfTitleLink,
    FieldWithProvenance,
    PartyParty,
    Provenance,
    TitleDocument,
)


@pytest.mark.asyncio
async def test_draft_orchestrator_generates_eight_cited_sections_without_api_keys() -> None:
    source = Provenance(doc_id="sample_commitment", page=1, char_span=(0, 120), snippet="Schedule A vested owner Sam Seller")
    document = TitleDocument(
        doc_id="sample_commitment",
        doc_type="title_commitment",
        file_path="sample.pdf",
        page_count=1,
        parsed_at="2026-05-14",
        proposed_insured=FieldWithProvenance(value="Sam Buyer", confidence=0.9, source=source),
        vesting=[PartyParty(name="Sam Seller", role="owner")],
        chain_of_title=[
            ChainOfTitleLink(
                instrument_type="warranty_deed",
                grantor=["Prior Owner"],
                grantee=["Sam Seller"],
                book="12",
                page="34",
                source=source,
            )
        ],
    )
    markdown = """
    SCHEDULE A
    Title to the fee simple estate or interest in the land is vested in Sam Seller.

    SCHEDULE B-I
    Requirements include payoff and recordation matters.

    SCHEDULE B-II
    Exceptions include taxes and standard survey matters.
    """
    chunks = await chunk_title_document(
        document,
        markdown,
        ChunkerConfig(chunk_tokens=80, overlap_tokens=10, use_gemini_context=False),
    )
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    store = HybridChunkStore()
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)

    summary = await DraftOrchestrator(retriever, use_gemini=False).generate(document)

    assert summary.matter_id == "sample_commitment"
    assert summary.s1_vesting_and_estate.section_name == "Vesting and Estate"
    assert len([field for field in type(summary).model_fields if field.startswith("s")]) == 8
    assert summary.s1_vesting_and_estate.summary[0].citations
    assert summary.s1_vesting_and_estate.summary[0].citations[0].doc_id == "sample_commitment"
    assert summary.overall_summary[0].citations
