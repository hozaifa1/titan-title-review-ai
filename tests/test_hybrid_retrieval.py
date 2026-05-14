from __future__ import annotations

import pytest

from titan.index.chunker import ChunkerConfig, chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import PartyParty, TitleDocument


@pytest.mark.asyncio
async def test_vested_owner_query_returns_schedule_a_chunk() -> None:
    document = TitleDocument(
        doc_id="sample_commitment",
        doc_type="title_commitment",
        file_path="sample.pdf",
        page_count=1,
        parsed_at="2026-05-14",
        vesting=[PartyParty(name="Sam Seller", role="owner")],
    )
    markdown = """
    SCHEDULE A
    4. Title to the fee simple estate or interest in the land is at the
    Effective Date vested in:
    Sam Seller
    """
    chunks = await chunk_title_document(
        document,
        markdown,
        ChunkerConfig(chunk_tokens=80, overlap_tokens=10, use_gemini_context=False),
    )
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    store = HybridChunkStore()
    store.upsert(embedded, bm25)

    hits = await HybridRetriever(store, dense_embedder, bm25).retrieve("Who is the vested owner?", top_k=5)

    assert hits
    assert any("Sam Seller" in hit.chunk.text for hit in hits)
    assert hits[0].chunk.provenance.doc_id == "sample_commitment"


@pytest.mark.asyncio
async def test_qdrant_mirror_uses_named_vectors_and_provenance_payload() -> None:
    document = TitleDocument(
        doc_id="sample_commitment",
        doc_type="title_commitment",
        file_path="sample.pdf",
        page_count=1,
        parsed_at="2026-05-14",
        vesting=[PartyParty(name="Sam Seller", role="owner")],
    )
    markdown = """
    SCHEDULE A
    Title to the fee simple estate or interest in the land is vested in Sam Seller.
    """
    chunks = await chunk_title_document(
        document,
        markdown,
        ChunkerConfig(chunk_tokens=80, overlap_tokens=10, use_gemini_context=False),
    )
    embedded, bm25, _dense_embedder = embed_chunks(chunks)
    store = HybridChunkStore(qdrant_url=":memory:")
    store.upsert(embedded, bm25)

    assert store.qdrant_mirrored
    collection = store._qdrant.get_collection(store.collection_name)
    vectors = collection.config.params.vectors
    assert set(vectors) == {"dense", "sparse"}

    points, _next_page = store._qdrant.scroll(store.collection_name, limit=1, with_payload=True)
    assert points
    assert points[0].payload["doc_id"] == "sample_commitment"
    assert points[0].payload["doc_type"] == "title_commitment"
    assert points[0].payload["provenance"]["doc_id"] == "sample_commitment"
