"""Indexing and retrieval contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from titan.schemas import Provenance


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    text: str
    contextual_text: str
    provenance: Provenance
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    dense: list[float]
    sparse: list[float]


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    source: str
