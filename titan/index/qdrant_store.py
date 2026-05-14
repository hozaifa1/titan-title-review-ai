"""Hybrid chunk store with optional Qdrant backing."""

from __future__ import annotations

from typing import Any

from titan.index.embed import SimpleBM25, cosine
from titan.index.models import Chunk, EmbeddedChunk, SearchHit
from titan.schemas import Provenance


class HybridChunkStore:
    """Stores chunks and exposes dense/sparse search.

    The in-process path is the deterministic checkpoint backend. If
    `qdrant-client` is installed and a URL is supplied, the same payloads are
    mirrored to a Qdrant collection with named vectors `dense` and `sparse`.
    Use ``qdrant_url=":memory:"`` for a no-Docker verification backend.
    """

    def __init__(
        self,
        collection_name: str = "title_chunks",
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embedded: list[EmbeddedChunk] = []
        self.bm25: SimpleBM25 | None = None
        self._qdrant: Any | None = None

    @property
    def qdrant_mirrored(self) -> bool:
        return self._qdrant is not None

    def upsert(self, embedded: list[EmbeddedChunk], bm25: SimpleBM25) -> None:
        self.embedded = embedded
        self.bm25 = bm25
        if self.qdrant_url:
            self._mirror_to_qdrant()

    def dense_search(self, query_dense: list[float], top_k: int = 30) -> list[SearchHit]:
        if self._qdrant is not None:
            try:
                response = self._qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_dense,
                    using="dense",
                    limit=top_k,
                    with_payload=True,
                )
                return _hits_from_qdrant(response.points, "qdrant-dense")
            except Exception:
                pass
        scored = [
            SearchHit(chunk=item.chunk, score=cosine(query_dense, item.dense), rank=0, source="dense")
            for item in self.embedded
        ]
        return _rank(scored, top_k)

    def sparse_search(self, query: str, query_sparse: list[float] | None = None, top_k: int = 30) -> list[SearchHit]:
        if self._qdrant is not None and query_sparse is not None:
            try:
                response = self._qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_sparse,
                    using="sparse",
                    limit=top_k,
                    with_payload=True,
                )
                return _hits_from_qdrant(response.points, "qdrant-sparse")
            except Exception:
                pass
        scored: list[SearchHit] = []
        for index, item in enumerate(self.embedded):
            bm25_score = self.bm25.score(query, index) if self.bm25 else 0.0
            sparse_score = cosine(query_sparse, item.sparse) if query_sparse else 0.0
            scored.append(SearchHit(chunk=item.chunk, score=bm25_score + sparse_score, rank=0, source="sparse"))
        return _rank(scored, top_k)

    def _mirror_to_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore[import-not-found]

            client = (
                QdrantClient(location=":memory:")
                if self.qdrant_url == ":memory:"
                else QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            )
            dense_size = len(self.embedded[0].dense) if self.embedded else 1
            sparse_size = len(self.embedded[0].sparse) if self.embedded else 1
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=dense_size, distance=Distance.COSINE),
                    "sparse": VectorParams(size=sparse_size, distance=Distance.COSINE),
                },
            )
            points = [
                PointStruct(
                    id=index,
                    vector={"dense": item.dense, "sparse": item.sparse},
                    payload={
                        "chunk_id": item.chunk.chunk_id,
                        "doc_id": item.chunk.doc_id,
                        "doc_type": item.chunk.doc_type,
                        "provenance": item.chunk.provenance.model_dump(mode="json"),
                        "text": item.chunk.text,
                        "contextual_text": item.chunk.contextual_text,
                    },
                )
                for index, item in enumerate(self.embedded)
            ]
            client.upsert(collection_name=self.collection_name, points=points)
            self._qdrant = client
        except Exception:
            self._qdrant = None


def _rank(scored: list[SearchHit], top_k: int) -> list[SearchHit]:
    ranked = sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k]
    return [hit.model_copy(update={"rank": index + 1}) for index, hit in enumerate(ranked)]


def _hits_from_qdrant(points: list[Any], source: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        provenance = Provenance.model_validate(payload.get("provenance") or {})
        chunk = Chunk(
            chunk_id=str(payload.get("chunk_id")),
            doc_id=str(payload.get("doc_id")),
            doc_type=str(payload.get("doc_type")),
            text=str(payload.get("text") or ""),
            contextual_text=str(payload.get("contextual_text") or payload.get("text") or ""),
            provenance=provenance,
        )
        hits.append(SearchHit(chunk=chunk, score=float(point.score), rank=index, source=source))
    return hits
