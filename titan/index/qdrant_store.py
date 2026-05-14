"""Hybrid chunk store with optional Qdrant backing."""

from __future__ import annotations

from titan.index.embed import SimpleBM25, cosine
from titan.index.models import EmbeddedChunk, SearchHit


class HybridChunkStore:
    """Stores chunks and exposes dense/sparse search.

    The in-process path is the deterministic checkpoint backend. If
    `qdrant-client` is installed and a URL is supplied, the same payloads are
    mirrored to a Qdrant collection with named vectors `dense` and `sparse`.
    """

    def __init__(self, collection_name: str = "title_chunks", qdrant_url: str | None = None) -> None:
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embedded: list[EmbeddedChunk] = []
        self.bm25: SimpleBM25 | None = None
        self._qdrant = None

    @property
    def qdrant_mirrored(self) -> bool:
        return self._qdrant is not None

    def upsert(self, embedded: list[EmbeddedChunk], bm25: SimpleBM25) -> None:
        self.embedded = embedded
        self.bm25 = bm25
        if self.qdrant_url:
            self._mirror_to_qdrant()

    def dense_search(self, query_dense: list[float], top_k: int = 30) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=item.chunk, score=cosine(query_dense, item.dense), rank=0, source="dense")
            for item in self.embedded
        ]
        return _rank(scored, top_k)

    def sparse_search(self, query: str, query_sparse: list[float] | None = None, top_k: int = 30) -> list[SearchHit]:
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

            client = QdrantClient(url=self.qdrant_url)
            dense_size = len(self.embedded[0].dense) if self.embedded else 1
            sparse_size = len(self.embedded[0].sparse) if self.embedded else 1
            client.recreate_collection(
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
