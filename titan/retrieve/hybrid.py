"""Parallel hybrid retrieval with RRF fusion and reranking."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from titan.config import get_settings
from titan.index.embed import DenseEmbedder, SimpleBM25
from titan.index.models import SearchHit
from titan.index.qdrant_store import HybridChunkStore


class HybridRetriever:
    def __init__(
        self,
        store: HybridChunkStore,
        dense_embedder: DenseEmbedder,
        bm25: SimpleBM25,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.dense_embedder = dense_embedder
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self._reranker = _Reranker()

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchHit]:
        dense_query = self.dense_embedder.embed([query])[0]
        sparse_query = self.bm25.query_vector(query)
        dense_task = asyncio.to_thread(self.store.dense_search, dense_query, 30)
        sparse_task = asyncio.to_thread(self.store.sparse_search, query, sparse_query, 30)
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        fused = self._rrf([dense_hits, sparse_hits], limit=20)
        return self._reranker.rerank(query, fused, top_k)

    def _rrf(self, rankings: list[list[SearchHit]], limit: int) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        best: dict[str, SearchHit] = {}
        for hits in rankings:
            for hit in hits:
                scores[hit.chunk.chunk_id] += 1.0 / (self.rrf_k + hit.rank)
                best.setdefault(hit.chunk.chunk_id, hit)

        fused = [
            best[chunk_id].model_copy(update={"score": score, "rank": 0, "source": "rrf"})
            for chunk_id, score in scores.items()
        ]
        ranked = sorted(fused, key=lambda hit: hit.score, reverse=True)[:limit]
        return [hit.model_copy(update={"rank": index + 1}) for index, hit in enumerate(ranked)]


class _Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.backend = "lexical-fallback"
        self._model = None
        if get_settings().use_local_models:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

                self._model = CrossEncoder(model_name)
                self.backend = model_name
            except Exception:
                self._model = None

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if self._model is not None and hits:
            pairs = [(query, hit.chunk.contextual_text) for hit in hits]
            scores = self._model.predict(pairs)
            rescored = [
                hit.model_copy(update={"score": float(score), "source": "bge-reranker-v2-m3"})
                for hit, score in zip(hits, scores)
            ]
        else:
            query_terms = set(query.lower().split())
            rescored = []
            for hit in hits:
                text_terms = set(hit.chunk.contextual_text.lower().split())
                overlap = len(query_terms & text_terms)
                bonus = 2.0 if "vested" in hit.chunk.contextual_text.lower() else 0.0
                rescored.append(hit.model_copy(update={"score": hit.score + overlap + bonus, "source": "lexical-reranker"}))

        ranked = sorted(rescored, key=lambda hit: hit.score, reverse=True)[:top_k]
        return [hit.model_copy(update={"rank": index + 1}) for index, hit in enumerate(ranked)]
