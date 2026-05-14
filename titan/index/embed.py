"""Dense and sparse embedding utilities."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from titan.index.models import Chunk, EmbeddedChunk

DEFAULT_DENSE_DIM = 384
DEFAULT_SPARSE_DIM = 1024


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseEmbedder:
    """BGE-M3 dense embeddings with a deterministic local fallback."""

    def __init__(self, model_name: str = "BAAI/bge-m3", fallback_dim: int = DEFAULT_DENSE_DIM) -> None:
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self._model = None
        if os.getenv("TITAN_LOCAL_MODELS") == "1":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

                self._model = SentenceTransformer(model_name)
                self.dimension = int(self._model.get_sentence_embedding_dimension() or fallback_dim)
                self.backend = model_name
                return
            except Exception:
                pass
        self.dimension = fallback_dim
        self.backend = "hashing-fallback"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [list(map(float, vector)) for vector in vectors]
        return [self._hashing_vector(text) for text in texts]

    def _hashing_vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            index = _stable_hash(token) % self.dimension
            sign = -1.0 if (_stable_hash(f"{token}:sign") % 2) else 1.0
            vector[index] += sign * (1.0 + math.log(count))
        return _normalize(vector)


@dataclass
class SimpleBM25:
    """Small in-process BM25 index used for sparse retrieval."""

    documents: list[list[str]]
    avgdl: float
    idf: dict[str, float]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "SimpleBM25":
        documents = [tokenize(text) for text in texts]
        doc_count = max(len(documents), 1)
        avgdl = sum(len(doc) for doc in documents) / doc_count
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        idf = {
            term: math.log(1.0 + ((doc_count - freq + 0.5) / (freq + 0.5)))
            for term, freq in doc_freq.items()
        }
        return cls(documents=documents, avgdl=avgdl or 1.0, idf=idf)

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        if doc_index >= len(self.documents):
            return 0.0
        doc = self.documents[doc_index]
        freqs = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for term in query_terms:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * ((tf * (self.k1 + 1.0)) / denom)
        return score

    def query_vector(self, query: str, dimension: int = DEFAULT_SPARSE_DIM) -> list[float]:
        vector = [0.0] * dimension
        for term, freq in Counter(tokenize(query)).items():
            vector[_stable_hash(term) % dimension] += self.idf.get(term, 1.0) * freq
        return _normalize(vector)

    def document_vector(self, doc_index: int, dimension: int = DEFAULT_SPARSE_DIM) -> list[float]:
        vector = [0.0] * dimension
        if doc_index >= len(self.documents):
            return vector
        for term, freq in Counter(self.documents[doc_index]).items():
            vector[_stable_hash(term) % dimension] += self.idf.get(term, 1.0) * (1.0 + math.log(freq))
        return _normalize(vector)


def embed_chunks(
    chunks: list[Chunk],
    dense_embedder: DenseEmbedder | None = None,
    sparse_dim: int = DEFAULT_SPARSE_DIM,
) -> tuple[list[EmbeddedChunk], SimpleBM25, DenseEmbedder]:
    embedder = dense_embedder or DenseEmbedder()
    texts = [chunk.contextual_text for chunk in chunks]
    bm25 = SimpleBM25.fit(texts)
    dense_vectors = embedder.embed(texts)
    embedded = [
        EmbeddedChunk(chunk=chunk, dense=dense_vectors[index], sparse=bm25.document_vector(index, sparse_dim))
        for index, chunk in enumerate(chunks)
    ]
    return embedded, bm25, embedder


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _stable_hash(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
