"""Edit memory: embed EditEvents and store them in a Qdrant collection.

Backs the dynamic-few-shot half of the learning loop. The same in-process
fallback used by :class:`titan.index.qdrant_store.HybridChunkStore` keeps tests
hermetic; when a real Qdrant URL is supplied, edits are mirrored to a
dedicated ``edit_memory`` collection with payload-side filtering by
``section`` and ``matter_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

from titan.index.embed import DenseEmbedder, cosine
from titan.learn.diff import summarize_for_embedding
from titan.schemas import EditEvent
from titan.telemetry import get_logger

logger = get_logger(__name__)

DEFAULT_COLLECTION = "edit_memory"


@dataclass
class StoredEdit:
    event: EditEvent
    embedding: list[float]


@dataclass
class EditMemory:
    """Searchable store of operator edits.

    The in-process list is the deterministic backend (used by tests and the
    offline fallback). When ``qdrant_url`` is supplied, the same payloads are
    mirrored to Qdrant; reads/writes prefer the live Qdrant client and fall
    back to the in-process list on any error.
    """

    collection_name: str = DEFAULT_COLLECTION
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    embedder: DenseEmbedder = field(default_factory=DenseEmbedder)
    _edits: list[StoredEdit] = field(default_factory=list)
    _qdrant: Any = None
    _qdrant_ready: bool = False
    _next_point_id: int = 0

    @property
    def qdrant_active(self) -> bool:
        return self._qdrant is not None

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return len(self._edits)

    def ensure_qdrant(self) -> None:
        if self._qdrant_ready or not self.qdrant_url:
            return
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.http.models import Distance, VectorParams  # type: ignore[import-not-found]

            client = (
                QdrantClient(location=":memory:")
                if self.qdrant_url == ":memory:"
                else QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            )
            dimension = self.embedder.dimension
            if not client.collection_exists(self.collection_name):
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=dimension, distance=Distance.COSINE),
                    },
                )
            self._qdrant = client
        except Exception as exc:  # pragma: no cover - networked path
            logger.warning("edit_memory.qdrant_disabled", error=str(exc))
            self._qdrant = None
        finally:
            self._qdrant_ready = True

    def add_many(self, events: Iterable[EditEvent]) -> int:
        events_list = list(events)
        if not events_list:
            return 0
        self.ensure_qdrant()
        texts = [summarize_for_embedding(event) for event in events_list]
        vectors = self.embedder.embed(texts)
        for event, vector in zip(events_list, vectors):
            self._edits.append(StoredEdit(event=event, embedding=list(vector)))
        if self._qdrant is not None:
            self._upsert_to_qdrant(events_list, vectors)
        return len(events_list)

    def _upsert_to_qdrant(
        self,
        events_list: list[EditEvent],
        vectors: list[list[float]],
    ) -> None:
        try:
            from qdrant_client.http.models import PointStruct  # type: ignore[import-not-found]

            points = []
            for event, vector in zip(events_list, vectors):
                point_id = self._next_point_id
                self._next_point_id += 1
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={"dense": list(vector)},
                        payload=_event_payload(event),
                    )
                )
            self._qdrant.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # pragma: no cover - networked path
            logger.warning("edit_memory.upsert_failed", error=str(exc))
            self._qdrant = None

    def search(
        self,
        query_text: str,
        section: str | None = None,
        top_k: int = 3,
    ) -> list[EditEvent]:
        if not self._edits and self._qdrant is None:
            return []
        query_vector = self.embedder.embed([query_text])[0]
        if self._qdrant is not None:
            hits = self._search_qdrant(query_vector, section=section, top_k=top_k)
            if hits is not None:
                return hits
        return self._search_local(query_vector, section=section, top_k=top_k)

    def _search_qdrant(
        self,
        query_vector: list[float],
        section: str | None,
        top_k: int,
    ) -> list[EditEvent] | None:
        try:
            from qdrant_client.http.models import (  # type: ignore[import-not-found]
                FieldCondition,
                Filter,
                MatchValue,
            )

            query_filter = None
            if section:
                query_filter = Filter(
                    must=[FieldCondition(key="section_name", match=MatchValue(value=section))]
                )
            response = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
            return [
                _event_from_payload(point.payload or {})
                for point in response.points
            ]
        except Exception as exc:  # pragma: no cover - networked path
            logger.warning("edit_memory.search_failed", error=str(exc))
            return None

    def _search_local(
        self,
        query_vector: list[float],
        section: str | None,
        top_k: int,
    ) -> list[EditEvent]:
        scored: list[tuple[float, EditEvent]] = []
        for stored in self._edits:
            if section and stored.event.section_name != section:
                continue
            scored.append((cosine(query_vector, stored.embedding), stored.event))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [event for _, event in scored[:top_k]]

    def all_events(self, section: str | None = None) -> list[EditEvent]:
        if section:
            return [stored.event for stored in self._edits if stored.event.section_name == section]
        return [stored.event for stored in self._edits]

    def dump_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for stored in self._edits:
                handle.write(stored.event.model_dump_json() + "\n")
        return path

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        embedder: DenseEmbedder | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> "EditMemory":
        memory = cls(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedder=embedder or DenseEmbedder(),
        )
        if not path.exists():
            return memory
        events: list[EditEvent] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(EditEvent.model_validate_json(line))
        memory.add_many(events)
        return memory


def _event_payload(event: EditEvent) -> dict[str, Any]:
    return json.loads(event.model_dump_json())


def _event_from_payload(payload: dict[str, Any]) -> EditEvent:
    return EditEvent.model_validate(payload)


def _json_default(value: Any) -> str:  # pragma: no cover - small helper
    if isinstance(value, (datetime, Decimal)):
        return str(value)
    raise TypeError(type(value))
