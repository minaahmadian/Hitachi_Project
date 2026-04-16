from __future__ import annotations

import math
from typing import Any

from vectordb.base import VectorDBProvider
from vectordb.types import DocumentChunk, SearchResult, VectorDBConfig, Vector


class InMemoryProvider(VectorDBProvider):
    """
    Simple in-process vector database for demos/tests.

    This avoids external dependencies like Docker/Qdrant while exercising:
    - upsert
    - similarity search
    - metadata filtering (basic exact match)
    """

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Any]] = {}
        self._connected: bool = False

    def connect(self, config: VectorDBConfig) -> None:
        # No-op: data lives in memory.
        self._connected = True

    def disconnect(self) -> None:
        self._collections.clear()
        self._connected = False

    def create_collection(self, name: str, dimension: int, distance_metric: str) -> None:
        if not self._connected:
            # Keep behavior similar to real providers: require connect().
            raise RuntimeError("InMemoryProvider not connected. Call connect() first.")
        if dimension <= 0:
            raise ValueError("dimension must be > 0")

        self._collections[name] = {
            "dimension": int(dimension),
            "distance_metric": distance_metric,
            "chunks": {},  # id -> DocumentChunk
        }

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def upsert_documents(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        collection = self._require_collection(collection_name)
        chunks_by_id: dict[str, DocumentChunk] = collection["chunks"]

        for chunk in chunks:
            # Skip empty embeddings; embedding should exist after the pipeline.
            if not chunk.embedding:
                continue
            chunks_by_id[chunk.id] = chunk

    def search(
        self,
        collection_name: str,
        query_vector: Vector,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        collection = self._require_collection(collection_name)
        stored_chunks: dict[str, DocumentChunk] = collection["chunks"]

        metric = str(collection["distance_metric"]).lower().strip()
        results: list[SearchResult] = []

        for chunk in stored_chunks.values():
            if not self._passes_filters(chunk, filters):
                continue

            score = self._score(query_vector=query_vector, doc_vector=chunk.embedding, metric=metric)
            results.append(SearchResult(chunk=chunk, score=score, distance=None))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def delete_documents(self, collection_name: str, ids: list[str]) -> None:
        collection = self._require_collection(collection_name)
        chunks_by_id: dict[str, DocumentChunk] = collection["chunks"]
        for chunk_id in ids:
            chunks_by_id.pop(chunk_id, None)

    def health_check(self) -> bool:
        return self._connected

    def _require_collection(self, name: str) -> dict[str, Any]:
        if name not in self._collections:
            raise RuntimeError(f"Collection '{name}' not found. Call create_collection() first.")
        return self._collections[name]

    @staticmethod
    def _passes_filters(chunk: DocumentChunk, filters: dict[str, object] | None) -> bool:
        if not filters:
            return True
        metadata = chunk.metadata or {}
        for key, expected in filters.items():
            if key not in metadata:
                return False
            # Exact match, but be tolerant to type differences.
            if str(metadata.get(key)) != str(expected):
                return False
        return True

    @staticmethod
    def _cosine_similarity(a: Vector, b: Vector) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += float(x) * float(y)
            norm_a += float(x) * float(x)
            norm_b += float(y) * float(y)
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    @staticmethod
    def _euclidean_distance(a: Vector, b: Vector) -> float:
        if not a or not b:
            return float("inf")
        s = 0.0
        for x, y in zip(a, b):
            dx = float(x) - float(y)
            s += dx * dx
        return math.sqrt(s)

    def _score(self, query_vector: Vector, doc_vector: Vector, metric: str) -> float:
        """
        Return a higher-is-better score (like Qdrant's relevance score).
        """
        if metric == "cosine":
            return self._cosine_similarity(query_vector, doc_vector)

        if metric in ("euclidean", "l2"):
            # Convert distance to similarity-like score.
            # Lower distance => higher score.
            dist = self._euclidean_distance(query_vector, doc_vector)
            return 1.0 / (1.0 + dist)

        # Fallback: cosine.
        return self._cosine_similarity(query_vector, doc_vector)

