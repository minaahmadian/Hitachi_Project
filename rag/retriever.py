from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Callable, Protocol

from vectordb.base import VectorDBProvider
from vectordb.types import SearchResult, Vector


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> Vector:
        ...


class GenericEmbedder(Protocol):
    def embed(self, text: str) -> Vector:
        ...


@dataclass(slots=True)
class RAGRetriever:
    provider: VectorDBProvider
    embedder: QueryEmbedder | GenericEmbedder
    collection_name: str
    hybrid_search: bool = True
    semantic_weight: float = 0.8

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or top_k <= 0:
            return []

        query_vector = self._embed_query(query)

        candidate_top_k = top_k * 3 if self.hybrid_search else top_k
        provider_filters, post_filters = self._split_filters(filters)
        semantic_results = self.provider.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=candidate_top_k,
            filters=provider_filters,
        )

        if post_filters:
            semantic_results = [
                result
                for result in semantic_results
                if self._passes_post_filters(result, post_filters)
            ]

        if not semantic_results:
            return []

        if self.hybrid_search:
            semantic_results = self._hybrid_rerank(query, semantic_results)

        return semantic_results[:top_k]

    def retrieve_with_scores(
        self,
        query: str,
        min_score: float = 0.7,
    ) -> list[SearchResult]:
        results = self.retrieve(query=query, top_k=50)
        return [result for result in results if result.score >= min_score]

    def _embed_query(self, query: str) -> Vector:
        embed_query_fn = getattr(self.embedder, "embed_query", None)
        embed_fn = getattr(self.embedder, "embed", None)

        if callable(embed_query_fn):
            vector = self._call_embedder(embed_query_fn, query)
        elif callable(embed_fn):
            vector = self._call_embedder(embed_fn, query)
        else:
            raise TypeError("Embedder must implement embed_query(text) or embed(text)")

        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedder returned an invalid empty query vector")

        return vector

    @staticmethod
    def _call_embedder(func: Callable[[str], object], query: str) -> Vector:
        vector = func(query)
        if not isinstance(vector, list):
            raise ValueError("Embedder must return a list[float]")
        normalized: list[float] = []
        for value in vector:
            if not isinstance(value, (int, float)):
                raise ValueError("Embedder returned a non-numeric vector value")
            normalized.append(float(value))
        return normalized

    def _hybrid_rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return sorted(results, key=lambda item: item.score, reverse=True)

        semantic_weight = max(0.0, min(1.0, self.semantic_weight))
        keyword_weight = 1.0 - semantic_weight

        reranked: list[SearchResult] = []
        for result in results:
            doc_terms = self._tokenize(result.chunk.text)
            keyword_score = self._keyword_overlap(query_terms, doc_terms)
            hybrid_score = (semantic_weight * result.score) + (keyword_weight * keyword_score)
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=hybrid_score,
                    distance=result.distance,
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _split_filters(
        self,
        filters: dict[str, object] | None,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        if not filters:
            return None, {}

        provider_filters = dict(filters)
        post_filters: dict[str, object] = {}

        for key in ("date_range", "source_type", "tags"):
            if key in provider_filters:
                post_filters[key] = provider_filters.pop(key)

        return provider_filters or None, post_filters

    def _passes_post_filters(self, result: SearchResult, filters: dict[str, object]) -> bool:
        metadata = result.chunk.metadata or {}

        date_range = filters.get("date_range")
        date_range_dict = self._as_object_dict(date_range)
        if date_range_dict is not None and not self._passes_date_range(metadata, date_range_dict):
            return False

        source_type = filters.get("source_type")
        if source_type and metadata.get("source_type") != source_type:
            return False

        tags = filters.get("tags")
        if tags:
            result_tags_raw = metadata.get("tags")
            result_tags = self._as_string_set(result_tags_raw)
            requested_tags = self._as_string_set(tags)
            if not requested_tags.intersection(result_tags):
                return False

        return True

    def _passes_date_range(self, metadata: dict[str, object], date_range: dict[str, object]) -> bool:
        raw_date = (
            metadata.get("date")
            or metadata.get("created_at")
            or metadata.get("timestamp")
            or metadata.get("datetime")
        )
        if not raw_date:
            return False

        current = self._parse_datetime(raw_date)
        if current is None:
            return False

        start = self._parse_datetime(date_range.get("start")) if date_range.get("start") else None
        end = self._parse_datetime(date_range.get("end")) if date_range.get("end") else None

        if start and current < start:
            return False
        if end and current > end:
            return False
        return True

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None

        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _keyword_overlap(query_terms: set[str], doc_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        return len(query_terms.intersection(doc_terms)) / len(query_terms)

    @staticmethod
    def _as_string_set(value: object) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, (list, tuple, set, frozenset)):
            return {str(item) for item in value}
        return set()

    @staticmethod
    def _as_object_dict(value: object) -> dict[str, object] | None:
        if not isinstance(value, dict):
            return None
        output: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(key, str):
                output[key] = item
        return output