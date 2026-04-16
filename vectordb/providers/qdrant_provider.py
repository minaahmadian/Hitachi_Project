from __future__ import annotations

import logging
import time
import importlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ..base import VectorDBError, VectorDBProvider
from ..types import DocumentChunk, SearchResult, Vector, VectorDBConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class QdrantConnectionError(VectorDBError):
    pass


class QdrantOperationError(VectorDBError):
    pass


class QdrantProvider(VectorDBProvider):
    def __init__(self) -> None:
        self._client: Any | None = None
        self._models: Any | None = None
        self._response_handling_exception: Any | None = None
        self._unexpected_response_exception: Any | None = None
        self._retry_attempts: int = 3
        self._retry_backoff_seconds: float = 0.5
        self._batch_size: int = 100

    def connect(self, config: VectorDBConfig) -> None:
        try:
            qdrant_client_module = importlib.import_module("qdrant_client")
            qdrant_http_module = importlib.import_module("qdrant_client.http")
            QdrantClient = getattr(qdrant_client_module, "QdrantClient")
            models = getattr(qdrant_http_module, "models")
        except ImportError as exc:
            raise QdrantConnectionError(
                "qdrant-client is not installed. Install it with `pip install qdrant-client`."
            ) from exc
        except Exception as exc:
            raise QdrantConnectionError(f"Failed to import qdrant-client: {exc}") from exc

        try:
            qdrant_exceptions = importlib.import_module("qdrant_client.http.exceptions")
            self._response_handling_exception = getattr(qdrant_exceptions, "ResponseHandlingException")
            self._unexpected_response_exception = getattr(qdrant_exceptions, "UnexpectedResponse")
        except Exception:
            self._response_handling_exception = None
            self._unexpected_response_exception = None

        timeout = config.timeout_seconds
        local_path = cast(str | None, config.options.get("path") if config.options else None)

        try:
            client: Any
            if local_path:
                logger.info("Connecting to local Qdrant storage at path=%s", local_path)
                client = QdrantClient(path=local_path, timeout=timeout)
            elif config.url:
                logger.info("Connecting to remote Qdrant via url=%s", config.url)
                client = QdrantClient(url=config.url, api_key=config.api_key, timeout=timeout)
            elif config.host:
                logger.info("Connecting to remote Qdrant via host=%s port=%s", config.host, config.port or 6333)
                client = QdrantClient(
                    host=config.host,
                    port=config.port or 6333,
                    api_key=config.api_key,
                    timeout=timeout,
                )
            else:
                raise QdrantConnectionError(
                    "Invalid Qdrant configuration: provide options['path'] for local mode, or url/host for remote mode."
                )

            self._client = client
            self._models = models
            client.get_collections()
            logger.info("Qdrant connection established successfully")
        except VectorDBError:
            raise
        except Exception as exc:
            self._client = None
            self._models = None
            logger.exception("Failed to connect to Qdrant")
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {exc}") from exc

    def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
            logger.info("Disconnected from Qdrant")
        except Exception as exc:
            logger.exception("Failed while disconnecting from Qdrant")
            raise QdrantOperationError(f"Failed to disconnect from Qdrant: {exc}") from exc
        finally:
            self._client = None
            self._models = None

    def create_collection(self, name: str, dimension: int, distance_metric: str) -> None:
        client = self._require_client()
        models = self._require_models()
        distance = self._map_distance(distance_metric)
        logger.info(
            "Creating/recreating Qdrant collection name=%s dimension=%s metric=%s",
            name,
            dimension,
            distance_metric,
        )

        def _create() -> None:
            client.recreate_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dimension, distance=distance),
            )

        self._with_retry(_create, operation=f"create collection '{name}'")

    def delete_collection(self, name: str) -> None:
        client = self._require_client()
        logger.info("Deleting Qdrant collection name=%s", name)
        self._with_retry(
            lambda: client.delete_collection(collection_name=name),
            operation=f"delete collection '{name}'",
        )

    def upsert_documents(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        client = self._require_client()
        if not chunks:
            logger.debug("No chunks provided for upsert on collection=%s", collection_name)
            return

        logger.info(
            "Upserting %s chunks to collection=%s with batch_size=%s",
            len(chunks),
            collection_name,
            self._batch_size,
        )

        for index in range(0, len(chunks), self._batch_size):
            batch = chunks[index : index + self._batch_size]
            points = [self._to_point_struct(chunk) for chunk in batch]
            self._with_retry(
                lambda points=points: client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                ),
                operation=f"upsert batch {index // self._batch_size + 1}",
            )

    def search(
        self,
        collection_name: str,
        query_vector: Vector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        client = self._require_client()
        query_filter = self._build_filter(filters)

        def _search() -> list[Any]:
            # Compatibility across qdrant-client versions:
            # - older APIs expose `search(...)`
            # - newer APIs expose `query_points(...)` and return QueryResponse(points=[...])
            search_fn = getattr(client, "search", None)
            if callable(search_fn):
                return search_fn(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True,
                )

            query_points_fn = getattr(client, "query_points", None)
            if callable(query_points_fn):
                response = query_points_fn(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True,
                )
                points = getattr(response, "points", None)
                if isinstance(points, list):
                    return points
                raise QdrantOperationError("query_points returned unexpected response format (missing points)")

            raise QdrantOperationError("Qdrant client does not support search() or query_points()")

        points = self._with_retry(_search, operation=f"search collection '{collection_name}'")
        return [self._from_scored_point(point) for point in points]

    def delete_documents(self, collection_name: str, ids: list[str]) -> None:
        client = self._require_client()
        models = self._require_models()
        if not ids:
            logger.debug("No ids provided for delete on collection=%s", collection_name)
            return

        logger.info("Deleting %s documents from collection=%s", len(ids), collection_name)
        self._with_retry(
            lambda: client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            ),
            operation=f"delete documents from '{collection_name}'",
        )

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant health check failed")
            return False

    def _to_point_struct(self, chunk: DocumentChunk) -> Any:
        models = self._require_models()
        payload = {"text": chunk.text, **(chunk.metadata or {})}
        return models.PointStruct(id=chunk.id, vector=chunk.embedding, payload=payload)

    def _from_scored_point(self, point: Any) -> SearchResult:
        payload = dict(getattr(point, "payload", {}) or {})
        text = str(payload.pop("text", ""))
        point_id = str(getattr(point, "id", ""))
        embedding = cast(Vector, getattr(point, "vector", []) or [])
        score = float(getattr(point, "score", 0.0))
        chunk = DocumentChunk(id=point_id, text=text, metadata=payload, embedding=embedding)
        return SearchResult(chunk=chunk, score=score, distance=None)

    def _require_client(self) -> Any:
        if self._client is None:
            raise QdrantConnectionError("Qdrant client is not connected. Call connect() first.")
        return self._client

    def _require_models(self) -> Any:
        if self._models is None:
            raise QdrantConnectionError("Qdrant models are unavailable. Ensure connect() succeeded.")
        return self._models

    def _map_distance(self, metric: str) -> Any:
        models = self._require_models()
        normalized = metric.lower().strip()
        if normalized == "cosine":
            return models.Distance.COSINE
        if normalized == "euclidean":
            return models.Distance.EUCLID
        if normalized == "dot":
            return models.Distance.DOT
        raise QdrantOperationError(
            f"Unsupported distance metric '{metric}'. Supported metrics: cosine, euclidean, dot."
        )

    def _build_filter(self, filters: dict[str, Any] | None) -> Any | None:
        if not filters:
            return None
        models = self._require_models()
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
        return models.Filter(must=conditions)

    def _with_retry(self, fn: Callable[[], _T], operation: str) -> _T:
        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self._retry_attempts or not self._is_transient_error(exc):
                    logger.exception(
                        "Qdrant operation failed: %s (attempt=%s/%s)",
                        operation,
                        attempt,
                        self._retry_attempts,
                    )
                    raise QdrantOperationError(f"Failed to {operation}: {exc}") from exc

                sleep_for = self._retry_backoff_seconds * attempt
                logger.warning(
                    "Transient Qdrant error during %s (attempt=%s/%s): %s. Retrying in %.2fs",
                    operation,
                    attempt,
                    self._retry_attempts,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

        raise QdrantOperationError(f"Failed to {operation}: {last_error}") from last_error

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if self._response_handling_exception and isinstance(exc, self._response_handling_exception):
            return True
        if self._unexpected_response_exception and isinstance(exc, self._unexpected_response_exception):
            return True

        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "timeout",
                "temporarily unavailable",
                "connection reset",
                "connection aborted",
                "service unavailable",
                "too many requests",
                "gateway timeout",
            )
        )