"""Shared types for vector database providers.

This module defines strongly-typed, provider-agnostic data structures used by
the vector database abstraction layer.
"""

from dataclasses import dataclass, field
Vector = list[float]  # Embedding vector type alias


@dataclass(slots=True)
class DocumentChunk:
    """A document chunk and its associated vector representation.

    Args:
        id: Unique chunk identifier within a collection.
        text: Raw text content for the chunk.
        metadata: Arbitrary key-value metadata for filtering and traceability.
        embedding: Dense vector embedding for semantic search.
    """

    id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    embedding: Vector = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    """A ranked result returned by a vector similarity search.

    Args:
        chunk: The matched document chunk.
        score: Provider-specific relevance score (higher is typically better).
        distance: Raw distance metric value from the vector index, if available.
    """

    chunk: DocumentChunk
    score: float
    distance: float | None = None


@dataclass(slots=True)
class VectorDBConfig:
    """Configuration used to connect to a vector database provider.

    Args:
        provider: Provider identifier (for example: "qdrant", "pinecone").
        host: Hostname or IP address of the vector database endpoint.
        port: Network port used by the endpoint.
        url: Full endpoint URL when host/port is not used.
        api_key: Optional API key or token for authenticated providers.
        timeout_seconds: Request timeout in seconds.
        options: Provider-specific extension options.
    """

    provider: str
    host: str | None = None
    port: int | None = None
    url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0
    options: dict[str, object] = field(default_factory=dict)