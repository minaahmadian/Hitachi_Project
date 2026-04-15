"""Abstract base classes for pluggable vector database providers."""

from abc import ABC, abstractmethod
from vectordb.types import DocumentChunk, SearchResult, Vector, VectorDBConfig


class VectorDBError(Exception):
    """Base exception for vector database provider errors."""


class VectorDBProvider(ABC):
    """Provider interface for vector database operations.

    Concrete providers should implement this interface to support a specific
    backend (for example, Qdrant, Pinecone, or Weaviate) while keeping the rest
    of the application provider-agnostic.
    """

    @abstractmethod
    def connect(self, config: VectorDBConfig) -> None:
        """Initialize a connection to the vector database.

        Args:
            config: Connection and provider configuration settings.

        Returns:
            None

        Raises:
            VectorDBError: If connection initialization fails.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up provider resources and close active connections.

        Returns:
            None

        Raises:
            VectorDBError: If resource cleanup fails.
        """

    @abstractmethod
    def create_collection(self, name: str, dimension: int, distance_metric: str) -> None:
        """Create a collection (or index) for vector storage.

        Args:
            name: Collection name.
            dimension: Embedding dimensionality for vectors in the collection.
            distance_metric: Distance metric identifier (for example, "cosine").

        Returns:
            None

        Raises:
            VectorDBError: If collection creation fails.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete an existing collection.

        Args:
            name: Collection name.

        Returns:
            None

        Raises:
            VectorDBError: If collection deletion fails.
        """

    @abstractmethod
    def upsert_documents(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks in a collection.

        Args:
            collection_name: Target collection name.
            chunks: Document chunks to upsert.

        Returns:
            None

        Raises:
            VectorDBError: If upsert fails.
        """

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: Vector,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest document chunks in a collection.

        Args:
            collection_name: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters applied by the provider.

        Returns:
            A ranked list of search results.

        Raises:
            VectorDBError: If search execution fails.
        """

    @abstractmethod
    def delete_documents(self, collection_name: str, ids: list[str]) -> None:
        """Delete document chunks from a collection by identifier.

        Args:
            collection_name: Target collection name.
            ids: Chunk identifiers to remove.

        Returns:
            None

        Raises:
            VectorDBError: If deletion fails.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check provider availability and readiness.

        Returns:
            True if the provider is healthy, otherwise False.
        """