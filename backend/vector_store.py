"""High-level vector store for document storage and retrieval.

This module provides a unified interface for:
1. Processing documents (chunking + embedding)
2. Storing documents in vector database
3. Retrieving relevant documents via RAG
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from documents import Document
from documents.models import EmailDocument
from documents.parsers import ParserFactory
from processing.pipeline import ProcessingPipeline
from processing.chunker import TextChunker, ChunkingStrategy
from processing.embedder import GroqEmbedder
from rag.retriever import RAGRetriever
from rag.context_builder import ContextBuilder, ContextStrategy
from rag import query_rag, RAGResult
from vectordb import VectorDBConfig, DocumentChunk, SearchResult
from vectordb.providers.qdrant_provider import QdrantProvider

if TYPE_CHECKING:
    from vectordb.base import VectorDBProvider
    from processing.embedder import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for VectorStore.

    Args:
        provider_type: VectorDB provider type ("qdrant", etc.)
        host: Database host
        port: Database port
        collection_name: Default collection for documents
        embedding_model: OpenAI embedding model name
        chunk_size: Text chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        api_key: API key for embedding service
    """

    provider_type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 500
    chunk_overlap: int = 50
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> VectorStoreConfig:
        return cls(
            provider_type=os.getenv("VECTORDB_PROVIDER", "qdrant"),
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            api_key=os.getenv("GROQ_API_KEY"),
        )


class VectorStore:
    """Unified interface for document storage and RAG retrieval.

    This class orchestrates the full pipeline:
    1. Parse documents (files, emails, etc.)
    2. Chunk and embed documents
    3. Store in vector database
    4. Retrieve via semantic search
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        provider: VectorDBProvider | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.config: Final[VectorStoreConfig] = config or VectorStoreConfig.from_env()
        self._provider: VectorDBProvider | None = provider
        self._embedder: EmbeddingProvider | None = embedder
        self._pipeline: ProcessingPipeline | None = None
        self._retriever: RAGRetriever | None = None
        self._context_builder: ContextBuilder | None = None

    def connect(self) -> None:
        """Initialize connections to vector database."""
        logger.info("Connecting to vector store (provider=%s)", self.config.provider_type)

        if self._provider is None:
            if self.config.provider_type == "qdrant":
                self._provider = QdrantProvider()
                db_config = VectorDBConfig(
                    provider="qdrant",
                    host=self.config.host,
                    port=self.config.port,
                )
                self._provider.connect(db_config)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider_type}")

        if self._embedder is None:
            self._embedder = GroqEmbedder(model=self.config.embedding_model, api_key=self.config.api_key)

        chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        self._pipeline = ProcessingPipeline(
            chunker=chunker,
            embedder=self._embedder,
        )

        self._retriever = RAGRetriever(
            provider=self._provider,
            embedder=self._embedder,
            collection_name=self.config.collection_name,
            hybrid_search=True,
        )

        self._context_builder = ContextBuilder(strategy=ContextStrategy.HIGHEST_SCORE)

        logger.info("Vector store connected successfully")

    def disconnect(self) -> None:
        """Close connections to vector database."""
        if self._provider:
            self._provider.disconnect()
            self._provider = None
        logger.info("Vector store disconnected")

    def setup_collection(self, dimension: int = 768, distance_metric: str = "cosine") -> None:
        """Create or recreate the document collection.

        Args:
            dimension: Vector dimension (768 for nomic-embed-text)
            distance_metric: Distance metric for similarity search
        """
        if not self._provider:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info(
            "Setting up collection %s (dimension=%s, metric=%s)",
            self.config.collection_name,
            dimension,
            distance_metric,
        )
        self._provider.create_collection(
            name=self.config.collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
        )

    def add_document(self, document: Document) -> list[DocumentChunk]:
        """Process and store a document.

        Args:
            document: Document to store

        Returns:
            List of stored chunks
        """
        if not self._pipeline or not self._provider:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info("Adding document: %s (source=%s)", document.id, document.metadata.source_type)

        chunks = self._pipeline.process_document(document)
        if not chunks:
            logger.warning("No chunks generated for document %s", document.id)
            return []

        self._provider.upsert_documents(self.config.collection_name, chunks)
        logger.info("Stored %s chunks for document %s", len(chunks), document.id)

        return chunks

    def add_file(self, filepath: str, metadata: dict[str, object] | None = None) -> list[DocumentChunk]:
        """Parse and store a file.

        Args:
            filepath: Path to file
            metadata: Optional additional metadata

        Returns:
            List of stored chunks
        """
        parser = ParserFactory.get_parser(filepath)
        document = parser.parse(filepath)

        if metadata:
            document.metadata.extra_fields.update(metadata)

        return self.add_document(document)

    def add_email(
        self,
        body: str,
        subject: str | None = None,
        from_addr: str | None = None,
        to_addr: list[str] | None = None,
        source_id: str | None = None,
        **kwargs: object,
    ) -> list[DocumentChunk]:
        """Store an email document.

        Args:
            body: Email body text
            subject: Email subject
            from_addr: Sender address
            to_addr: Recipient addresses
            source_id: Unique identifier for this email
            **kwargs: Additional metadata fields

        Returns:
            List of stored chunks
        """
        from documents.models import EmailDocument
        from datetime import datetime, timezone

        document = EmailDocument.from_email_fields(
            body=body,
            source_id=source_id or f"email:{datetime.now(timezone.utc).isoformat()}",
            subject=subject,
            from_addr=from_addr,
            to_addr=to_addr or [],
            extra_fields=kwargs,
            tags=["email"],
        )

        return self.add_document(document)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        """Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._retriever:
            raise RuntimeError("Not connected. Call connect() first.")

        return self._retriever.retrieve(query=query, top_k=top_k, filters=filters)

    def query(self, query: str, top_k: int = 5, max_context_tokens: int = 2000) -> RAGResult:
        """Full RAG query with context building.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            max_context_tokens: Maximum tokens in context

        Returns:
            RAG result with query, context, and sources
        """
        if not self._retriever or not self._context_builder:
            raise RuntimeError("Not connected. Call connect() first.")

        return query_rag(
            query=query,
            retriever=self._retriever,
            context_builder=self._context_builder,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
        )

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Note: This deletes chunks by prefix matching on document_id.

        Args:
            document_id: Document ID to delete
        """
        if not self._provider:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info("Deleting document: %s", document_id)
        logger.warning("Delete by document ID not fully implemented - use delete_chunks with specific IDs")

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not self._provider:
            raise RuntimeError("Not connected. Call connect() first.")

        self._provider.delete_documents(self.config.collection_name, chunk_ids)
        logger.info("Deleted %s chunks", len(chunk_ids))

    def health_check(self) -> bool:
        """Check if vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self._provider:
            return False
        return self._provider.health_check()