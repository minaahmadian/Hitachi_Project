from __future__ import annotations

import logging
from typing import Final

from documents import Document
from .chunker import TextChunker
from .embedder import EmbeddingProvider
from vectordb import DocumentChunk


class ProcessingPipeline:
    def __init__(
        self,
        *,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
        logger: logging.Logger | None = None,
    ) -> None:
        self.chunker: Final[TextChunker] = chunker
        self.embedder: Final[EmbeddingProvider] = embedder
        self.logger: Final[logging.Logger] = logger or logging.getLogger(__name__)

    def process_document(self, document: Document) -> list[DocumentChunk]:
        try:
            chunks = self.chunker.chunk(document)
        except Exception as exc:
            self.logger.exception("Failed to chunk document %s: %s", document.id, exc)
            return []

        if not chunks:
            return []

        return self._embed_chunks(document_id=document.id, chunks=chunks)

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        total = len(documents)
        for idx, document in enumerate(documents, start=1):
            self.logger.info("Processing document %s/%s (id=%s)", idx, total, document.id)
            processed = self.process_document(document)
            all_chunks.extend(processed)
            self.logger.info(
                "Processed document %s/%s (id=%s), generated %s chunks",
                idx,
                total,
                document.id,
                len(processed),
            )
        return all_chunks

    def _embed_chunks(self, *, document_id: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        batch_size = self.embedder.batch_size
        embedded_chunks: list[DocumentChunk] = []

        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            texts = [chunk.text for chunk in batch]
            try:
                vectors = self.embedder.embed(texts)
            except Exception as exc:
                self.logger.exception(
                    "Embedding failed for document %s batch [%s:%s]: %s",
                    document_id,
                    offset,
                    offset + len(batch),
                    exc,
                )
                continue

            if len(vectors) != len(batch):
                self.logger.error(
                    "Embedding count mismatch for document %s batch [%s:%s]: expected=%s got=%s",
                    document_id,
                    offset,
                    offset + len(batch),
                    len(batch),
                    len(vectors),
                )
                continue

            for chunk, vector in zip(batch, vectors):
                chunk.embedding = vector
                embedded_chunks.append(chunk)

        return embedded_chunks