from __future__ import annotations

from enum import Enum
import re
from typing import Final

from documents import Document
from vectordb import DocumentChunk


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class TextChunker:
    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        split_by_paragraph: bool = True,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size: Final[int] = chunk_size
        self.chunk_overlap: Final[int] = chunk_overlap
        self.split_by_paragraph: Final[bool] = split_by_paragraph
        self.strategy: Final[ChunkingStrategy] = strategy

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.content or ""
        if not text.strip():
            return []

        spans = self._split_spans(text)
        total_chunks = len(spans)
        chunks: list[DocumentChunk] = []

        for idx, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            metadata: dict[str, object] = {
                "document_id": document.id,
                "parent_document_id": document.id,
                "source_id": document.metadata.source_id,
                "source_type": str(document.metadata.source_type.value),
                "title": document.metadata.title,
                "author": document.metadata.author,
                "created_at": document.metadata.created_at.isoformat() if document.metadata.created_at else None,
                "processed_at": document.processed_at.isoformat(),
                "tags": list(document.metadata.tags),
                "chunk_index": idx,
                "chunk_number": idx + 1,
                "total_chunks": total_chunks,
                "chunk_position": f"{idx + 1} of {total_chunks}",
                "chunk_start": start,
                "chunk_end": end,
            }
            if document.metadata.extra_fields:
                metadata["source_extra_fields"] = dict(document.metadata.extra_fields)

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}:{idx}",
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_spans(self, text: str) -> list[tuple[int, int]]:
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._split_fixed(text)
        return self._split_with_boundaries(text)

    def _split_fixed(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            spans.append((start, end))
            if end >= text_len:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return spans

    def _split_with_boundaries(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            max_end = min(start + self.chunk_size, text_len)
            end = max_end

            if max_end < text_len:
                boundary = self._find_best_boundary(text, start, max_end)
                if boundary is not None and boundary > start:
                    end = boundary

            spans.append((start, end))
            if end >= text_len:
                break

            next_start = max(end - self.chunk_overlap, start + 1)
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return spans

    def _find_best_boundary(self, text: str, start: int, max_end: int) -> int | None:
        lower = start + max(1, int(self.chunk_size * 0.6))
        if lower >= max_end:
            lower = start

        if self.split_by_paragraph:
            paragraph_pos = text.rfind("\n\n", lower, max_end)
            if paragraph_pos != -1:
                return paragraph_pos

        sentence_boundary = None
        for match in re.finditer(r"[.!?]\s+", text[lower:max_end]):
            sentence_boundary = lower + match.end()
        if sentence_boundary is not None:
            return sentence_boundary

        word_pos = text.rfind(" ", lower, max_end)
        if word_pos != -1:
            return word_pos

        return None