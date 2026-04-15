"""Vector database abstraction package."""

from vectordb.base import VectorDBError, VectorDBProvider
from vectordb.types import DocumentChunk, SearchResult, Vector, VectorDBConfig

__all__ = [
    "DocumentChunk",
    "SearchResult",
    "Vector",
    "VectorDBConfig",
    "VectorDBError",
    "VectorDBProvider",
]