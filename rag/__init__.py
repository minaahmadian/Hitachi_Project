from __future__ import annotations

from dataclasses import dataclass

from .context_builder import ContextBuilder, ContextStrategy
from .retriever import RAGRetriever


@dataclass(slots=True)
class RAGResult:
    query: str
    context: str
    sources: list[dict[str, object]]
    total_chunks_found: int


def query_rag(
    query: str,
    retriever: RAGRetriever,
    context_builder: ContextBuilder,
    top_k: int = 5,
    max_context_tokens: int = 2000,
) -> RAGResult:
    results = retriever.retrieve(query=query, top_k=top_k)
    if not results:
        return RAGResult(
            query=query,
            context="",
            sources=[],
            total_chunks_found=0,
        )

    context = context_builder.build_context(results, max_tokens=max_context_tokens)
    sources = [
        {
            "id": result.chunk.id,
            "filename": (result.chunk.metadata or {}).get("filename")
            or (result.chunk.metadata or {}).get("source")
            or "unknown",
            "chunk_index": (result.chunk.metadata or {}).get("chunk_index"),
            "total_chunks": (result.chunk.metadata or {}).get("total_chunks"),
            "score": result.score,
            "distance": result.distance,
            "metadata": dict(result.chunk.metadata or {}),
        }
        for result in results
    ]
    return RAGResult(
        query=query,
        context=context,
        sources=sources,
        total_chunks_found=len(results),
    )


__all__ = [
    "ContextBuilder",
    "ContextStrategy",
    "RAGResult",
    "RAGRetriever",
    "query_rag",
]