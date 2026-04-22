from __future__ import annotations

import os
import re
from dataclasses import dataclass

from documents import Document, DocumentMetadata, DocumentSource
from processing.chunker import ChunkingStrategy, TextChunker
from processing.embedder import make_embedder_from_env
from processing.pipeline import ProcessingPipeline
from rag.retriever import RAGRetriever
from vectordb import SearchResult, VectorDBConfig
from vectordb.providers.inmemory_provider import InMemoryProvider


@dataclass(slots=True)
class RSSOMRAGHit:
    text: str
    score: float
    chunk_index: int | None
    total_chunks: int | None


class RSSOMRAGIndex:
    """
    Lightweight vector retrieval over the RSSOM/FIT corpus.

    This gives Phase 2 a real semantic retrieval path over the parsed FIT evidence,
    instead of relying only on exact line scanning.
    """

    def __init__(self, corpus_text: str, *, title: str = "RSSOM FIT corpus") -> None:
        self._provider = InMemoryProvider()
        self._retriever: RAGRetriever | None = None
        self._ready = False
        self._title = title

        raw = str(corpus_text or "").strip()
        if not raw:
            return

        embedder = make_embedder_from_env()
        chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500") or "500"),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50") or "50"),
            strategy=ChunkingStrategy.RECURSIVE,
        )
        pipeline = ProcessingPipeline(chunker=chunker, embedder=embedder)
        doc = Document(
            content=raw,
            metadata=DocumentMetadata(
                source_type=DocumentSource.MANUAL,
                source_id="rssom_test_evidence_corpus",
                title=title,
                tags=["rssom", "fit", "traceability", "rag"],
            ),
        )
        chunks = pipeline.process_document(doc)
        if not chunks or not chunks[0].embedding:
            return

        self._provider.connect(VectorDBConfig(provider="inmemory"))
        self._provider.create_collection(
            name="rssom_fit_corpus",
            dimension=len(chunks[0].embedding),
            distance_metric="cosine",
        )
        self._provider.upsert_documents("rssom_fit_corpus", chunks)
        self._retriever = RAGRetriever(
            provider=self._provider,
            embedder=embedder,
            collection_name="rssom_fit_corpus",
            hybrid_search=True,
        )
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready and self._retriever is not None

    def retrieve(self, query: str, *, top_k: int = 3, max_chars: int = 320) -> list[RSSOMRAGHit]:
        if not self.ready or not query.strip():
            return []
        assert self._retriever is not None
        results = self._retriever.retrieve(query=query, top_k=max(1, top_k))
        return [_to_hit(item, max_chars=max_chars) for item in results]


def classify_requirement_with_rag(
    rag: RSSOMRAGIndex | None,
    *,
    requirement_id: str,
    title: str,
    top_k: int = 3,
) -> tuple[str, str | None, dict[str, object]]:
    """
    Retrieve RSSOM chunks for one requirement and derive a PASS/FAIL/UNKNOWN/NOT_FOUND verdict.

    Guards against random semantic hits by requiring either:
    - exact requirement ID in a retrieved chunk, or
    - some lexical overlap from the title.
    """
    req_id = str(requirement_id or "").strip()
    title_s = str(title or "").strip()
    query = " ".join(part for part in (req_id, title_s, "verification fit rssom test evidence") if part).strip()
    if not rag or not rag.ready or not query:
        return "NOT_FOUND", None, {"enabled": False, "query": query, "hits": []}

    hits = rag.retrieve(query, top_k=top_k)
    if not hits:
        return "NOT_FOUND", None, {"enabled": True, "query": query, "hits": []}

    texts = [h.text for h in hits if h.text]
    score_max = max((h.score for h in hits), default=0.0)
    exact_id_hit = any(req_id.lower() in text.lower() for text in texts if req_id)
    title_overlap = any(_title_overlap(title_s, text) > 0 for text in texts if title_s)
    relevant = exact_id_hit or title_overlap
    if not relevant:
        return "NOT_FOUND", None, {
            "enabled": True,
            "query": query,
            "hits": [_hit_dict(h) for h in hits],
            "top_score": round(score_max, 4),
            "relevance_gate": "failed",
        }

    joined = "\n".join(texts)
    line_fail = re.compile(r"\bfail(?:ed|ure)?\b|✗", re.IGNORECASE)
    line_pass = re.compile(r"\bpass(?:ed)?\b|✓|✔|\bok\b", re.IGNORECASE)
    snippet = texts[0][:300] if texts else None

    if line_fail.search(joined):
        verdict = "FAIL"
    elif line_pass.search(joined):
        verdict = "PASS"
    else:
        verdict = "UNKNOWN"
    return verdict, snippet, {
        "enabled": True,
        "query": query,
        "hits": [_hit_dict(h) for h in hits],
        "top_score": round(score_max, 4),
        "relevance_gate": "passed",
    }


def _title_overlap(title: str, text: str) -> int:
    tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in {"and", "with", "the", "for", "level", "check", "through"}
    }
    if not tokens:
        return 0
    text_l = str(text or "").lower()
    return sum(1 for tok in tokens if tok in text_l)


def _to_hit(item: SearchResult, *, max_chars: int) -> RSSOMRAGHit:
    meta = item.chunk.metadata or {}
    text = (item.chunk.text or "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return RSSOMRAGHit(
        text=text,
        score=round(float(item.score), 4),
        chunk_index=_as_int(meta.get("chunk_index")),
        total_chunks=_as_int(meta.get("total_chunks")),
    )


def _hit_dict(hit: RSSOMRAGHit) -> dict[str, object]:
    return {
        "text": hit.text,
        "score": hit.score,
        "chunk_index": hit.chunk_index,
        "total_chunks": hit.total_chunks,
    }


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
