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
    requirement_id: str | None
    index_kind: str | None


class RSSOMRAGIndex:
    """
    Lightweight vector retrieval over the RSSOM/FIT corpus.

    This gives Phase 2 a real semantic retrieval path over the parsed FIT evidence,
    instead of relying only on exact line scanning.
    """

    def __init__(
        self,
        corpus_text: str,
        *,
        title: str = "RSSOM FIT corpus",
        requirement_rows: list[dict[str, str]] | None = None,
    ) -> None:
        self._provider = InMemoryProvider()
        self._retriever: RAGRetriever | None = None
        self._ready = False
        self._title = title

        raw = str(corpus_text or "").strip()
        if not raw:
            return

        embedder = make_embedder_from_env()
        chunker = TextChunker(
            chunk_size=int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_SIZE", "1400") or "1400"),
            chunk_overlap=int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_OVERLAP", "120") or "120"),
            strategy=ChunkingStrategy.RECURSIVE,
        )
        pipeline = ProcessingPipeline(chunker=chunker, embedder=embedder)
        docs = _build_requirement_documents(raw, requirement_rows=requirement_rows, title=title)
        if not docs:
            docs = [
                Document(
                    content=raw,
                    metadata=DocumentMetadata(
                        source_type=DocumentSource.MANUAL,
                        source_id="rssom_test_evidence_corpus",
                        title=title,
                        tags=["rssom", "fit", "traceability", "rag", "raw_corpus"],
                        extra_fields={"index_kind": "raw_corpus"},
                    ),
                )
            ]
        chunks = pipeline.process_documents(docs)
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
    exact_meta_hit = any((h.requirement_id or "").upper() == req_id.upper() for h in hits if req_id)
    title_overlap = any(_title_overlap(title_s, text) > 0 for text in texts if title_s)
    relevant = exact_id_hit or exact_meta_hit or title_overlap
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
        requirement_id=str(meta.get("requirement_id", "")).strip() or None,
        index_kind=str(meta.get("index_kind", "")).strip() or None,
    )


def _hit_dict(hit: RSSOMRAGHit) -> dict[str, object]:
    return {
        "text": hit.text,
        "score": hit.score,
        "chunk_index": hit.chunk_index,
        "total_chunks": hit.total_chunks,
        "requirement_id": hit.requirement_id,
        "index_kind": hit.index_kind,
    }


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_requirement_documents(
    corpus_text: str,
    *,
    requirement_rows: list[dict[str, str]] | None,
    title: str,
) -> list[Document]:
    rows = requirement_rows or []
    if not rows:
        return []
    corpus = str(corpus_text or "")
    corpus_l = corpus.lower()
    docs: list[Document] = []
    seen: set[str] = set()

    for row in rows:
        rid = str(row.get("requirement_id", "")).strip()
        req_title = str(row.get("title", "")).strip()
        if not rid or rid.upper() in seen:
            continue
        # Keep only IDs that truly exist in RSSOM evidence; this avoids indexing eval-only synthetic rows.
        if rid.lower() not in corpus_l:
            continue
        seen.add(rid.upper())
        excerpts = _evidence_excerpts_for_requirement(corpus, rid, req_title)
        body = [
            f"Requirement ID: {rid}",
            f"Requirement title: {req_title}",
        ]
        if excerpts:
            body.append("RSSOM evidence excerpts:")
            body.extend(f"- {line}" for line in excerpts)
        content = "\n".join(part for part in body if part).strip()
        docs.append(
            Document(
                content=content,
                metadata=DocumentMetadata(
                    source_type=DocumentSource.MANUAL,
                    source_id=f"rssom_requirement::{rid}",
                    title=f"{rid} requirement entry",
                    tags=["rssom", "fit", "traceability", "rag", "requirement_entry"],
                    extra_fields={
                        "index_kind": "requirement_entry",
                        "requirement_id": rid,
                        "requirement_title": req_title,
                        "corpus_title": title,
                    },
                ),
            )
        )
    return docs


def _evidence_excerpts_for_requirement(corpus: str, requirement_id: str, title: str, *, max_lines: int = 6) -> list[str]:
    lines: list[str] = []
    rid_l = requirement_id.lower()
    title_tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in {"and", "with", "the", "for", "level", "check", "through"}
    ][:8]
    for raw in corpus.splitlines():
        line = raw.strip()
        if not line:
            continue
        ll = line.lower()
        if rid_l in ll:
            lines.append(line[:280])
            if len(lines) >= max_lines:
                break
            continue
        overlap = sum(1 for tok in title_tokens if tok in ll)
        if overlap >= 2:
            lines.append(line[:280])
            if len(lines) >= max_lines:
                break
    return lines
