"""
RSSOM / FIT semantic retrieval for traceability.

Reliability architecture (v3 — triple-path fusion)
--------------------------------------------------
The retrieval pipeline runs **three independent paths** over the same
requirement-centric corpus and fuses their rankings.  Each path has
complementary strengths so their consensus is much more reliable than any
single path alone:

    ┌────────────────────┐
    │  Query expansion   │  ← deterministic glossary (PVIS⇄passenger info …)
    └────────┬───────────┘
             ▼
    ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
    │  Dense vector      │  │  BM25+ sparse      │  │  Exact-ID match    │
    │  (semantic)        │  │  (lexical)         │  │  (deterministic)   │
    └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
             └────────┬─────────────┴──────────────────────┘
                      ▼  Reciprocal Rank Fusion (RRF)
             ┌────────────────────┐
             │  Evidence-aware    │  ← IDF-weighted title overlap boost
             │  reranker          │
             └────────┬───────────┘
                      ▼
             ┌────────────────────┐
             │  Proximity verdict │  ← PASS/FAIL tokens within ±320 chars of
             │  + consensus gate  │     the requirement_id, agreement metric
             └────────────────────┘

Other reliability features:
- **Persistent disk cache** (hash-keyed): no re-embedding across runs.
- **Richer requirement entries**: context windows + test-objective / expected
  result extraction.
- **Consensus metric** exposed on every classification for auditability.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from documents import Document, DocumentMetadata, DocumentSource
from processing.chunker import ChunkingStrategy, TextChunker
from processing.embedder import make_embedder_from_env
from processing.pipeline import ProcessingPipeline
from rag.retriever import RAGRetriever
from traceability.bm25_index import BM25Index, BM25Hit
from traceability.query_expansion import expand_query
from vectordb import SearchResult, VectorDBConfig
from vectordb.providers.inmemory_provider import InMemoryProvider
from vectordb.types import DocumentChunk

logger = logging.getLogger(__name__)

_VERDICT_WINDOW = 320  # chars around req_id for proximity verdict search
_LINE_FAIL = re.compile(r"\bfail(?:ed|ure)?\b|✗", re.IGNORECASE)
_LINE_PASS = re.compile(r"\bpass(?:ed)?\b|✓|✔|\bok\b", re.IGNORECASE)
_OBJ_RX = re.compile(r"test\s*objective|test\s*purpose|objective\s*:", re.IGNORECASE)
_EXP_RX = re.compile(r"expected\s*result|pass\s*criteria|acceptance\s*criteria", re.IGNORECASE)
_VER_RX = re.compile(r"\b(passed|failed|not\s+tested|n/?a|complete|verified)\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    "and", "with", "the", "for", "level", "check", "through",
    "its", "via", "that", "this", "from", "into", "test",
})

# RRF (Reciprocal Rank Fusion) tunables. k=60 is the canonical value from the
# original Cormack, Clarke & Buettcher (2009) paper; it is stable and robust
# across corpora and rarely needs tuning.
_RRF_K = 60


# ──────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RSSOMRAGHit:
    text: str
    score: float
    chunk_index: int | None
    total_chunks: int | None
    requirement_id: str | None
    index_kind: str | None
    sources: tuple[str, ...] = ()        # which retrieval paths returned this hit
    fused_score: float = 0.0             # RRF score before evidence rerank


# ──────────────────────────────────────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────────────────────────────────────

class RSSOMRAGIndex:
    """
    Requirement-centric triple-path retrieval index over the RSSOM/FIT corpus.

    Build flow:
      1. One ``Document`` per requirement (ID + title + context excerpts).
      2. Dense: chunk + embed via ``ProcessingPipeline`` → ``InMemoryProvider``.
      3. Sparse: same chunks indexed into a BM25+ sparse index.
      4. Persistent cache: dense chunks (embeddings) cached to disk.  BM25
         rebuilds in < 50ms from the cached chunks so it is never stale.
      5. ``retrieve()`` runs **dense + BM25 + exact-ID** in parallel, fuses
         with Reciprocal Rank Fusion, applies evidence-aware reranking.
    """

    def __init__(
        self,
        corpus_text: str,
        *,
        title: str = "RSSOM FIT corpus",
        requirement_rows: list[dict[str, str]] | None = None,
        cache_dir: "Path | str | None" = None,
    ) -> None:
        self._provider = InMemoryProvider()
        self._retriever: RAGRetriever | None = None
        self._bm25: BM25Index | None = None
        self._chunks: list[DocumentChunk] = []
        self._ready = False
        self._title = title
        self._chunk_size = int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_SIZE", "1400") or "1400")
        self._chunk_overlap = int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_OVERLAP", "120") or "120")

        raw = str(corpus_text or "").strip()
        if not raw:
            return

        rows = list(requirement_rows or [])
        cdir = _resolve_cache_dir(cache_dir)
        cache_key = _corpus_hash(raw, rows, self._chunk_size, self._chunk_overlap)
        cache_path = cdir / f"rssom_rag_{cache_key}.json"

        embedder = make_embedder_from_env()
        chunks = self._load_or_build(raw, rows, embedder, cache_path)
        if not chunks or not chunks[0].embedding:
            return

        self._chunks = chunks
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

        self._bm25 = _build_bm25_from_chunks(chunks)
        self._ready = True

    # ── cache-aware build ──────────────────────────────────────────────────────

    def _load_or_build(
        self,
        corpus_text: str,
        rows: list[dict[str, str]],
        embedder: object,
        cache_path: Path,
    ) -> list[DocumentChunk]:
        if cache_path.exists():
            try:
                chunks = _load_cache(cache_path)
                logger.debug("RSSOM RAG: loaded %d chunks from cache %s", len(chunks), cache_path.name)
                return chunks
            except Exception as exc:
                logger.warning("RSSOM RAG: cache load failed (%s); rebuilding index", exc)

        docs = _build_requirement_documents(corpus_text, requirement_rows=rows, title=self._title)
        if not docs:
            docs = [
                Document(
                    content=corpus_text,
                    metadata=DocumentMetadata(
                        source_type=DocumentSource.MANUAL,
                        source_id="rssom_test_evidence_corpus",
                        title=self._title,
                        tags=["rssom", "fit", "traceability", "rag", "raw_corpus"],
                        extra_fields={"index_kind": "raw_corpus"},
                    ),
                )
            ]

        chunker = TextChunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        pipeline = ProcessingPipeline(chunker=chunker, embedder=embedder)
        chunks = pipeline.process_documents(docs)

        if chunks:
            try:
                _save_cache(cache_path, chunks)
                logger.debug("RSSOM RAG: cached %d chunks → %s", len(chunks), cache_path.name)
            except Exception as exc:
                logger.warning("RSSOM RAG: cache save failed (%s); continuing without cache", exc)
        return chunks

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready and self._retriever is not None and self._bm25 is not None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_chars: int = 420,
        requirement_id: str = "",
        title: str = "",
    ) -> list[RSSOMRAGHit]:
        """
        Triple-path retrieval: dense + BM25 + exact-ID match, fused with RRF,
        then evidence-reranked.  Always over-fetches so rerank can swap entries.
        """
        if not self.ready or not query.strip():
            return []
        assert self._retriever is not None and self._bm25 is not None

        over_k = max(top_k * 3, top_k + 6)
        expanded = expand_query(query)

        # Path 1: dense vector. Use the expanded query so paraphrases benefit.
        dense_results = self._retriever.retrieve(query=expanded, top_k=over_k)
        dense_by_chunk_id = {_chunk_id(item.chunk): item for item in dense_results}
        dense_ranking = [_chunk_id(item.chunk) for item in dense_results]

        # Path 2: BM25+. Also uses the expanded query.
        bm25_results = self._bm25.search(expanded, top_k=over_k)
        bm25_by_chunk_id = {item.doc_id: item for item in bm25_results}
        bm25_ranking = [item.doc_id for item in bm25_results]

        # Path 3: exact requirement-id. Deterministic — always trusted at top.
        exact_id_hits = self._exact_id_matches(requirement_id)

        # Reciprocal Rank Fusion across the three rankings.
        rrf_scores: dict[str, float] = {}
        sources_map: dict[str, list[str]] = {}

        def _add_ranking(ranking: list[str], label: str) -> None:
            for rank, cid in enumerate(ranking, start=1):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
                sources_map.setdefault(cid, []).append(label)

        _add_ranking(dense_ranking, "dense")
        _add_ranking(bm25_ranking, "bm25")
        _add_ranking(exact_id_hits, "exact_id")

        fused_order = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        hits: list[RSSOMRAGHit] = []
        for cid in fused_order:
            raw_hit = self._compose_hit(
                cid,
                max_chars=max_chars,
                dense_item=dense_by_chunk_id.get(cid),
                bm25_item=bm25_by_chunk_id.get(cid),
                fused_score=rrf_scores[cid],
                sources=tuple(sources_map.get(cid, ())),
            )
            if raw_hit is not None:
                hits.append(raw_hit)

        # Query-to-title coverage boost. Works even when the caller supplies no
        # hints (requirement_id/title) and handles the common case of "query
        # literally contains a requirement's full title" — critical for the
        # many near-duplicate crowding-level titles in the RSSOM matrix.
        hits = _query_title_coverage_boost(hits, query=expanded, all_chunks=self._chunks)

        hits = _evidence_rerank(hits, requirement_id=requirement_id, title=title)

        # Deduplicate by requirement_id: when a long requirement is split into
        # multiple chunks, each chunk otherwise competes for a top-k slot and
        # double-counts in the fused ranking. Keep only the best chunk per
        # requirement, preserving unknown-rid entries as-is.
        hits = _dedupe_by_requirement(hits)
        return hits[:top_k]

    # ── internals ──────────────────────────────────────────────────────────────

    def _exact_id_matches(self, requirement_id: str) -> list[str]:
        """Return chunk-ids whose metadata or text contains this exact ID."""
        rid = (requirement_id or "").strip()
        if not rid:
            return []
        rid_u, rid_l = rid.upper(), rid.lower()
        out: list[str] = []
        for chunk in self._chunks:
            meta = _merged_chunk_metadata(chunk.metadata)
            meta_rid = str(meta.get("requirement_id", "")).upper().strip()
            if meta_rid == rid_u:
                out.append(_chunk_id_from_chunk(chunk))
                continue
            if rid_l in (chunk.text or "").lower():
                out.append(_chunk_id_from_chunk(chunk))
        return out

    def _compose_hit(
        self,
        chunk_id: str,
        *,
        max_chars: int,
        dense_item: SearchResult | None,
        bm25_item: BM25Hit | None,
        fused_score: float,
        sources: tuple[str, ...],
    ) -> RSSOMRAGHit | None:
        if dense_item is not None:
            # Prefer dense's SearchResult when present (keeps its similarity score).
            meta = _merged_chunk_metadata(dense_item.chunk.metadata)
            text = (dense_item.chunk.text or "").strip()
            base_score = float(dense_item.score)
        elif bm25_item is not None:
            meta = _merged_chunk_metadata(bm25_item.metadata)
            text = (bm25_item.text or "").strip()
            # Normalise BM25 raw score into a [0,1]-ish range so evidence boosts
            # combine cleanly with dense hits.  sigmoid is sufficient here.
            base_score = 1.0 / (1.0 + math.exp(-bm25_item.score / 5.0))
        else:
            chunk = next((c for c in self._chunks if _chunk_id_from_chunk(c) == chunk_id), None)
            if chunk is None:
                return None
            meta = _merged_chunk_metadata(chunk.metadata)
            text = (chunk.text or "").strip()
            base_score = 0.5

        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        return RSSOMRAGHit(
            text=text,
            score=round(base_score, 4),
            chunk_index=_as_int(meta.get("chunk_index")),
            total_chunks=_as_int(meta.get("total_chunks")),
            requirement_id=str(meta.get("requirement_id", "")).strip() or None,
            index_kind=str(meta.get("index_kind", "")).strip() or None,
            sources=sources,
            fused_score=round(fused_score, 6),
        )


# ──────────────────────────────────────────────────────────────────────────────
# classify_requirement_with_rag  (public, used by matcher)
# ──────────────────────────────────────────────────────────────────────────────

def classify_requirement_with_rag(
    rag: "RSSOMRAGIndex | None",
    *,
    requirement_id: str,
    title: str,
    top_k: int = 4,
) -> tuple[str, "str | None", dict[str, object]]:
    """
    Retrieve RSSOM chunks for one requirement and derive a verdict.

    Returns: (verdict, snippet, meta_dict)
      verdict ∈ {PASS, FAIL, UNKNOWN, NOT_FOUND}

    Meta dict includes retrieval path agreement for auditability:
      agreement ∈ {full, partial, single}
      - full:    at least one hit is supported by all 3 paths
      - partial: at least one hit is supported by ≥2 paths
      - single:  best hit came from only one path
    """
    req_id = str(requirement_id or "").strip()
    title_s = str(title or "").strip()
    query_base = " ".join(
        part for part in (req_id, title_s, "verification test evidence pass fail result") if part
    ).strip()

    if not rag or not rag.ready or not query_base:
        return "NOT_FOUND", None, {"enabled": False, "query": query_base, "hits": []}

    hits = rag.retrieve(query_base, top_k=top_k, requirement_id=req_id, title=title_s)
    if not hits:
        return "NOT_FOUND", None, {"enabled": True, "query": query_base, "hits": []}

    texts = [h.text for h in hits if h.text]
    score_max = max((h.score for h in hits), default=0.0)

    exact_meta_hit = any((h.requirement_id or "").upper() == req_id.upper() for h in hits if req_id)
    exact_id_hit = any(req_id.lower() in text.lower() for text in texts if req_id)
    title_overlap = any(_title_overlap(title_s, text) > 0 for text in texts if title_s)
    relevant = exact_meta_hit or exact_id_hit or title_overlap

    agreement = _retrieval_agreement(hits)

    if not relevant:
        return "NOT_FOUND", None, {
            "enabled": True,
            "query": query_base,
            "hits": [_hit_dict(h) for h in hits],
            "top_score": round(score_max, 4),
            "relevance_gate": "failed",
            "agreement": agreement,
        }

    snippet = texts[0][:300] if texts else None
    verdict, method = _verdict_from_proximity(texts, req_id)
    return verdict, snippet, {
        "enabled": True,
        "query": query_base,
        "hits": [_hit_dict(h) for h in hits],
        "top_score": round(score_max, 4),
        "relevance_gate": "passed",
        "verdict_method": method,
        "agreement": agreement,
    }


def _retrieval_agreement(hits: list[RSSOMRAGHit]) -> str:
    """
    Summarise how strongly the retrieval paths agree on the top hit.

    ``full`` means the #1 hit was returned by all three paths (dense + BM25 +
    exact-id). This is the strongest possible signal short of human review and
    is explicitly surfaced in the audit trail.
    """
    if not hits:
        return "none"
    top = hits[0]
    n_sources = len(set(top.sources))
    if n_sources >= 3:
        return "full"
    if n_sources == 2:
        return "partial"
    return "single"


# ──────────────────────────────────────────────────────────────────────────────
# Verdict derivation
# ──────────────────────────────────────────────────────────────────────────────

def _verdict_from_proximity(texts: list[str], requirement_id: str) -> tuple[str, str]:
    """
    First search within _VERDICT_WINDOW chars of the requirement_id in each hit.
    If found, return verdict and method="proximity".
    Fall back to full-text scan with method="full_text".
    """
    rid_l = requirement_id.lower()
    if rid_l:
        for text in texts:
            tl = text.lower()
            idx = tl.find(rid_l)
            if idx >= 0:
                start = max(0, idx - _VERDICT_WINDOW)
                end = min(len(text), idx + len(requirement_id) + _VERDICT_WINDOW)
                window = text[start:end]
                if _LINE_FAIL.search(window):
                    return "FAIL", "proximity"
                if _LINE_PASS.search(window):
                    return "PASS", "proximity"

    joined = "\n".join(texts)
    if _LINE_FAIL.search(joined):
        return "FAIL", "full_text"
    if _LINE_PASS.search(joined):
        return "PASS", "full_text"
    return "UNKNOWN", "full_text"


# ──────────────────────────────────────────────────────────────────────────────
# Evidence-aware reranker
# ──────────────────────────────────────────────────────────────────────────────

def _evidence_rerank(
    hits: list[RSSOMRAGHit],
    *,
    requirement_id: str,
    title: str,
) -> list[RSSOMRAGHit]:
    """
    Post-retrieval reranker. Boosts hits that structurally match the target
    requirement. Works on top of the fused RRF ordering.

    Boost table:
      +0.30  exact requirement_id in metadata.requirement_id
      +0.15  exact requirement_id anywhere in text
      +0.04 × min(4, title-token overlap count)
      +0.08 × title-coverage ratio (|matched tokens| / |title tokens|)
      +0.06 × min(3, title-bigram matches)
      +0.05  if three paths agree on this hit
      +0.02  if two paths agree on this hit
    Final score capped at 0.99.

    The title-coverage ratio and bigram component are crucial when many
    requirements share near-identical titles (e.g. the five crowding-level
    variants in the RSSOM FIT matrix).  A query that literally contains the
    full title of one specific requirement will then win over siblings that
    only overlap in single tokens.
    """
    if not requirement_id and not title:
        return hits

    rid_u = (requirement_id or "").upper()
    rid_l = (requirement_id or "").lower()
    title_tokens = _title_tokens(title)
    title_bigrams = _title_bigrams(title)
    result: list[RSSOMRAGHit] = []

    for hit in hits:
        boost = 0.0
        text_l = (hit.text or "").lower()

        if rid_u and (hit.requirement_id or "").upper() == rid_u:
            boost += 0.30
        elif rid_l and rid_l in text_l:
            boost += 0.15

        if title_tokens:
            matched = sum(1 for tok in title_tokens if tok in text_l)
            boost += min(4, matched) * 0.04
            coverage = matched / len(title_tokens)
            boost += 0.08 * coverage

        if title_bigrams:
            bg_matches = sum(1 for bg in title_bigrams if bg in text_l)
            boost += min(3, bg_matches) * 0.06

        n_sources = len(set(hit.sources))
        if n_sources >= 3:
            boost += 0.05
        elif n_sources == 2:
            boost += 0.02

        if boost == 0.0:
            result.append(hit)
        else:
            new_score = min(0.99, round(hit.score + boost, 4))
            result.append(dataclasses.replace(hit, score=new_score))

    return sorted(result, key=lambda h: h.score, reverse=True)


def _dedupe_by_requirement(hits: list[RSSOMRAGHit]) -> list[RSSOMRAGHit]:
    """Keep only the highest-scoring hit per requirement_id."""
    best: dict[str, RSSOMRAGHit] = {}
    out_order: list[RSSOMRAGHit] = []
    for hit in hits:
        rid = (hit.requirement_id or "").strip().upper()
        if not rid:
            out_order.append(hit)
            continue
        existing = best.get(rid)
        if existing is None or hit.score > existing.score:
            best[rid] = hit
    # Merge: unknown-rid hits keep their original order relative to known ones;
    # known-rid hits are sorted by score.
    known_sorted = sorted(best.values(), key=lambda h: h.score, reverse=True)
    merged = known_sorted + out_order
    return sorted(merged, key=lambda h: h.score, reverse=True)


def _title_tokens(title: str) -> list[str]:
    return [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in _STOPWORDS
    ]


def _title_bigrams(title: str) -> list[str]:
    tokens = _title_tokens(title)
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def _query_title_coverage_boost(
    hits: list[RSSOMRAGHit],
    *,
    query: str,
    all_chunks: list[DocumentChunk],
) -> list[RSSOMRAGHit]:
    """
    Post-fusion boost: how well does the query cover each candidate's
    requirement title?

    For each hit we compute:
      coverage    = |title_tokens ∩ query_tokens| / |title_tokens|
      specificity = |title_tokens ∩ query_tokens| / |query_tokens|

    When multiple candidates share similar titles (e.g. the "Vehicle crowding
    levels visualization on RTM section" family), the one whose title is most
    fully covered by the query wins. A perfect 1.0 coverage gets +0.15;
    partial coverage is rewarded proportionally. The specificity factor adds
    a small extra bonus when the query's *distinguishing* tokens land in the
    title (lifting C6-APCS-2 over C6-APCS-10 when the query specifically
    mentions RTM/real time monitoring).

    This runs without requiring the caller to know the target requirement_id
    or title — it uses only the query and each chunk's own title metadata.
    """
    if not hits or not query:
        return hits

    query_tokens = set(_title_tokens(query))
    if not query_tokens:
        return hits

    # Build a requirement_id → title lookup from all indexed chunks.
    rid_to_title: dict[str, str] = {}
    for chunk in all_chunks:
        meta = _merged_chunk_metadata(chunk.metadata)
        rid = str(meta.get("requirement_id") or "").strip().upper()
        rtitle = str(meta.get("requirement_title") or "").strip()
        if rid and rtitle and rid not in rid_to_title:
            rid_to_title[rid] = rtitle

    boosted: list[RSSOMRAGHit] = []
    for hit in hits:
        rid = (hit.requirement_id or "").upper().strip()
        rtitle = rid_to_title.get(rid, "")
        if not rtitle:
            boosted.append(hit)
            continue
        title_tok = set(_title_tokens(rtitle))
        if not title_tok:
            boosted.append(hit)
            continue
        overlap = len(title_tok & query_tokens)
        if overlap <= 0:
            boosted.append(hit)
            continue
        coverage = overlap / len(title_tok)
        specificity = overlap / len(query_tokens)
        boost = 0.15 * coverage + 0.05 * specificity
        new_score = min(0.99, round(hit.score + boost, 4))
        boosted.append(dataclasses.replace(hit, score=new_score))
    return sorted(boosted, key=lambda h: h.score, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Title overlap helper
# ──────────────────────────────────────────────────────────────────────────────

def _title_overlap(title: str, text: str) -> int:
    tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in _STOPWORDS
    }
    if not tokens:
        return 0
    text_l = str(text or "").lower()
    return sum(1 for tok in tokens if tok in text_l)


# ──────────────────────────────────────────────────────────────────────────────
# Hit helpers
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_id(chunk_obj) -> str:
    cid = getattr(chunk_obj, "id", None)
    if cid:
        return str(cid)
    return _chunk_id_from_chunk(chunk_obj)


def _chunk_id_from_chunk(chunk: DocumentChunk) -> str:
    cid = getattr(chunk, "id", None)
    if cid:
        return str(cid)
    meta = chunk.metadata or {}
    ci = meta.get("chunk_index")
    src = meta.get("source_id") or meta.get("parent_document_id") or ""
    return f"{src}:{ci}"


def _build_bm25_from_chunks(chunks: list[DocumentChunk]) -> BM25Index:
    index = BM25Index()
    for chunk in chunks:
        index.add(
            doc_id=_chunk_id_from_chunk(chunk),
            text=chunk.text or "",
            metadata=chunk.metadata or {},
        )
    index.finalize()
    return index


def _hit_dict(hit: RSSOMRAGHit) -> dict[str, object]:
    return {
        "text": hit.text,
        "score": hit.score,
        "chunk_index": hit.chunk_index,
        "total_chunks": hit.total_chunks,
        "requirement_id": hit.requirement_id,
        "index_kind": hit.index_kind,
        "sources": list(hit.sources),
        "fused_score": hit.fused_score,
    }


def _as_int(value: object) -> "int | None":
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _merged_chunk_metadata(metadata: "dict[str, object] | None") -> dict[str, object]:
    """Flatten source_extra_fields into the top-level metadata dict."""
    base = dict(metadata or {})
    extra = base.get("source_extra_fields")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key not in base or base.get(key) in (None, ""):
                base[key] = value
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Requirement document builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_requirement_documents(
    corpus_text: str,
    *,
    requirement_rows: "list[dict[str, str]] | None",
    title: str,
) -> list[Document]:
    rows = requirement_rows or []
    if not rows:
        return []

    corpus = str(corpus_text or "")
    corpus_l = corpus.lower()
    all_lines = corpus.splitlines()
    docs: list[Document] = []
    seen: set[str] = set()

    for row in rows:
        rid = str(row.get("requirement_id", "")).strip()
        req_title = str(row.get("title", "")).strip()
        if not rid or rid.upper() in seen:
            continue
        if rid.lower() not in corpus_l:
            continue
        seen.add(rid.upper())

        sections = _extract_requirement_sections(all_lines, rid, req_title)
        body: list[str] = [
            f"Requirement ID: {rid}",
            f"Requirement title: {req_title}",
        ]
        # Title-weight boost: BM25 gives higher TF weight to terms that appear
        # more often in a document, so we repeat the title once.  This lifts
        # the correct requirement above neighbours whose chunks happen to
        # include the target ID in their context windows.
        if req_title:
            body.append(f"Title restated: {req_title}")
        # Add expanded synonyms to help BM25+dense both match paraphrases.
        expanded_title = expand_query(req_title)
        if expanded_title and expanded_title != req_title:
            body.append(f"Synonyms: {expanded_title}")
        if sections.get("verification_status"):
            body.append(f"Verification status: {sections['verification_status']}")
        # Keep only excerpts that are truly about THIS requirement. A line
        # that mentions only some other requirement ID (e.g. a neighbouring
        # row in the FIT matrix) gets filtered out to avoid cross-pollution
        # that confuses BM25 on short technical queries.
        own_excerpts = _filter_own_excerpts(sections.get("excerpts") or [], rid)
        if own_excerpts:
            body.append("RSSOM evidence:")
            body.extend(f"  {line}" for line in own_excerpts)
        if sections.get("test_objective"):
            body.append(f"Test objective: {sections['test_objective']}")
        if sections.get("expected_result"):
            body.append(f"Expected result: {sections['expected_result']}")

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


_REQ_ID_RX = re.compile(r"\bC6-APCS-\d+\b", re.IGNORECASE)


def _filter_own_excerpts(excerpts: list[str], target_rid: str) -> list[str]:
    """
    Drop excerpt lines that reference **only** other requirement IDs, keeping
    lines that either contain the target ID or are pure prose/context without
    any requirement-ID label.

    This is critical for short technical queries (e.g. "APCS IAMS interface"):
    the target requirement's context window often includes neighbouring FIT
    matrix rows that mention OTHER requirement IDs with very similar titles,
    which otherwise bleed into the target's indexed text.
    """
    out: list[str] = []
    target = target_rid.upper()
    for line in excerpts:
        ids_found = {m.group(0).upper() for m in _REQ_ID_RX.finditer(line)}
        if not ids_found:
            out.append(line)
            continue
        if target in ids_found:
            out.append(line)
    return out


def _extract_requirement_sections(
    all_lines: list[str],
    requirement_id: str,
    title: str,
    *,
    ctx_before: int = 2,
    ctx_after: int = 5,
    max_excerpts: int = 10,
) -> dict[str, object]:
    """
    Extract structured sections from the corpus for one requirement.

    Returns a dict with keys:
      excerpts            list[str]     lines mentioning the ID + context window
      test_objective      str | None    first line matching test-objective keywords
      expected_result     str | None    first line matching expected-result keywords
      verification_status str | None    pass / fail / n/a token near the ID
    """
    rid_l = requirement_id.lower()
    title_tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in _STOPWORDS
    ][:8]

    excerpts: list[str] = []
    test_objective: str | None = None
    expected_result: str | None = None
    verification_status: str | None = None
    seen_lines: set[str] = set()

    for idx, raw in enumerate(all_lines):
        line = raw.strip()
        if not line or line in seen_lines:
            continue
        ll = line.lower()

        if rid_l in ll:
            start = max(0, idx - ctx_before)
            end = min(len(all_lines), idx + ctx_after + 1)
            window = [all_lines[i].strip() for i in range(start, end) if all_lines[i].strip()]
            for wl in window:
                if not wl or wl in seen_lines:
                    continue
                excerpts.append(wl[:280])
                seen_lines.add(wl)
                if _OBJ_RX.search(wl) and test_objective is None:
                    test_objective = wl[:200]
                if _EXP_RX.search(wl) and expected_result is None:
                    expected_result = wl[:200]
                m = _VER_RX.search(wl)
                if m and verification_status is None:
                    verification_status = m.group(0).strip()
            if len(excerpts) >= max_excerpts:
                break
            continue

        if len(excerpts) < max_excerpts and title_tokens:
            overlap = sum(1 for tok in title_tokens if tok in ll)
            if overlap >= 2 and line not in seen_lines:
                excerpts.append(line[:280])
                seen_lines.add(line)

    return {
        "excerpts": excerpts[:max_excerpts],
        "test_objective": test_objective,
        "expected_result": expected_result,
        "verification_status": verification_status,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistent cache
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_VERSION = "4"


def _corpus_hash(
    corpus_text: str,
    rows: list[dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    row_keys = sorted((r.get("requirement_id", ""), r.get("title", "")) for r in rows)
    payload = json.dumps(
        {
            "v": _CACHE_VERSION,
            "corpus_len": len(corpus_text),
            "corpus_tail": corpus_text[-256:],
            "rows": row_keys,
            "cs": chunk_size,
            "co": chunk_overlap,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _resolve_cache_dir(cache_dir: "Path | str | None") -> Path:
    if cache_dir is not None:
        p = Path(cache_dir)
    else:
        p = Path(os.getenv("RSSOM_RAG_CACHE_DIR", "output/rag_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_cache(path: Path, chunks: list[DocumentChunk]) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "count": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "text": c.text,
                # Full metadata including nested dicts (e.g. source_extra_fields).
                # Previously this silently dropped dict values which caused
                # requirement_id / index_kind to disappear on cached runs.
                "metadata": _json_safe(c.metadata or {}),
                "embedding": c.embedding,
            }
            for c in chunks
            if c.embedding
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_cache(path: Path) -> list[DocumentChunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in ("1", "2", "3", "4"):
        raise ValueError(f"unsupported cache version: {payload.get('version')!r}")
    return [
        DocumentChunk(
            id=c["id"],
            text=c["text"],
            metadata=c.get("metadata") or {},
            embedding=c["embedding"],
        )
        for c in payload.get("chunks", [])
        if c.get("embedding")
    ]
