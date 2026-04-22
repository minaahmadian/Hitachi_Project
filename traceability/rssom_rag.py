"""
RSSOM / FIT semantic retrieval for traceability.

Improvements over v1:
- Persistent disk cache (hash-based): no re-embedding on every run.
- Richer requirement entries: context windows + test-objective / expected-result extraction.
- Evidence-aware post-retrieval reranker: boosts exact requirement_id matches.
- Proximity-based verdict derivation: looks for PASS/FAIL near the requirement ID, not
  just anywhere in the joined text.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from documents import Document, DocumentMetadata, DocumentSource
from processing.chunker import ChunkingStrategy, TextChunker
from processing.embedder import make_embedder_from_env
from processing.pipeline import ProcessingPipeline
from rag.retriever import RAGRetriever
from vectordb import SearchResult, VectorDBConfig
from vectordb.providers.inmemory_provider import InMemoryProvider
from vectordb.types import DocumentChunk

logger = logging.getLogger(__name__)

_VERDICT_WINDOW = 320          # chars around req_id for proximity verdict search
_LINE_FAIL = re.compile(r"\bfail(?:ed|ure)?\b|✗", re.IGNORECASE)
_LINE_PASS = re.compile(r"\bpass(?:ed)?\b|✓|✔|\bok\b", re.IGNORECASE)
_OBJ_RX    = re.compile(r"test\s*objective|test\s*purpose|objective\s*:", re.IGNORECASE)
_EXP_RX    = re.compile(r"expected\s*result|pass\s*criteria|acceptance\s*criteria", re.IGNORECASE)
_VER_RX    = re.compile(r"\b(passed|failed|not\s+tested|n/?a|complete|verified)\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    "and", "with", "the", "for", "level", "check", "through",
    "its", "via", "that", "this", "from", "into", "test",
})


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


# ──────────────────────────────────────────────────────────────────────────────
# Index
# ──────────────────────────────────────────────────────────────────────────────

class RSSOMRAGIndex:
    """
    Requirement-centric vector index over the RSSOM/FIT evidence corpus.

    Build flow:
      1. One ``Document`` per requirement (ID + title + context excerpts).
      2. Chunk + embed via ``ProcessingPipeline``.
      3. Cache chunks + embeddings to disk (hash-keyed JSON).  Subsequent runs
         skip embedding entirely and load in milliseconds.
      4. Load into ``InMemoryProvider`` for cosine search.
      5. ``retrieve()`` over-fetches then applies evidence-aware reranking.
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
        self._ready = False
        self._title = title
        self._chunk_size    = int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_SIZE",    "1400") or "1400")
        self._chunk_overlap = int(os.getenv("RSSOM_RAG_ENTRY_CHUNK_OVERLAP", "120")  or "120")

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
        return self._ready and self._retriever is not None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_chars: int = 420,
        requirement_id: str = "",
        title: str = "",
    ) -> list[RSSOMRAGHit]:
        """Retrieve + evidence-aware rerank. Always over-fetches 3× then trims."""
        if not self.ready or not query.strip():
            return []
        assert self._retriever is not None
        over_k = max(top_k * 3, top_k + 6)
        candidates = self._retriever.retrieve(query=query, top_k=over_k)
        hits = [_to_hit(item, max_chars=max_chars) for item in candidates]
        hits = _evidence_rerank(hits, requirement_id=requirement_id, title=title)
        return hits[:top_k]


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

    Changes vs v1:
    - Passes requirement_id + title into retrieve() for evidence-aware reranking.
    - Uses proximity-based verdict (looks near req ID, not just whole joined text).
    - Exposes ``verdict_method`` in meta for traceability.
    """
    req_id  = str(requirement_id or "").strip()
    title_s = str(title or "").strip()
    query = " ".join(
        part for part in (req_id, title_s, "verification test evidence pass fail result") if part
    ).strip()

    if not rag or not rag.ready or not query:
        return "NOT_FOUND", None, {"enabled": False, "query": query, "hits": []}

    hits = rag.retrieve(query, top_k=top_k, requirement_id=req_id, title=title_s)
    if not hits:
        return "NOT_FOUND", None, {"enabled": True, "query": query, "hits": []}

    texts     = [h.text for h in hits if h.text]
    score_max = max((h.score for h in hits), default=0.0)

    exact_meta_hit = any((h.requirement_id or "").upper() == req_id.upper() for h in hits if req_id)
    exact_id_hit   = any(req_id.lower() in text.lower() for text in texts if req_id)
    title_overlap  = any(_title_overlap(title_s, text) > 0 for text in texts if title_s)
    relevant = exact_meta_hit or exact_id_hit or title_overlap

    if not relevant:
        return "NOT_FOUND", None, {
            "enabled": True,
            "query": query,
            "hits": [_hit_dict(h) for h in hits],
            "top_score": round(score_max, 4),
            "relevance_gate": "failed",
        }

    snippet = texts[0][:300] if texts else None
    verdict, method = _verdict_from_proximity(texts, req_id)
    return verdict, snippet, {
        "enabled": True,
        "query": query,
        "hits": [_hit_dict(h) for h in hits],
        "top_score": round(score_max, 4),
        "relevance_gate": "passed",
        "verdict_method": method,
    }


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
            tl  = text.lower()
            idx = tl.find(rid_l)
            if idx >= 0:
                start  = max(0, idx - _VERDICT_WINDOW)
                end    = min(len(text), idx + len(requirement_id) + _VERDICT_WINDOW)
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
    Post-retrieval reranker.  Boosts hits that structurally match the target requirement.

    Boost table:
      +0.30  exact requirement_id in metadata.requirement_id  (structural match → most reliable)
      +0.15  exact requirement_id anywhere in text
      +0.025 per title-token overlap, capped at +0.10
    Final score capped at 0.99.
    """
    if not requirement_id and not title:
        return hits

    rid_u = (requirement_id or "").upper()
    rid_l = (requirement_id or "").lower()
    result: list[RSSOMRAGHit] = []

    for hit in hits:
        boost  = 0.0
        text_l = (hit.text or "").lower()

        if rid_u and (hit.requirement_id or "").upper() == rid_u:
            boost += 0.30
        elif rid_l and rid_l in text_l:
            boost += 0.15

        if title:
            overlap = min(4, _title_overlap(title, hit.text or ""))
            boost  += overlap * 0.025

        if boost == 0.0:
            result.append(hit)
        else:
            new_score = min(0.99, round(hit.score + boost, 4))
            result.append(dataclasses.replace(hit, score=new_score))

    return sorted(result, key=lambda h: h.score, reverse=True)


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

def _to_hit(item: SearchResult, *, max_chars: int) -> RSSOMRAGHit:
    meta = _merged_chunk_metadata(item.chunk.metadata)
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


def _as_int(value: object) -> "int | None":
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _merged_chunk_metadata(metadata: "dict[str, object] | None") -> dict[str, object]:
    """Flatten source_extra_fields into the top-level metadata dict."""
    base  = dict(metadata or {})
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

    corpus   = str(corpus_text or "")
    corpus_l = corpus.lower()
    all_lines = corpus.splitlines()
    docs: list[Document] = []
    seen: set[str] = set()

    for row in rows:
        rid       = str(row.get("requirement_id", "")).strip()
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
        if sections.get("verification_status"):
            body.append(f"Verification status: {sections['verification_status']}")
        if sections.get("excerpts"):
            body.append("RSSOM evidence:")
            body.extend(f"  {line}" for line in sections["excerpts"])
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


def _extract_requirement_sections(
    all_lines: list[str],
    requirement_id: str,
    title: str,
    *,
    ctx_before: int = 2,
    ctx_after: int  = 5,
    max_excerpts: int = 10,
) -> dict[str, object]:
    """
    Extract structured sections from the corpus for one requirement.

    Returns a dict with keys:
      excerpts           list[str]     lines mentioning the ID + context window
      test_objective     str | None    first line matching test-objective keywords
      expected_result    str | None    first line matching expected-result keywords
      verification_status str | None   pass / fail / n/a token near the ID
    """
    rid_l = requirement_id.lower()
    title_tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title or "")
        if tok.lower() not in _STOPWORDS
    ][:8]

    excerpts: list[str]          = []
    test_objective: str | None   = None
    expected_result: str | None  = None
    verification_status: str | None = None
    seen_lines: set[str]         = set()

    for idx, raw in enumerate(all_lines):
        line = raw.strip()
        if not line or line in seen_lines:
            continue
        ll = line.lower()

        if rid_l in ll:
            start  = max(0, idx - ctx_before)
            end    = min(len(all_lines), idx + ctx_after + 1)
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

_CACHE_VERSION = "3"


def _corpus_hash(
    corpus_text: str,
    rows: list[dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    row_keys = sorted((r.get("requirement_id", ""), r.get("title", "")) for r in rows)
    payload  = json.dumps(
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
                "metadata": {
                    k: v
                    for k, v in (c.metadata or {}).items()
                    if isinstance(v, (str, int, float, bool, list, type(None)))
                },
                "embedding": c.embedding,
            }
            for c in chunks
            if c.embedding
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_cache(path: Path) -> list[DocumentChunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in ("1", "2", "3"):
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
