from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from processing.embedder import make_embedder_from_env
from rag.retriever import RAGRetriever
from vectordb import VectorDBConfig
from vectordb.providers.qdrant_provider import QdrantProvider

_DEFAULT_STORAGE: Final[str] = "qdrant_storage/regulatory_oracle"
_DEFAULT_COLLECTION: Final[str] = "regulatory_cei_en_50128"


def _rag_enabled_flag() -> str:
    return os.getenv("REGULATORY_RAG", "auto").strip().lower()


def _should_attempt_retrieval(storage_path: Path) -> tuple[bool, str | None]:
    flag = _rag_enabled_flag()
    if flag in {"0", "false", "no", "off"}:
        return False, "REGULATORY_RAG disabled"
    if flag in {"1", "true", "yes", "on"}:
        return True, None
    # auto
    if not storage_path.exists():
        return False, "regulatory Qdrant path missing (build index or set REGULATORY_QDRANT_PATH)"
    return True, None


def retrieve_regulatory_clauses(
    query: str,
    *,
    repo_root: Path,
    top_k: int = 5,
    max_text_chars: int = 1200,
) -> dict[str, Any]:
    """
    Retrieve top regulatory clauses from the local Qdrant index.

    Environment:
    - REGULATORY_RAG: auto (default), 1/on, 0/off
    - REGULATORY_QDRANT_PATH: path relative to repo root or absolute
    - REGULATORY_COLLECTION: collection name (default regulatory_cei_en_50128)
    - EMBEDDING_PROVIDER / HASH_EMBED_DIM / LOCAL_EMBED_MODEL: must match index build
    """
    rel = os.getenv("REGULATORY_QDRANT_PATH", _DEFAULT_STORAGE).strip()
    storage = Path(rel) if os.path.isabs(rel) else (repo_root / rel).resolve()
    collection = os.getenv("REGULATORY_COLLECTION", _DEFAULT_COLLECTION).strip() or _DEFAULT_COLLECTION

    attempt, skip_reason = _should_attempt_retrieval(storage)
    if not attempt:
        return {"status": "skipped", "reason": skip_reason, "hits": []}

    provider: QdrantProvider | None = None
    try:
        provider = QdrantProvider()
        provider.connect(
            VectorDBConfig(
                provider="qdrant",
                options={"path": str(storage)},
            )
        )
        embedder = make_embedder_from_env()
        retriever = RAGRetriever(
            provider=provider,
            embedder=embedder,
            collection_name=collection,
            hybrid_search=True,
        )
        results = retriever.retrieve(query=query, top_k=top_k)
        hits: list[dict[str, Any]] = []
        for result in results:
            meta = result.chunk.metadata or {}
            text = (result.chunk.text or "").strip()
            if len(text) > max_text_chars:
                text = text[: max_text_chars - 3].rstrip() + "..."
            hits.append(
                {
                    "clause_id": meta.get("clause_id"),
                    "title": meta.get("title"),
                    "score": round(float(result.score), 4),
                    "page_start": meta.get("page_start"),
                    "text": text,
                }
            )
        return {"status": "ok", "reason": None, "hits": hits, "storage_path": str(storage), "collection": collection}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "hits": []}
    finally:
        if provider is not None:
            try:
                provider.disconnect()
            except Exception:
                pass
