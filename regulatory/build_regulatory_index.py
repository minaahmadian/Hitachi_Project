from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid5, NAMESPACE_URL

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from processing.embedder import GroqEmbedder, HashEmbedder, LocalEmbedder
from rag.context_builder import ContextBuilder, ContextStrategy
from rag.retriever import RAGRetriever
from vectordb import DocumentChunk, VectorDBConfig
from vectordb.providers.qdrant_provider import QdrantProvider


def _load_clause_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Clause file must be a JSON list: {path}")
    records: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clause_id = str(item.get("clause_id", "")).strip()
        text = str(item.get("text", "")).strip()
        title = str(item.get("title", "")).strip()
        if not clause_id or not text:
            continue
        records.append(item)
    return records


def _make_embedder() -> Any:
    provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()
    if provider == "groq":
        return GroqEmbedder(
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            api_key=os.getenv("GROQ_API_KEY"),
        )
    if provider in {"local", "sentence_transformers", "sentence-transformers"}:
        return LocalEmbedder(
            model_name=os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"),
        )
    dim = int(os.getenv("HASH_EMBED_DIM", "128"))
    return HashEmbedder(dim=dim)


def _to_chunks(records: list[dict[str, Any]], vectors: list[list[float]]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for record, vector in zip(records, vectors):
        clause_id = str(record["clause_id"]).strip()
        title = str(record.get("title", "")).strip()
        text = str(record.get("text", "")).strip()
        source_document = str(record.get("source_document", "CEI EN 50128.pdf"))
        page_start = record.get("page_start")

        chunk_text = f"{clause_id} {title}\n{text}".strip()
        point_id = str(uuid5(NAMESPACE_URL, f"cei_en_50128::{clause_id}"))
        chunks.append(
            DocumentChunk(
                id=point_id,
                text=chunk_text,
                metadata={
                    "source": source_document,
                    "filename": source_document,
                    "document_type": "regulation_clause",
                    "regulation": "CEI EN 50128",
                    "clause_id": clause_id,
                    "title": title,
                    "section": str(record.get("section", "")).strip(),
                    "page_start": page_start,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                embedding=vector,
            )
        )
    return chunks


def _index_chunks(
    *,
    chunks: list[DocumentChunk],
    vector_dim: int,
    collection: str,
    storage_path: Path,
) -> QdrantProvider:
    provider = QdrantProvider()
    provider.connect(
        VectorDBConfig(
            provider="qdrant",
            options={"path": str(storage_path)},
        )
    )
    provider.create_collection(name=collection, dimension=vector_dim, distance_metric="cosine")
    provider.upsert_documents(collection_name=collection, chunks=chunks)
    return provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEI EN 50128 regulatory vector index")
    parser.add_argument(
        "--clauses",
        default="data/regulatory/cei_en_50128_clauses.json",
        help="Path to normalized clause JSON",
    )
    parser.add_argument(
        "--collection",
        default="regulatory_cei_en_50128",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--storage-path",
        default="qdrant_storage/regulatory_oracle",
        help="Local Qdrant storage path",
    )
    parser.add_argument(
        "--sample-query",
        default="traceability requirements and verification evidence",
        help="Sanity-check query after indexing",
    )
    parser.add_argument(
        "--skip-query-check",
        action="store_true",
        help="Skip post-index retrieval sanity check",
    )
    args = parser.parse_args()

    clauses_path = (REPO_ROOT / args.clauses).resolve()
    storage_path = (REPO_ROOT / args.storage_path).resolve()

    records = _load_clause_records(clauses_path)
    if not records:
        raise RuntimeError(f"No valid clause records found in {clauses_path}")

    embedder = _make_embedder()
    vectors = embedder.embed([str(item["text"]) for item in records])
    if not vectors:
        raise RuntimeError("Embedding returned zero vectors")

    dim = len(vectors[0])
    chunks = _to_chunks(records, vectors)

    provider = _index_chunks(
        chunks=chunks,
        vector_dim=dim,
        collection=args.collection,
        storage_path=storage_path,
    )

    print(f"Indexed clauses: {len(chunks)}")
    print(f"Vector dimension: {dim}")
    print(f"Collection: {args.collection}")
    print(f"Storage path: {storage_path}")

    if args.skip_query_check:
        print("Sample retrieval check: skipped by flag")
    else:
        try:
            retriever = RAGRetriever(
                provider=provider,
                embedder=embedder,
                collection_name=args.collection,
                hybrid_search=True,
            )
            sample_results = retriever.retrieve(query=args.sample_query, top_k=3)
            context = ContextBuilder(strategy=ContextStrategy.HIGHEST_SCORE).build_context(
                sample_results,
                max_tokens=400,
            )
            print(f"Sample retrieval hits: {len(sample_results)}")
            if sample_results:
                top = sample_results[0]
                print(f"Top hit clause: {top.chunk.metadata.get('clause_id')} ({top.chunk.metadata.get('title')})")
            print("Context preview:")
            print(context[:600])
        except Exception as exc:
            print(f"Sample retrieval check failed (index build still succeeded): {exc}")

    provider.disconnect()


if __name__ == "__main__":
    main()
