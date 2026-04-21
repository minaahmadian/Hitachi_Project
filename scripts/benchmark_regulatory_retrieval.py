#!/usr/bin/env python3
"""
Numerical benchmark for regulatory clause retrieval (Qdrant + RAGRetriever).

Reads labeled queries with expected ``clause_id`` values from CEI EN 50128 index,
runs :func:`regulatory.clause_retrieval.retrieve_regulatory_clauses`, and reports
Hit@K, Precision@K, Recall@K, and MRR (mean reciprocal rank of the first hit).

Prerequisites (same as production RAG):
  - Build index: ``python regulatory/build_regulatory_index.py`` (or existing path)
  - ``REGULATORY_RAG=auto`` with ``qdrant_storage/...`` present, or ``REGULATORY_RAG=1``
  - ``EMBEDDING_PROVIDER`` / hash settings must match index build

Usage:
  python scripts/benchmark_regulatory_retrieval.py \\
    --golden data/eval/retrieval/golden_queries.json \\
    --out output/retrieval_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regulatory.clause_retrieval import retrieve_regulatory_clauses


def _norm_cid(x: Any) -> str:
    return str(x or "").strip()


def _retrieved_ids(hits: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for h in hits:
        cid = _norm_cid(h.get("clause_id"))
        if cid:
            out.append(cid)
    return out


def _first_relevant_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return i
    return None


def _prec_rec_at_k(retrieved: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    if k <= 0:
        return 0.0, 0.0
    top = retrieved[:k]
    inter = sum(1 for x in top if x in relevant)
    prec = inter / k
    rec = inter / len(relevant) if relevant else 0.0
    return prec, rec


def _eval_one(
    retrieved: list[str],
    relevant: set[str],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    rr = _first_relevant_rank(retrieved, relevant)
    mrr_contrib = (1.0 / rr) if rr is not None else 0.0
    row: dict[str, Any] = {
        "first_relevant_rank": rr,
        "mrr_contribution": round(mrr_contrib, 6),
    }
    for k in ks:
        hit = 1 if rr is not None and rr <= k else 0
        p, r = _prec_rec_at_k(retrieved, relevant, k)
        row[f"hit_at_{k}"] = hit
        row[f"precision_at_{k}"] = round(p, 4)
        row[f"recall_at_{k}"] = round(r, 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark regulatory clause retrieval")
    ap.add_argument(
        "--golden",
        type=Path,
        default=REPO_ROOT / "data/eval/retrieval/golden_queries.json",
        help="JSON list of {id, query, relevant_clause_ids}",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Retrieve this many hits per query")
    ap.add_argument(
        "--ks",
        default="1,3,5,10",
        help="Comma-separated K values for P/R/Hit (default 1,3,5,10)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write full JSON metrics here")
    args = ap.parse_args()

    ks = tuple(int(x.strip()) for x in str(args.ks).split(",") if x.strip())
    raw = json.loads(args.golden.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("golden file must be a JSON array")

    per_query: list[dict[str, Any]] = []
    mrr_vals: list[float] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        rel = {_norm_cid(x) for x in (item.get("relevant_clause_ids") or []) if _norm_cid(x)}
        if not query or not rel:
            continue

        res = retrieve_regulatory_clauses(
            query,
            repo_root=REPO_ROOT,
            top_k=max(args.top_k, max(ks) if ks else 10),
            max_text_chars=400,
        )
        status = str(res.get("status", ""))
        hits = res.get("hits") if isinstance(res.get("hits"), list) else []
        retrieved = _retrieved_ids(hits)

        m = _eval_one(retrieved, rel, ks)
        mrr_vals.append(m["mrr_contribution"])
        per_query.append(
            {
                "id": qid,
                "query": query,
                "relevant_clause_ids": sorted(rel),
                "retrieval_status": status,
                "retrieved_clause_ids": retrieved[: args.top_k],
                "metrics": m,
                "skip_reason": res.get("reason") if status != "ok" else None,
            }
        )

    summary: dict[str, Any] = {
        "golden_path": str(args.golden.resolve()),
        "queries_evaluated": len(per_query),
        "mean_reciprocal_rank": round(mean(mrr_vals), 6) if mrr_vals else 0.0,
    }
    for k in ks:
        hits_k = [p["metrics"].get(f"hit_at_{k}", 0) for p in per_query]
        precs = [p["metrics"].get(f"precision_at_{k}", 0.0) for p in per_query]
        recs = [p["metrics"].get(f"recall_at_{k}", 0.0) for p in per_query]
        summary[f"mean_hit_at_{k}"] = round(mean(hits_k), 4) if hits_k else 0.0
        summary[f"mean_precision_at_{k}"] = round(mean(precs), 4) if precs else 0.0
        summary[f"mean_recall_at_{k}"] = round(mean(recs), 4) if recs else 0.0

    print("=== Regulatory retrieval benchmark ===")
    print(f"golden: {args.golden}")
    print(f"MRR (mean): {summary['mean_reciprocal_rank']}")
    for k in ks:
        print(
            f"Hit@{k}: {summary.get(f'mean_hit_at_{k}')}  "
            f"P@{k}: {summary.get(f'mean_precision_at_{k}')}  "
            f"R@{k}: {summary.get(f'mean_recall_at_{k}')}",
        )
    print()
    for p in per_query:
        m = p["metrics"]
        rnk = m.get("first_relevant_rank")
        print(f"{p['id']}: rank={rnk} status={p['retrieval_status']} retrieved={p['retrieved_clause_ids'][:5]}...")

    out_blob = {
        "summary": summary,
        "per_query": per_query,
        "env": {
            "REGULATORY_RAG": os.getenv("REGULATORY_RAG", ""),
            "REGULATORY_QDRANT_PATH": os.getenv("REGULATORY_QDRANT_PATH", ""),
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", ""),
        },
    }
    if args.out:
        args.out = args.out.expanduser()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_blob, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
