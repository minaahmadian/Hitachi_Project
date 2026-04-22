#!/usr/bin/env python3
"""
Numerical benchmark for RSSOM retrieval quality.

Builds the RSSOM/FIT corpus the same way the runtime matcher does, then evaluates
semantic retrieval over that corpus using labeled natural-language queries.

Metrics:
- Hit@K
- Precision@K
- Recall@K
- MRR

Scoring rule:
- A retrieved chunk counts as relevant when it contains at least one of the golden
  ``relevant_requirement_ids`` tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.docx_parser import parse_docx
from core.project_ingestion import build_test_evidence_corpus
from core.rssom_requirements_trace import rows_from_rssom_docx
from traceability.rssom_rag import RSSOMRAGIndex


def _norm(x: Any) -> str:
    return str(x or "").strip().upper()


def _chunk_relevant(hit: dict[str, Any], relevant_ids: set[str]) -> bool:
    return bool(_hit_relevant_ids(hit, relevant_ids))


def _hit_relevant_ids(hit: dict[str, Any], relevant_ids: set[str]) -> set[str]:
    matched: set[str] = set()
    rid = _norm(hit.get("requirement_id"))
    if rid and rid in relevant_ids:
        matched.add(rid)
    text = hit.get("text", "")
    blob = str(text or "").upper()
    matched.update(rid for rid in relevant_ids if rid in blob)
    return matched


def _first_relevant_rank(hits: list[dict[str, Any]], relevant_ids: set[str]) -> int | None:
    for idx, hit in enumerate(hits, start=1):
        if _chunk_relevant(hit, relevant_ids):
            return idx
    return None


def _prec_rec_at_k(hits: list[dict[str, Any]], relevant_ids: set[str], k: int) -> tuple[float, float]:
    top = hits[:k]
    matched_ids: set[str] = set()
    relevant_hits = 0
    for hit in top:
        hit_matches = _hit_relevant_ids(hit, relevant_ids)
        if hit_matches:
            relevant_hits += 1
            matched_ids.update(hit_matches)
    precision = relevant_hits / k if k > 0 else 0.0
    recall = len(matched_ids) / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark RSSOM semantic retrieval")
    ap.add_argument(
        "--golden",
        type=Path,
        default=REPO_ROOT / "data/eval/retrieval/rssom_golden_queries.json",
        help="JSON list of {id, query, relevant_requirement_ids}",
    )
    ap.add_argument(
        "--docx",
        type=Path,
        default=REPO_ROOT / "data/RSSOM_APCS_FIT.docx",
        help="RSSOM/FIT docx used to build the retrieval corpus",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Retrieve this many chunks")
    ap.add_argument("--ks", default="1,3,5,10", help="Comma-separated K values")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON report output")
    args = ap.parse_args()

    ks = tuple(int(x.strip()) for x in str(args.ks).split(",") if x.strip())
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    if not isinstance(golden, list):
        raise SystemExit("golden file must be a JSON array")

    parsed = parse_docx(str(args.docx))
    corpus = build_test_evidence_corpus(parsed, REPO_ROOT, requirement_ids=[])
    req_rows = rows_from_rssom_docx(args.docx)
    rag = RSSOMRAGIndex(
        corpus,
        title="RSSOM FIT retrieval benchmark corpus",
        requirement_rows=req_rows,
    )
    if not rag.ready:
        raise SystemExit("RSSOM RAG index was not created; check corpus / embedding configuration")

    per_query: list[dict[str, Any]] = []
    mrr_vals: list[float] = []

    for item in golden:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        relevant_ids = {_norm(x) for x in (item.get("relevant_requirement_ids") or []) if _norm(x)}
        if not qid or not query or not relevant_ids:
            continue

        hits_raw = rag.retrieve(query, top_k=max(args.top_k, max(ks) if ks else args.top_k), max_chars=300)
        hits = [
            {
                "text": h.text,
                "score": h.score,
                "chunk_index": h.chunk_index,
                "total_chunks": h.total_chunks,
                "requirement_id": h.requirement_id,
                "index_kind": h.index_kind,
            }
            for h in hits_raw
        ]
        rank = _first_relevant_rank(hits, relevant_ids)
        rr = (1.0 / rank) if rank is not None else 0.0
        mrr_vals.append(rr)

        metrics: dict[str, Any] = {
            "first_relevant_rank": rank,
            "mrr_contribution": round(rr, 6),
        }
        for k in ks:
            p, r = _prec_rec_at_k(hits, relevant_ids, k)
            metrics[f"hit_at_{k}"] = 1 if rank is not None and rank <= k else 0
            metrics[f"precision_at_{k}"] = round(p, 4)
            metrics[f"recall_at_{k}"] = round(r, 4)

        per_query.append(
            {
                "id": qid,
                "query": query,
                "relevant_requirement_ids": sorted(relevant_ids),
                "retrieved_hits": hits[: args.top_k],
                "metrics": metrics,
            }
        )

    summary: dict[str, Any] = {
        "golden_path": str(args.golden.resolve()),
        "docx_path": str(args.docx.resolve()),
        "queries_evaluated": len(per_query),
        "mean_reciprocal_rank": round(mean(mrr_vals), 6) if mrr_vals else 0.0,
    }
    for k in ks:
        hit_vals = [p["metrics"][f"hit_at_{k}"] for p in per_query]
        prec_vals = [p["metrics"][f"precision_at_{k}"] for p in per_query]
        rec_vals = [p["metrics"][f"recall_at_{k}"] for p in per_query]
        summary[f"mean_hit_at_{k}"] = round(mean(hit_vals), 4) if hit_vals else 0.0
        summary[f"mean_precision_at_{k}"] = round(mean(prec_vals), 4) if prec_vals else 0.0
        summary[f"mean_recall_at_{k}"] = round(mean(rec_vals), 4) if rec_vals else 0.0

    print("=== RSSOM retrieval benchmark ===")
    print(f"golden: {args.golden}")
    print(f"docx:   {args.docx}")
    print(f"MRR (mean): {summary['mean_reciprocal_rank']}")
    for k in ks:
        print(
            f"Hit@{k}: {summary[f'mean_hit_at_{k}']}  "
            f"P@{k}: {summary[f'mean_precision_at_{k}']}  "
            f"R@{k}: {summary[f'mean_recall_at_{k}']}"
        )
    print()
    for row in per_query:
        print(
            f"{row['id']}: rank={row['metrics']['first_relevant_rank']} "
            f"relevant={row['relevant_requirement_ids']} "
            f"top_score={(row['retrieved_hits'][0]['score'] if row['retrieved_hits'] else 'n/a')}"
        )

    payload = {"summary": summary, "per_query": per_query}
    if args.out:
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
