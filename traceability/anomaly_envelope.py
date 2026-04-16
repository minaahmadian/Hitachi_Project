from __future__ import annotations

import hashlib
import re
from typing import Any


def build_anomaly_envelope(matcher_report: dict[str, Any]) -> dict[str, Any]:
    """
    Phase 3 handoff (Step A): deterministic summary of Phase 2 output for RAG and agents.

    Fields:
    - ``primary_requirement_ids``: requirement IDs touched by HIGH-severity anomalies (deduped, stable order).
    - ``human_summary``: 1–3 short sentences derived only from matcher facts (no LLM).
    - ``retrieval_query``: compact keyword line for regulatory / clause search (Step B will consume this).
    - ``phase2_fingerprint``: SHA-256 over canonicalized anomaly rows for audit reproducibility.
    """
    anomalies = matcher_report.get("anomalies")
    if not isinstance(anomalies, list):
        anomalies = []

    primary_ids: list[str] = []
    for item in anomalies:
        if not isinstance(item, dict):
            continue
        if str(item.get("severity", "")).upper() != "HIGH":
            continue
        rid = item.get("requirement_id")
        if rid is None or not str(rid).strip():
            continue
        token = str(rid).strip()
        if token.upper() not in {x.upper() for x in primary_ids}:
            primary_ids.append(token)

    type_keys: list[str] = []
    for item in anomalies:
        if isinstance(item, dict) and item.get("type"):
            type_keys.append(str(item["type"]))

    type_counts: dict[str, int] = {}
    for t in type_keys:
        type_counts[t] = type_counts.get(t, 0) + 1

    status = str(matcher_report.get("status", "UNKNOWN")).strip() or "UNKNOWN"

    sentences: list[str] = []
    if primary_ids:
        sentences.append(
            "High-severity traceability anomalies involve requirement(s): "
            + ", ".join(primary_ids)
            + "."
        )
    if type_counts:
        breakdown = ", ".join(f"{k} ({v})" for k, v in sorted(type_counts.items()))
        sentences.append(f"Recorded anomaly classes: {breakdown}.")
    sentences.append(f"Overall traceability matcher status: {status}.")

    human_summary = " ".join(sentences).strip()
    if not anomalies:
        human_summary = "No traceability anomalies were recorded for this run."

    retrieval_query = _build_retrieval_query(primary_ids, anomalies, matcher_report)
    phase2_fingerprint = _fingerprint_anomalies(anomalies)

    return {
        "primary_requirement_ids": primary_ids,
        "human_summary": human_summary,
        "retrieval_query": retrieval_query,
        "phase2_fingerprint": phase2_fingerprint,
        "source": "traceability_matcher",
        "schema_version": "anomaly_envelope.v1",
    }


def _build_retrieval_query(
    primary_ids: list[str],
    anomalies: list[Any],
    matcher_report: dict[str, Any],
) -> str:
    """Keyword line for Phase 0 RAG (deterministic, length-capped)."""
    segments: list[str] = [
        "CEI EN 50128",
        "railway software safety",
        "traceability",
        "verification",
        "evidence",
        "derogation",
        "deviation",
        "failed test",
    ]
    segments.extend(primary_ids)
    for item in anomalies:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "")).strip()
        if t:
            segments.append(t.replace("_", " "))
        detail = str(item.get("detail", "")).strip()
        if detail:
            for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", detail)[:8]:
                segments.append(word)

    summary = matcher_report.get("summary")
    if isinstance(summary, dict):
        mft = summary.get("metrics_failed_tests")
        if mft is not None:
            segments.append(f"failed_tests_{mft}")

    seen: set[str] = set()
    ordered: list[str] = []
    for s in segments:
        key = s.lower()
        if key in seen or not s:
            continue
        seen.add(key)
        ordered.append(s)

    text = " ".join(ordered)
    return text[:480].strip()


def _fingerprint_anomalies(anomalies: list[Any]) -> str:
    lines: list[str] = []
    normalized: list[tuple[str, str, str, str, str]] = []
    for item in anomalies:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("anomaly_id", "")).strip()
        aty = str(item.get("type", "")).strip()
        sev = str(item.get("severity", "")).strip().upper()
        rid = str(item.get("requirement_id", "")).strip()
        det = str(item.get("detail", "")).strip()
        normalized.append((aid, aty, sev, rid, det))

    for aid, aty, sev, rid, det in sorted(normalized):
        lines.append(f"{aid}|{aty}|{sev}|{rid}|{det}")

    raw = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
