"""
Priority-aware blob compaction for LLM prompts.

Why this module exists
----------------------
Every LLM call in the pipeline has a hard character budget (Groq free-tier
TPM limits, per-request size, rate limiting).  The naïve pattern

    json.dumps(report, ensure_ascii=False)[:max_chars]

is *unsafe* for any report that carries safety-critical evidence because it
silently truncates whatever happens to be at the end of the JSON.  In this
project that repeatedly included ``verdict_per_anomaly`` entries with
``RED_FLAG``, HIGH-severity anomalies, and the ``inputs_digest`` gate — i.e.
exactly the fields that must always be visible to the decision-making LLM.

This module provides three complementary primitives that, together, guarantee
that negative evidence cannot be hidden by truncation:

1. :func:`anomaly_digest`  — produce a compact, one-line digest for each
   anomaly so dozens of them fit in a few hundred characters.
2. :func:`verdict_digest`  — same for per-anomaly verdicts.
3. :func:`compact_report`  — serialise a report with two guarantees:
      (a) a caller-specified ``must_keep`` set of top-level fields is ALWAYS
          present in full;
      (b) remaining fields are progressively dropped (lowest priority first)
          until the JSON fits in ``max_chars``.
   If even the ``must_keep`` fields exceed the budget, the function appends a
   `[truncated]` marker so downstream prompts can treat it as a safety signal.

All three are deterministic, fast, and unit-testable.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

TRUNCATION_MARKER = "[truncated]"
"""String appended to any compacted blob that had to drop content. Downstream
prompts should treat its presence as a safety signal."""


SEVERITY_ORDER: dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
VERDICT_ORDER: dict[str, int] = {
    "RED_FLAG": 0,
    "REVIEW": 1,
    "JUSTIFICATION_SIGNALS": 2,
    "TRACKED": 3,
}
_UNKNOWN_SEV_RANK = 3
_UNKNOWN_VER_RANK = 4


# ─────────────────────────────────────────────────────────────────────────────
# Digests
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_digest(anomaly: dict[str, Any]) -> str:
    """
    Compact one-line digest for one anomaly.

    Example:  "C6-APCS-2 | HIGH | DOC_EVIDENCE_FAIL | Requirement C6-APCS-2 …"
    """
    rid = str(anomaly.get("requirement_id") or "").strip() or "-"
    sev = str(anomaly.get("severity") or "").strip().upper() or "UNKNOWN"
    typ = str(anomaly.get("type") or "").strip() or "UNKNOWN"
    detail = str(anomaly.get("detail") or "").strip()
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return f"{rid} | {sev} | {typ} | {detail}".rstrip(" |")


def verdict_digest(verdict: dict[str, Any]) -> str:
    """
    Compact one-line digest for one per-anomaly verdict.

    Example:  "C6-APCS-2 | HIGH | RED_FLAG | evidence_source=rssom_rag_fallback"
    """
    rid = str(verdict.get("requirement_id") or "").strip() or "-"
    sev = str(verdict.get("matcher_severity") or "").strip().upper() or "UNKNOWN"
    ver = str(verdict.get("verdict") or "").strip().upper() or "UNKNOWN"
    reason = str(verdict.get("reason") or verdict.get("evidence_source") or "").strip()
    if len(reason) > 140:
        reason = reason[:137] + "..."
    return f"{rid} | {sev} | {ver} | {reason}".rstrip(" |")


def severity_verdict_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    """Sort key: HIGH & RED_FLAG first, UNKNOWN last."""
    sev_raw = str(item.get("severity") or item.get("matcher_severity") or "").upper()
    ver_raw = str(item.get("verdict") or "").upper()
    sev = SEVERITY_ORDER.get(sev_raw, _UNKNOWN_SEV_RANK)
    ver = VERDICT_ORDER.get(ver_raw, _UNKNOWN_VER_RANK)
    return (sev, ver)


def worst_verdict_in(verdicts: Iterable[dict[str, Any]]) -> str | None:
    """Return the worst verdict token across a list of verdict entries."""
    worst = _UNKNOWN_VER_RANK
    worst_label: str | None = None
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        label = str(v.get("verdict") or "").upper()
        rank = VERDICT_ORDER.get(label, _UNKNOWN_VER_RANK)
        if rank < worst:
            worst = rank
            worst_label = label
    return worst_label


def has_high_red_flag(verdicts: Iterable[dict[str, Any]]) -> bool:
    """True iff any verdict is RED_FLAG on a HIGH-severity anomaly."""
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        sev = str(v.get("matcher_severity") or "").upper()
        ver = str(v.get("verdict") or "").upper()
        if sev == "HIGH" and ver == "RED_FLAG":
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Progressive dropper
# ─────────────────────────────────────────────────────────────────────────────

def compact_report(
    report: dict[str, Any] | None,
    *,
    max_chars: int,
    must_keep: Sequence[str],
    drop_order: Sequence[str] = (),
    extras: dict[str, Any] | None = None,
) -> str:
    """
    Serialise ``report`` into JSON that fits in ``max_chars``, with two
    guarantees:

    1. Every field listed in ``must_keep`` is *always* serialised, in full,
       even if that means the final blob exceeds ``max_chars`` (we prefer
       exceeding the soft budget over hiding safety-critical evidence).
       When the budget is exceeded, the blob is suffixed with
       ``"\\n[truncated]"`` so downstream prompts can detect it.
    2. All other fields start included and are progressively removed in the
       order given by ``drop_order`` (leftmost = dropped first) until the
       blob fits.  Fields not in ``must_keep`` or ``drop_order`` are dropped
       last, in insertion order.

    ``extras`` lets callers inject derived fields (e.g. anomaly digests,
    truncation flags) without modifying the source report dict.

    Returns the serialised string, never ``None``.
    """
    if not isinstance(report, dict) or not report:
        # Nothing to compact — just empty-object JSON.
        base = json.dumps(report or {}, ensure_ascii=False)
        return base if len(base) <= max_chars else base[:max_chars] + "\n" + TRUNCATION_MARKER

    must = list(must_keep)
    drop = list(drop_order)
    optional = [k for k in report if k not in must and k not in drop]
    # Final drop order: drop_order first, then optional-keys in insertion order.
    # We pop from the end of the sacrifice list, so reverse it.
    sacrifice: list[str] = list(reversed(drop + optional))

    extras_dict = dict(extras or {})

    def _build(current_report: dict[str, Any]) -> str:
        combined: dict[str, Any] = {}
        for k in must:
            if k in current_report:
                combined[k] = current_report[k]
        for k, v in current_report.items():
            if k in combined:
                continue
            combined[k] = v
        combined.update(extras_dict)
        return json.dumps(combined, ensure_ascii=False)

    working = dict(report)
    serialized = _build(working)
    truncated_fields: list[str] = []

    # Progressive drop: remove non-must-keep fields from the end until we fit.
    while len(serialized) > max_chars and sacrifice:
        key = sacrifice.pop(0)
        if key in must:
            # Should never happen given our filtering, but guard anyway.
            continue
        if key in working:
            working.pop(key, None)
            truncated_fields.append(key)
            serialized = _build(working)

    if truncated_fields:
        # Add provenance so prompts can reason about what was dropped.
        working_with_marker = dict(working)
        working_with_marker["_compaction"] = {
            "dropped_fields": truncated_fields,
            "marker": TRUNCATION_MARKER,
        }
        candidate = json.dumps({**working_with_marker, **extras_dict}, ensure_ascii=False)
        # If adding the marker blows the budget, keep the lighter version.
        if len(candidate) <= max_chars:
            serialized = candidate

    # Final fallback: if must_keep alone still overflows, we accept that but
    # make the overflow explicit so the LLM (and the audit log) know.
    if len(serialized) > max_chars:
        serialized = serialized + "\n" + TRUNCATION_MARKER

    return serialized
