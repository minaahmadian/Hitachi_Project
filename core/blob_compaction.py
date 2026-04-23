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

from core.enums import Severity, Verdict


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

TRUNCATION_MARKER = "[truncated]"
"""String appended to any compacted blob that had to drop content. Downstream
prompts should treat its presence as a safety signal."""


def parse_compacted_blob(blob: str) -> tuple[dict[str, Any], bool]:
    """Safely parse a blob produced by :func:`compact_report`.

    A compacted blob has one of two shapes:

    * Pure JSON, when the compactor was able to fit within the budget.
    * ``<valid JSON>\\n[truncated]`` when even the ``must_keep`` fields
      exceeded the budget — this is the documented overflow contract, not
      a bug. In that case we still want downstream consumers to recover
      the JSON portion without crashing on ``json.loads``.

    Returns a tuple ``(parsed_dict, was_truncated)``.  If the JSON portion
    cannot be parsed, returns ``({}, True)`` rather than raising — the
    ``True`` tells the caller to treat the payload as untrusted.
    """
    if not isinstance(blob, str) or not blob:
        return {}, True
    truncated = blob.endswith(TRUNCATION_MARKER)
    json_part = blob[: -len(TRUNCATION_MARKER)].rstrip() if truncated else blob
    try:
        parsed = json.loads(json_part)
    except (json.JSONDecodeError, ValueError):
        return {}, True
    if not isinstance(parsed, dict):
        return {}, truncated
    return parsed, truncated


# Kept for backwards compatibility. New code should import Severity / Verdict
# from ``core.enums`` directly. These dicts are derived from the enums so the
# two sources can never disagree.
SEVERITY_ORDER: dict[str, int] = {
    Severity.HIGH.value: Severity.HIGH.order,
    Severity.MEDIUM.value: Severity.MEDIUM.order,
    Severity.LOW.value: Severity.LOW.order,
}
VERDICT_ORDER: dict[str, int] = {
    Verdict.RED_FLAG.value: Verdict.RED_FLAG.order,
    Verdict.REVIEW.value: Verdict.REVIEW.order,
    Verdict.JUSTIFICATION_SIGNALS.value: Verdict.JUSTIFICATION_SIGNALS.order,
    Verdict.TRACKED.value: Verdict.TRACKED.order,
}


def _read_severity(item: dict[str, Any]) -> Severity:
    """Read severity from an anomaly/verdict dict, accepting BOTH ``severity``
    and ``matcher_severity`` keys.

    This is the single source of truth for the safety gate: if either key
    encodes HIGH/MEDIUM/LOW, we honour it. A downstream renaming that leaves
    one key behind cannot silently disable the gate.
    """
    raw = item.get("severity")
    if raw in (None, ""):
        raw = item.get("matcher_severity")
    return Severity.normalize(raw)


def _read_verdict(item: dict[str, Any]) -> Verdict:
    return Verdict.normalize(item.get("verdict"))


# ─────────────────────────────────────────────────────────────────────────────
# Digests
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_digest(anomaly: dict[str, Any]) -> str:
    """
    Compact one-line digest for one anomaly.

    Example:  "C6-APCS-2 | HIGH | DOC_EVIDENCE_FAIL | Requirement C6-APCS-2 …"
    """
    rid = str(anomaly.get("requirement_id") or "").strip() or "-"
    sev = _read_severity(anomaly).value
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
    sev = _read_severity(verdict).value
    ver = _read_verdict(verdict).value
    reason = str(verdict.get("reason") or verdict.get("evidence_source") or "").strip()
    if len(reason) > 140:
        reason = reason[:137] + "..."
    return f"{rid} | {sev} | {ver} | {reason}".rstrip(" |")


def severity_verdict_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    """Sort key: HIGH & RED_FLAG first, UNKNOWN last."""
    return (_read_severity(item).order, _read_verdict(item).order)


def worst_verdict_in(verdicts: Iterable[dict[str, Any]]) -> str | None:
    """Return the worst verdict value across a list of verdict entries.

    Returns ``None`` if the list is empty or contains only UNKNOWN verdicts.
    """
    best_rank = Verdict.UNKNOWN.order
    worst_label: str | None = None
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        verdict = _read_verdict(v)
        if verdict is Verdict.UNKNOWN:
            continue
        if verdict.order < best_rank:
            best_rank = verdict.order
            worst_label = verdict.value
    return worst_label


def has_high_red_flag(verdicts: Iterable[dict[str, Any]]) -> bool:
    """True iff any verdict is RED_FLAG on a HIGH-severity anomaly.

    Reads severity from either ``severity`` or ``matcher_severity`` so the
    safety gate fires regardless of which schema variant the upstream node
    used. This is a deliberate choice: the gate must err on the side of
    detection, never silent miss.
    """
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        if _read_severity(v) is Severity.HIGH and _read_verdict(v) is Verdict.RED_FLAG:
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
