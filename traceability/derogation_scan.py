from __future__ import annotations

import re
from typing import Any


def _patterns(
    specs: list[tuple[str, str]],
    strength: str,
) -> list[tuple[re.Pattern[str], str, str]]:
    return [
        (re.compile(pattern, re.IGNORECASE | re.MULTILINE), pattern_id, strength)
        for pattern, pattern_id in specs
    ]


_STRONG = _patterns(
    [
        (r"\bapproved\s+derogation\b", "approved_derogation"),
        (r"\bformal\s+deviation\s+approved\b", "formal_deviation_approved"),
        (r"\b(deviation|exception)\s+approved\b", "deviation_or_exception_approved"),
        (r"\bwaiver\s+(granted|approved)\b", "waiver_granted_or_approved"),
        (r"\bccb\b.{0,80}\b(approv|agreed|accepted)\b", "ccb_approval_language"),
        (r"\bconfiguration\s+control\b.{0,120}\b(approv|accept)\b", "configuration_control_approval"),
        (r"\b(sign[- ]?off|signoff)\b.{0,60}\b(recorded|complete|obtained)\b", "signoff_recorded"),
        (r"\bofficially\s+accepted\s+risk\b", "officially_accepted_risk"),
    ],
    "strong",
)

_MEDIUM = _patterns(
    [
        (r"\bderogation\b", "derogation_mentioned"),
        (r"\b(waiver|deviation|exception)\b.{0,40}\b(request|pending)\b", "waiver_or_deviation_request"),
        (r"\baccepted\s+risk\b", "accepted_risk_phrase"),
        (r"\bqa\s+lead\b.{0,40}\b(sign|approv|accept)\b", "qa_lead_signoff_language"),
    ],
    "medium",
)

_RISK = _patterns(
    [
        (r"\b(workaround|bypass)\b", "workaround_or_bypass"),
        (r"\b(skip|disabled)\b.{0,40}\b(test|hil|check)\b", "skip_or_disable_tests"),
        (r"\bmanual(ly)?\b.{0,40}\b(kubectl|apply)\b", "manual_kubectl_apply"),
        (r"\bjust\s+to\s+pass\b", "just_to_pass_language"),
    ],
    "risk",
)


def _window_has_strong(window: str) -> bool:
    for rx, _pid, _st in _STRONG:
        if rx.search(window):
            return True
    return False


def _collect_windows(
    emails: str,
    auth: str,
    needle: str | None,
    *,
    radius: int = 280,
    global_cap: int = 3500,
) -> list[dict[str, Any]]:
    """Slices around ``needle`` in each corpus; if needle falsy, take bounded head of each blob."""
    out: list[dict[str, Any]] = []
    for source, blob in (("email", emails), ("authorization", auth)):
        raw = blob if isinstance(blob, str) else str(blob)
        if not raw.strip():
            continue
        if not needle or not str(needle).strip():
            out.append({"source": source, "text": raw.strip()[:global_cap]})
            continue
        lowered = raw.lower()
        n = str(needle).strip().lower()
        start = 0
        while True:
            idx = lowered.find(n, start)
            if idx < 0:
                break
            lo = max(0, idx - radius)
            hi = min(len(raw), idx + radius)
            out.append({"source": source, "text": raw[lo:hi].replace("\n", " ").strip()})
            start = idx + max(1, len(n))
    return out


def _scan_window(window: str, source: str, anomaly_id: str, requirement_id: str | None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if not window.strip():
        return hits

    for patterns, base_score in ((_STRONG, 3), (_MEDIUM, 1), (_RISK, -2)):
        for rx, pid, strength in patterns:
            m = rx.search(window)
            if not m:
                continue
            span = m.span()
            snip = window[max(0, span[0] - 40) : min(len(window), span[1] + 80)].strip()
            hits.append(
                {
                    "pattern_id": pid,
                    "strength": strength,
                    "base_score": base_score,
                    "source": source,
                    "snippet": snip[:220],
                    "parent_window": window[:800],
                    "anomaly_id": anomaly_id,
                    "requirement_id": requirement_id,
                }
            )
    return hits


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for h in hits:
        key = (h.get("pattern_id"), h.get("source"), str(h.get("snippet", ""))[:100])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _score_hits(hits: list[dict[str, Any]]) -> int:
    score = 0
    for h in hits:
        base = int(h.get("base_score", 0))
        if h.get("strength") == "risk":
            parent = str(h.get("parent_window") or h.get("snippet", ""))
            if _window_has_strong(parent):
                continue
        score += base
    return score


def scan_derogation_context(
    *,
    matcher_report: dict[str, Any],
    email_threads: str,
    authorization_text: str,
) -> dict[str, Any]:
    """
    Deterministic scan of emails + authorizations for governance / derogation language
    near matcher anomalies (especially HIGH severity and linked requirement IDs).
    """
    emails = email_threads if isinstance(email_threads, str) else str(email_threads)
    auth = authorization_text if isinstance(authorization_text, str) else str(authorization_text)

    envelope = matcher_report.get("anomaly_envelope") if isinstance(matcher_report.get("anomaly_envelope"), dict) else {}
    fp_ref = str(envelope.get("phase2_fingerprint", "")).strip()

    anomalies = matcher_report.get("anomalies")
    if not isinstance(anomalies, list):
        anomalies = []

    aggregate_hits: list[dict[str, Any]] = []
    per_anomaly: list[dict[str, Any]] = []
    high_scores: list[int] = []

    for item in anomalies:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("anomaly_id", "")).strip()
        sev = str(item.get("severity", "")).upper()
        rid = item.get("requirement_id")
        rid_s = str(rid).strip() if rid is not None else ""

        needle: str | None = rid_s if rid_s else None
        if not needle:
            detail = str(item.get("detail", "")).strip()
            if detail:
                needle = detail.split()[0][:48]

        windows = _collect_windows(emails, auth, needle)
        if not windows and not rid_s and not needle:
            windows = _collect_windows(emails, auth, None)

        hits: list[dict[str, Any]] = []
        for w in windows:
            text = str(w.get("text", ""))
            src = str(w.get("source", "email"))
            hits.extend(_scan_window(text, src, aid, rid_s or None))

        hits = _dedupe_hits(hits)
        score = _score_hits(hits)
        for h in hits:
            if isinstance(h, dict):
                h.pop("parent_window", None)

        if sev == "HIGH":
            high_scores.append(score)

        per_anomaly.append(
            {
                "anomaly_id": aid or None,
                "severity": sev or None,
                "requirement_id": rid_s or None,
                "anomaly_type": str(item.get("type", "")).strip() or None,
                "justification_score": score,
                "signals": hits,
            }
        )
        aggregate_hits.extend(hits)

    max_high = max(high_scores) if high_scores else 0
    any_hit = bool(aggregate_hits)

    if max_high >= 4:
        overall = "STRONG_SIGNALS"
    elif max_high >= 2 or any_hit:
        overall = "WEAK_SIGNALS"
    else:
        overall = "NO_SIGNALS"

    summary = (
        f"Derogation scan: overall={overall}, matcher_fingerprint={fp_ref or 'n/a'}, "
        f"pattern_hits={len(aggregate_hits)}, high_severity_best_score={max_high}."
    )

    return {
        "mode": "deterministic_derogation_scan",
        "overall": overall,
        "matcher_phase2_fingerprint": fp_ref or None,
        "summary_text": summary,
        "hits": aggregate_hits[:40],
        "per_anomaly": per_anomaly,
    }
