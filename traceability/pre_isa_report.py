from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.state import GraphState

# Shown in pre-ISA / VDD so reviewers see how RSSOM corpus, matcher, and emails connect.
EVIDENCE_CHAIN_TEXT = (
    "Evidence chain: Test-design text is parsed from the FIT Word document (RSSOM_APCS_FIT.docx) "
    "into the traceability corpus. The matcher compares each requirement ID from the trace CSV "
    "against that corpus. Email threads and authorizations are then reviewed in the same "
    "requirement context—derogation uses deterministic windows around matcher-linked requirement "
    "IDs; the communications detective triages emails using matcher and derogation summaries together."
)


def _derogation_by_anomaly_id(derogation_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in derogation_report.get("per_anomaly") or []:
        if not isinstance(row, dict):
            continue
        aid = str(row.get("anomaly_id", "")).strip()
        if aid:
            out[aid] = row
    return out


def _verdict_for_anomaly(*, severity: str, justification_score: int, signals: list[Any]) -> str:
    sev = str(severity).upper()
    if sev != "HIGH":
        return "TRACKED"

    has_strong = any(
        isinstance(s, dict) and str(s.get("strength", "")).lower() == "strong" for s in signals
    )
    score = int(justification_score)
    if score >= 4 and has_strong:
        return "JUSTIFICATION_SIGNALS"
    if score >= 2 or has_strong:
        return "REVIEW"
    return "RED_FLAG"


def _rationale_for_verdict(verdict: str, anomaly_type: str | None, score: int) -> str:
    if verdict == "JUSTIFICATION_SIGNALS":
        return (
            f"High-severity anomaly ({anomaly_type or 'unknown type'}) co-locates with strong "
            f"governance / approval language in communications or authorizations (score={score}). "
            "Human ISA must still confirm scope and validity."
        )
    if verdict == "REVIEW":
        return (
            f"High-severity anomaly ({anomaly_type or 'unknown type'}) has partial justification signals "
            f"(score={score}); independent review recommended before release."
        )
    if verdict == "RED_FLAG":
        return (
            f"High-severity anomaly ({anomaly_type or 'unknown type'}) lacks adequate documented "
            f"justification in scanned emails and authorizations (score={score})."
        )
    return "Non-high severity item tracked for completeness in the pre-ISA bundle."


def _build_citations(
    regulatory_report: dict[str, Any],
    derogation_report: dict[str, Any],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    retrieval = regulatory_report.get("retrieval") if isinstance(regulatory_report.get("retrieval"), dict) else {}
    hits = retrieval.get("hits") if isinstance(retrieval.get("hits"), list) else []
    for item in hits[:8]:
        if not isinstance(item, dict):
            continue
        citations.append(
            {
                "kind": "regulatory_clause",
                "clause_id": item.get("clause_id"),
                "title": item.get("title"),
                "score": item.get("score"),
                "page_start": item.get("page_start"),
            }
        )

    for item in (derogation_report.get("hits") or [])[:6]:
        if not isinstance(item, dict):
            continue
        if str(item.get("strength", "")).lower() not in {"strong", "medium"}:
            continue
        citations.append(
            {
                "kind": "derogation_language",
                "pattern_id": item.get("pattern_id"),
                "strength": item.get("strength"),
                "source": item.get("source"),
                "snippet": str(item.get("snippet", ""))[:240],
                "requirement_id": item.get("requirement_id"),
            }
        )

    return citations


def _compute_overall(
    *,
    matcher_status: str,
    regulatory_status: str,
    detective_status: str,
    verdict_per_anomaly: list[dict[str, Any]],
) -> str:
    ms = matcher_status.upper()
    rs = regulatory_status.upper()
    ds = detective_status.upper()

    if ms == "RED_FLAG" or rs == "RED_FLAG":
        return "RED_FLAG"
    if ds == "SUSPICIOUS":
        return "REVIEW_REQUIRED"
    if ms == "WARNING" or rs == "WARNING":
        return "REVIEW_REQUIRED"

    high_red = any(
        str(v.get("matcher_severity", "")).upper() == "HIGH" and str(v.get("verdict", "")).upper() == "RED_FLAG"
        for v in verdict_per_anomaly
        if isinstance(v, dict)
    )
    high_review = any(
        str(v.get("matcher_severity", "")).upper() == "HIGH" and str(v.get("verdict", "")).upper() == "REVIEW"
        for v in verdict_per_anomaly
        if isinstance(v, dict)
    )
    if high_red:
        return "REVIEW_REQUIRED"
    if high_review:
        return "REVIEW_REQUIRED"
    return "CLEAR"


def _summary_for_vdd(
    overall: str,
    matcher_status: str,
    regulatory_status: str,
    detective_status: str,
    derogation_overall: str,
    verdict_per_anomaly: list[dict[str, Any]],
) -> str:
    n_high = sum(
        1 for v in verdict_per_anomaly if isinstance(v, dict) and str(v.get("matcher_severity", "")).upper() == "HIGH"
    )
    n_just = sum(
        1
        for v in verdict_per_anomaly
        if isinstance(v, dict) and str(v.get("verdict", "")).upper() == "JUSTIFICATION_SIGNALS"
    )
    parts = [
        EVIDENCE_CHAIN_TEXT,
        f"Pre-ISA consolidated assessment: overall={overall}.",
        f"Traceability matcher status={matcher_status}; regulatory gate status={regulatory_status}; "
        f"communications triage status={detective_status}; derogation language scan={derogation_overall}.",
    ]
    if n_high:
        parts.append(f"High-severity traceability anomalies reviewed: {n_high}; with formal justification signals: {n_just}.")
    return " ".join(parts)[:2400]


def build_pre_isa_report(state: GraphState) -> dict[str, Any]:
    """
    Step D: single JSON bundle for lead assessor, audit export, and future VDD templating (docxtpl).
    """
    matcher = state.get("matcher_report") or {}
    derogation = state.get("derogation_report") or {}
    regulatory = state.get("regulatory_report") or {}
    detective = state.get("detective_report") or {}
    auditor = state.get("auditor_report") or {}

    if not isinstance(matcher, dict):
        matcher = {}
    if not isinstance(derogation, dict):
        derogation = {}
    if not isinstance(regulatory, dict):
        regulatory = {}
    if not isinstance(detective, dict):
        detective = {}
    if not isinstance(auditor, dict):
        auditor = {}

    by_aid = _derogation_by_anomaly_id(derogation)

    verdict_rows: list[dict[str, Any]] = []
    for item in matcher.get("anomalies") or []:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("anomaly_id", "")).strip()
        sev = str(item.get("severity", "")).strip()
        aty = str(item.get("type", "")).strip()
        rid = item.get("requirement_id")
        rid_s = str(rid).strip() if rid is not None else ""

        drow = by_aid.get(aid, {})
        score = int(drow.get("justification_score", 0) or 0) if isinstance(drow, dict) else 0
        signals = drow.get("signals") if isinstance(drow.get("signals"), list) else []

        verdict = _verdict_for_anomaly(severity=sev, justification_score=score, signals=signals)
        verdict_rows.append(
            {
                "anomaly_id": aid or None,
                "requirement_id": rid_s or None,
                "matcher_severity": sev or None,
                "matcher_type": aty or None,
                "derogation_justification_score": score,
                "verdict": verdict,
                "rationale": _rationale_for_verdict(verdict, aty or None, score),
            }
        )

    matcher_status = str(matcher.get("status", "UNKNOWN")).strip() or "UNKNOWN"
    regulatory_status = str(regulatory.get("status", "UNKNOWN")).strip() or "UNKNOWN"
    detective_status = str(detective.get("status", "UNKNOWN")).strip() or "UNKNOWN"
    derogation_overall = str(derogation.get("overall", "NO_SIGNALS")).strip() or "NO_SIGNALS"

    overall = _compute_overall(
        matcher_status=matcher_status,
        regulatory_status=regulatory_status,
        detective_status=detective_status,
        verdict_per_anomaly=verdict_rows,
    )

    envelope = matcher.get("anomaly_envelope") if isinstance(matcher.get("anomaly_envelope"), dict) else {}
    fp = str(envelope.get("phase2_fingerprint", "")).strip()

    citations = _build_citations(regulatory, derogation)

    summary_for_vdd = _summary_for_vdd(
        overall,
        matcher_status,
        regulatory_status,
        detective_status,
        derogation_overall,
        verdict_rows,
    )

    return {
        "schema_version": "pre_isa_report.v1",
        "mode": "deterministic_compilation",
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "release_readiness": overall,
        "evidence_chain_text": EVIDENCE_CHAIN_TEXT,
        "summary_for_vdd": summary_for_vdd,
        "fingerprints": {
            "phase2_traceability": fp or None,
        },
        "inputs_digest": {
            "matcher_status": matcher_status,
            "regulatory_status": regulatory_status,
            "regulatory_derogation_needed": int(regulatory.get("derogation_needed", 0) or 0),
            "detective_status": detective_status,
            "detective_severity": str(detective.get("severity", "")).strip() or None,
            "auditor_overall_assessment": str(auditor.get("overall_assessment", "")).strip() or None,
            "derogation_scan_overall": derogation_overall,
        },
        "verdict_per_anomaly": verdict_rows,
        "citations": citations,
    }
