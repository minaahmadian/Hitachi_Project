from __future__ import annotations

import re
from typing import Any


def _normalize_status(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def _log_results_index(test_logs: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    raw = test_logs.get("requirement_results")
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id", "")).strip()
        res = str(item.get("result", "")).strip().upper()
        if rid and res:
            out[rid.upper()] = res
    return out


def _doc_outcome_for_requirement(corpus: str, req_id: str) -> tuple[str, str | None]:
    """
    Classify document evidence for ``req_id`` using **line-local** tokens only
    (avoids picking up a neighbour row's FAIL on the next line).

    Returns (PASS | FAIL | UNKNOWN | NOT_FOUND, short_snippet).
    """
    if not corpus.strip() or not req_id.strip():
        return "NOT_FOUND", None

    needle = req_id.strip().lower()
    matching_lines: list[str] = []
    for line in corpus.splitlines():
        if needle in line.lower():
            matching_lines.append(line.strip())

    if not matching_lines:
        return "NOT_FOUND", None

    line_fail = re.compile(r"\bfail(?:ed|ure)?\b|✗", re.IGNORECASE)
    line_pass = re.compile(r"\bpass(?:ed)?\b|✓|✔|\bok\b", re.IGNORECASE)

    saw_fail = False
    saw_pass = False
    primary = matching_lines[0]
    for ln in matching_lines:
        if line_fail.search(ln):
            saw_fail = True
        if line_pass.search(ln):
            saw_pass = True

    snippet = primary[:300]
    if saw_fail:
        return "FAIL", snippet
    if saw_pass:
        return "PASS", snippet
    return "UNKNOWN", snippet


def _append_anomaly(
    anomalies: list[dict[str, Any]],
    *,
    anomaly_type: str,
    severity: str,
    requirement_id: str | None,
    detail: str,
    evidence_snippet: str | None = None,
) -> None:
    anomalies.append(
        {
            "anomaly_id": f"a-{len(anomalies) + 1:04d}",
            "type": anomaly_type,
            "severity": severity,
            "requirement_id": requirement_id,
            "detail": detail,
            "evidence_snippet": evidence_snippet,
        }
    )


def run_traceability_match(
    *,
    requirements_records: list[dict[str, str]],
    test_evidence_corpus: str,
    test_logs: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic Phase 2: cross CSV requirement IDs with test-design corpus and structured logs.
    """
    log_by_id = _log_results_index(test_logs)
    metrics = test_logs.get("metrics") if isinstance(test_logs.get("metrics"), dict) else {}
    failed_metrics = 0
    try:
        failed_metrics = int(metrics.get("failed_tests", 0) or 0)
    except (TypeError, ValueError):
        failed_metrics = 0

    anomalies: list[dict[str, Any]] = []
    per_req: list[dict[str, Any]] = []

    verified_states = {"VERIFIED", "VALIDATED", "APPROVED", "COMPLETE"}

    for row in requirements_records:
        req_id = str(row.get("requirement_id", "")).strip()
        if not req_id:
            continue
        csv_status = _normalize_status(str(row.get("verification_status", "")))
        doc_out, snippet = _doc_outcome_for_requirement(test_evidence_corpus, req_id)
        log_res = log_by_id.get(req_id.upper())

        consistent = True
        notes: list[str] = []

        if csv_status in verified_states:
            if doc_out == "NOT_FOUND":
                consistent = False
                _append_anomaly(
                    anomalies,
                    anomaly_type="MISSING_TEST_DESIGN_EVIDENCE",
                    severity="HIGH",
                    requirement_id=req_id,
                    detail=f"Requirement {req_id} is {csv_status} in trace CSV but not found in test evidence corpus.",
                )
                notes.append("no_doc_hit")
            elif doc_out == "FAIL":
                consistent = False
                _append_anomaly(
                    anomalies,
                    anomaly_type="DOC_EVIDENCE_FAIL",
                    severity="HIGH",
                    requirement_id=req_id,
                    detail=f"Test design evidence shows FAIL for {req_id} while CSV marks it verified.",
                    evidence_snippet=snippet,
                )
                notes.append("doc_fail")
            elif doc_out == "UNKNOWN":
                _append_anomaly(
                    anomalies,
                    anomaly_type="WEAK_VERIFICATION_SIGNAL",
                    severity="MEDIUM",
                    requirement_id=req_id,
                    detail=f"Requirement {req_id} referenced in corpus but no explicit PASS/FAIL token nearby.",
                    evidence_snippet=snippet,
                )
                notes.append("unknown_doc_verdict")

        if log_res == "FAIL":
            if doc_out == "PASS":
                consistent = False
                _append_anomaly(
                    anomalies,
                    anomaly_type="LOG_FAIL_VS_DOC_PASS",
                    severity="HIGH",
                    requirement_id=req_id,
                    detail="Execution logs record FAIL while test-design excerpt shows PASS.",
                    evidence_snippet=snippet,
                )
            if csv_status in verified_states:
                consistent = False
                _append_anomaly(
                    anomalies,
                    anomaly_type="LOG_FAIL_VS_VERIFIED_REQUIREMENT",
                    severity="HIGH",
                    requirement_id=req_id,
                    detail="Structured logs report FAIL for a requirement marked Verified in the trace matrix.",
                )

        per_req.append(
            {
                "requirement_id": req_id,
                "title": row.get("title", ""),
                "csv_verification_status": row.get("verification_status", ""),
                "document_outcome": doc_out,
                "log_result": log_res,
                "consistent": consistent,
                "notes": notes,
            }
        )

    if failed_metrics > 0 and not any(
        str(a.get("type")) == "METRICS_REPORT_FAILED_TESTS" for a in anomalies
    ):
        _append_anomaly(
            anomalies,
            anomaly_type="METRICS_REPORT_FAILED_TESTS",
            severity="HIGH" if failed_metrics > 0 else "MEDIUM",
            requirement_id=None,
            detail=f"Test log metrics report failed_tests={failed_metrics} for this build.",
        )

    high = sum(1 for a in anomalies if a.get("severity") == "HIGH")
    med = sum(1 for a in anomalies if a.get("severity") == "MEDIUM")

    if high:
        overall = "RED_FLAG"
    elif med or anomalies:
        overall = "WARNING"
    else:
        overall = "PASS"

    linked = sum(1 for r in per_req if r["document_outcome"] != "NOT_FOUND")

    return {
        "mode": "deterministic_traceability",
        "status": overall,
        "summary": {
            "total_requirements": len(per_req),
            "with_document_hit": linked,
            "anomalies_count": len(anomalies),
            "high_severity_count": high,
            "medium_severity_count": med,
            "metrics_failed_tests": failed_metrics,
        },
        "requirement_results": per_req,
        "anomalies": anomalies,
    }
