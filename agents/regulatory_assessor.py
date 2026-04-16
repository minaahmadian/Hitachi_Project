from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from core.state import GraphState
from regulatory.clause_retrieval import retrieve_regulatory_clauses
from regulatory.rule_engine import EvidenceItem, RegulatoryRuleEngine


def _build_anomaly_text(state: GraphState) -> str:
    auditor_report = state.get("auditor_report", {})
    detective_report = state.get("detective_report", {})
    matcher_report = state.get("matcher_report") or {}

    parts: list[str] = []
    parts.append(str(matcher_report.get("status", "")))
    for row in matcher_report.get("anomalies") or []:
        if isinstance(row, dict):
            parts.append(str(row.get("type", "")))
            parts.append(str(row.get("detail", "")))
    parts.append(str(auditor_report.get("overall_assessment", "")))
    parts.extend(str(item) for item in auditor_report.get("risks", []) if str(item).strip())
    parts.append(str(detective_report.get("status", "")))
    parts.append(str(detective_report.get("reason", "")))
    parts.extend(str(item) for item in detective_report.get("red_flags", []) if str(item).strip())
    text = " ".join(part for part in parts if part.strip()).strip()
    return text or "safety release compliance verification evidence traceability"


def _build_evidence(state: GraphState) -> list[EvidenceItem]:
    auditor_report = state.get("auditor_report", {})
    detective_report = state.get("detective_report", {})
    matcher_report = state.get("matcher_report") or {}

    evidence: list[EvidenceItem] = []
    for idx, row in enumerate(matcher_report.get("anomalies") or [], start=1):
        if not isinstance(row, dict):
            continue
        blob = " ".join(
            str(row.get(k, ""))
            for k in ("type", "severity", "requirement_id", "detail", "evidence_snippet")
            if row.get(k)
        ).strip()
        if blob:
            evidence.append(
                EvidenceItem(
                    evidence_id=f"matcher-anomaly-{idx}",
                    text=blob,
                    source_type="traceability_matcher",
                )
            )
    for idx, req in enumerate(auditor_report.get("requirements_found", []), start=1):
        req_text = str(req).strip()
        if req_text:
            evidence.append(EvidenceItem(evidence_id=f"auditor-req-{idx}", text=req_text, source_type="auditor_report"))

    for idx, risk in enumerate(auditor_report.get("risks", []), start=1):
        risk_text = str(risk).strip()
        if risk_text:
            evidence.append(EvidenceItem(evidence_id=f"auditor-risk-{idx}", text=risk_text, source_type="auditor_report"))

    reason = str(detective_report.get("reason", "")).strip()
    if reason:
        evidence.append(EvidenceItem(evidence_id="detective-reason-1", text=reason, source_type="detective_report"))

    for idx, flag in enumerate(detective_report.get("red_flags", []), start=1):
        flag_text = str(flag).strip()
        if flag_text:
            evidence.append(EvidenceItem(evidence_id=f"detective-flag-{idx}", text=flag_text, source_type="detective_report"))

    return evidence


def regulatory_assessor_node(state: GraphState):
    print("Regulatory Assessor: Running deterministic CEI EN 50128 rule checks...")

    repo_root = Path(__file__).resolve().parent.parent
    rules_path = repo_root / "data" / "regulatory" / "cei_en_50128_rules.json"
    engine = RegulatoryRuleEngine(rules_path=rules_path)

    anomaly_text = _build_anomaly_text(state)
    evidence = _build_evidence(state)
    summary = engine.evaluate(anomaly_text=anomaly_text, evidence=evidence, top_k_rules=30)

    top_k = int(os.getenv("REGULATORY_RAG_TOP_K", "5") or "5")
    retrieval = retrieve_regulatory_clauses(
        anomaly_text,
        repo_root=repo_root,
        top_k=max(1, min(top_k, 20)),
    )

    findings = [asdict(item) for item in summary.findings]
    high_failures = [f for f in findings if f["status"] == "FAIL" and f["severity"] == "HIGH"]

    report = {
        "mode": "deterministic_rule_engine",
        "status": "RED_FLAG" if high_failures else ("WARNING" if summary.warning > 0 else "PASS"),
        "rules_checked": summary.total_rules_checked,
        "passed": summary.passed,
        "failed": summary.failed,
        "warning": summary.warning,
        "derogation_needed": summary.derogation_needed,
        "top_findings": findings[:10],
        "retrieval": retrieval,
        "summary_text": (
            f"Deterministic rule-check result: checked={summary.total_rules_checked}, "
            f"pass={summary.passed}, fail={summary.failed}, warning={summary.warning}, "
            f"derogation_needed={summary.derogation_needed}"
        ),
    }

    return {"regulatory_report": report}
