from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from core.state import GraphState
from regulatory.rule_engine import EvidenceItem, RegulatoryRuleEngine


def _build_anomaly_text(state: GraphState) -> str:
    auditor_report = state.get("auditor_report", {})
    detective_report = state.get("detective_report", {})

    parts: list[str] = []
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

    evidence: list[EvidenceItem] = []
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

    rules_path = Path(__file__).resolve().parent.parent / "data" / "regulatory" / "cei_en_50128_rules.json"
    engine = RegulatoryRuleEngine(rules_path=rules_path)

    anomaly_text = _build_anomaly_text(state)
    evidence = _build_evidence(state)
    summary = engine.evaluate(anomaly_text=anomaly_text, evidence=evidence, top_k_rules=30)

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
        "summary_text": (
            f"Deterministic rule-check result: checked={summary.total_rules_checked}, "
            f"pass={summary.passed}, fail={summary.failed}, warning={summary.warning}, "
            f"derogation_needed={summary.derogation_needed}"
        ),
    }

    return {"regulatory_report": report}
