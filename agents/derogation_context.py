from __future__ import annotations

from core.state import GraphState
from traceability.derogation_scan import scan_derogation_context


def derogation_context_node(state: GraphState):
    print("Derogation context: Scanning emails + authorizations for approval / waiver language...")

    matcher = state.get("matcher_report") or {}
    if not isinstance(matcher, dict):
        matcher = {}

    report = scan_derogation_context(
        matcher_report=matcher,
        email_threads=state.get("email_threads") or "",
        authorization_text=state.get("authorization_text") or "",
    )

    print(f"   -> {report.get('summary_text', '')}")

    return {"derogation_report": report}
