from __future__ import annotations

from core.state import GraphState
from traceability.matcher import run_traceability_match


def traceability_matcher_node(state: GraphState):
    print("Traceability Matcher: Cross-linking requirements CSV ↔ test evidence ↔ execution logs...")

    requirements = state.get("requirements_records") or []
    corpus = state.get("test_evidence_corpus") or ""
    logs = state.get("test_logs") or {}

    report = run_traceability_match(
        requirements_records=requirements if isinstance(requirements, list) else [],
        test_evidence_corpus=corpus if isinstance(corpus, str) else str(corpus),
        test_logs=logs if isinstance(logs, dict) else {},
    )

    return {"matcher_report": report}
