"""
Context Detective node.

Uses the shared priority-aware compactor so that HIGH-severity matcher
anomalies are represented as compact digests and therefore cannot be lost to
tail-truncation — the same failure mode that was fixed in ``lead_assessor``.
"""
from __future__ import annotations

import json
import os
from langchain_core.messages import SystemMessage, HumanMessage

from core.blob_compaction import (
    TRUNCATION_MARKER,
    anomaly_digest,
    compact_report,
    severity_verdict_sort_key,
)
from core.llm_factory import invoke_chat_groq
from core.state import GraphState


_DETECTIVE_MATCHER_MUST_KEEP = ("status", "summary", "severity_counts", "anomaly_digests")
_DETECTIVE_MATCHER_DROP_ORDER = (
    "rssom_rag",
    "notes",
    "source_files",
    "test_log_snippets",
    "normalized_requirements",
)

_DETECTIVE_DEROG_MUST_KEEP = ("overall", "matches", "summary")
_DETECTIVE_DEROG_DROP_ORDER = ("citations", "evidence_chain_text", "notes")


def _compact_matcher_for_detective(report: dict | None, max_chars: int) -> str:
    if not isinstance(report, dict) or not report:
        return "{}"

    anomalies_raw = [a for a in (report.get("anomalies") or []) if isinstance(a, dict)]
    anomalies_sorted = sorted(anomalies_raw, key=severity_verdict_sort_key)
    digests = [anomaly_digest(a) for a in anomalies_sorted]

    slim = {k: v for k, v in report.items() if k != "anomalies"}
    return compact_report(
        slim,
        max_chars=max_chars,
        must_keep=_DETECTIVE_MATCHER_MUST_KEEP,
        drop_order=_DETECTIVE_MATCHER_DROP_ORDER,
        extras={"anomaly_digests": digests},
    )


def _compact_derog_for_detective(report: dict | None, max_chars: int) -> str:
    if not isinstance(report, dict) or not report:
        return "{}"
    return compact_report(
        report,
        max_chars=max_chars,
        must_keep=_DETECTIVE_DEROG_MUST_KEEP,
        drop_order=_DETECTIVE_DEROG_DROP_ORDER,
    )


def context_detective_node(state: GraphState):
    print("Detective: Semantic analysis of team communications in progress...")
    system_prompt = SystemMessage(
        content=(
            "SIL4 communications triage. Use matcher digests + derogation JSON + emails. "
            "anomaly_digests entries use '<requirement_id> | <severity> | <type> | <detail>'. "
            "Flag bypasses, rushed release, log discrepancies. "
            f"If you see '{TRUNCATION_MARKER}' anywhere, assume missing evidence and "
            "prefer the SUSPICIOUS status when in doubt. "
            'JSON only: {"status":"CLEAN"|"SUSPICIOUS","severity":"LOW"|"MEDIUM"|"HIGH"|"CRITICAL",'
            '"reason":"...","red_flags":["..."]}'
        )
    )
    mc = int(os.getenv("DETECTIVE_MATCHER_JSON_CHARS", "2600"))
    dc = int(os.getenv("DETECTIVE_DEROG_JSON_CHARS", "1400"))
    ec = int(os.getenv("DETECTIVE_EMAIL_CHARS", "3200"))

    matcher_blob = _compact_matcher_for_detective(state.get("matcher_report"), mc)
    derog_blob = _compact_derog_for_detective(state.get("derogation_report"), dc)

    emails = str(state.get("email_threads") or "")
    if len(emails) > ec:
        emails = emails[:ec] + "\n..." + TRUNCATION_MARKER

    user_message = HumanMessage(
        content=(
            f"Matcher JSON:\n{matcher_blob or '{}'}"
            f"\n\nDerogation JSON:\n{derog_blob or '{}'}"
            f"\n\nEmails:\n{emails}"
        )
    )

    try:
        response = invoke_chat_groq([system_prompt, user_message])
    except Exception as exc:
        return {
            "detective_report": {
                "status": "SUSPICIOUS",
                "severity": "HIGH",
                "reason": f"LLM unreachable in detective node: {exc}",
                "red_flags": [
                    "Unable to semantically verify communications due to LLM connectivity issue"
                ],
                "mode": "deterministic_fallback",
            }
        }
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
        if isinstance(report, dict):
            report["mode"] = "llm"
    except Exception:
        report = {"status": "ERROR", "mode": "deterministic_fallback"}
    return {"detective_report": report}
