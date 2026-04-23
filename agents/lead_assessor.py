"""
Lead SIL4 safety assessor node.

Safety-critical data management
-------------------------------
Historically this node serialised ``pre_isa_report`` and ``matcher_report`` as
one big ``json.dumps(...)[:N]`` blob.  Because the safety-critical fields
(per-anomaly verdicts, HIGH-severity anomalies, RED_FLAG signals) typically
live near the *end* of those JSON structures, a naïve tail-cut could hide
negative evidence from the LLM — the worst possible failure mode for a safety
gate.

This module now uses priority-aware compaction via :mod:`core.blob_compaction`:

* ``verdict_per_anomaly`` is always serialised as a *compact digest*
  (``"<rid> | <sev> | <verdict> | <reason>"``) so tens of entries fit in
  ≤ 2 KB.  The full list is always visible — nothing can be truncated away.
* Matcher anomalies are likewise digested, HIGH-severity first.
* Non-essential metadata (``citations``, ``fingerprints``,
  ``evidence_chain_text``, ``schema_version``, …) is the first thing dropped
  when we must shrink the payload.
* Any compaction produces a ``[truncated]`` marker that the system prompt
  instructs the LLM to treat as a NO-GO safety signal.
* The deterministic fallback iterates the *raw, untruncated*
  ``verdict_per_anomaly`` list, so a RED_FLAG on any HIGH anomaly forces
  NO-GO regardless of LLM availability or prompt length.
"""
from __future__ import annotations

import json
import os
from langchain_core.messages import SystemMessage, HumanMessage

from core.blob_compaction import (
    TRUNCATION_MARKER,
    anomaly_digest,
    compact_report,
    has_high_red_flag,
    severity_verdict_sort_key,
    verdict_digest,
    worst_verdict_in,
)
from core.llm_factory import invoke_chat_groq
from core.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# Priority-aware serialisers
# ─────────────────────────────────────────────────────────────────────────────

# Non-essential pre_isa_report fields that can be dropped to make room.
_PRE_ISA_DROP_ORDER = (
    "citations",
    "fingerprints",
    "schema_version",
    "mode",
    "compiled_at",
    "evidence_chain_text",
    "derogation_context",
)

# Fields that must survive at all costs.
_PRE_ISA_MUST_KEEP = (
    "overall",
    "release_readiness",
    "inputs_digest",
    "verdict_digests",        # digested form of verdict_per_anomaly (we inject)
    "summary_for_vdd_short",  # short text — we inject
)

_MATCHER_DROP_ORDER = (
    "notes",
    "source_files",
    "rssom_rag",          # rich RAG metadata, not needed for final gate
    "test_log_snippets",
    "normalized_requirements",
)
_MATCHER_MUST_KEEP = (
    "status",
    "summary",
    "severity_counts",
    "anomaly_digests",    # digested form (we inject)
)


def _compact_pre_isa_blob(report: dict, max_chars: int) -> str:
    """
    Build a compact, priority-ordered JSON blob for the pre_isa_report.

    Guarantees:
      - every per-anomaly verdict is present as a one-line digest
      - overall, release_readiness, and inputs_digest are always visible
      - lower-priority metadata is dropped first when we must shrink
      - a ``[truncated]`` marker is attached if anything had to go
    """
    if not isinstance(report, dict):
        return json.dumps(report, ensure_ascii=False)[:max_chars]

    verdicts_raw = [v for v in (report.get("verdict_per_anomaly") or []) if isinstance(v, dict)]
    verdicts_sorted = sorted(verdicts_raw, key=severity_verdict_sort_key)
    verdict_digests = [verdict_digest(v) for v in verdicts_sorted]

    # Short narrative (trim aggressively — this is the LAST thing we care about)
    vdd = str(report.get("summary_for_vdd") or "").strip()
    vdd_short = vdd[:360] + ("..." if len(vdd) > 360 else "")

    # We feed a *copy* of the report without the heavy original fields, then
    # inject our digests via ``extras`` so the compactor can't drop them.
    slim = {k: v for k, v in report.items() if k != "verdict_per_anomaly"}
    slim.pop("summary_for_vdd", None)  # replaced by short variant below

    extras = {
        "verdict_digests": verdict_digests,
        "summary_for_vdd_short": vdd_short,
    }
    return compact_report(
        slim,
        max_chars=max_chars,
        must_keep=_PRE_ISA_MUST_KEEP,
        drop_order=_PRE_ISA_DROP_ORDER,
        extras=extras,
    )


def _compact_matcher_blob(report: dict, max_chars: int) -> str:
    """
    Serialise the matcher_report with HIGH-severity anomalies represented as
    compact digests (never dropped) and low-priority metadata shed first.
    """
    if not isinstance(report, dict):
        return json.dumps(report, ensure_ascii=False)[:max_chars]

    anomalies_raw = [a for a in (report.get("anomalies") or []) if isinstance(a, dict)]
    anomalies_sorted = sorted(anomalies_raw, key=severity_verdict_sort_key)
    digests = [anomaly_digest(a) for a in anomalies_sorted]

    slim = {k: v for k, v in report.items() if k != "anomalies"}

    extras = {"anomaly_digests": digests}
    return compact_report(
        slim,
        max_chars=max_chars,
        must_keep=_MATCHER_MUST_KEEP,
        drop_order=_MATCHER_DROP_ORDER,
        extras=extras,
    )


def _compact_generic_blob(report: dict | None, max_chars: int, must_keep: tuple[str, ...]) -> str:
    """Generic compaction for derogation / auditor / detective / regulatory reports."""
    if not isinstance(report, dict) or not report:
        return "{}"
    return compact_report(
        report,
        max_chars=max_chars,
        must_keep=must_keep,
        drop_order=("citations", "fingerprints", "evidence_chain_text", "notes", "raw_excerpt"),
    )


def _truncate_auth_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n..." + TRUNCATION_MARKER


# ─────────────────────────────────────────────────────────────────────────────
# Graph node
# ─────────────────────────────────────────────────────────────────────────────

def lead_assessor_node(state: GraphState):
    print("Lead Assessor: Cross-referencing reports and making final decision (VDD)...")

    system_prompt = SystemMessage(
        content=(
            "Lead SIL4 safety assessor. Read Pre-ISA JSON first (overall, verdict_digests, inputs_digest). "
            "verdict_digests entries use the format '<requirement_id> | <severity> | <verdict> | <reason>'. "
            "Matcher anomaly_digests follow '<requirement_id> | <severity> | <type> | <detail>'. "
            "SAFETY RULES (apply BEFORE anything else): "
            "1) If ANY verdict_digests entry has verdict=RED_FLAG on a HIGH anomaly → NO-GO. "
            "2) If you see the marker '[truncated]' ANYWHERE in the payload, treat it as unknown-but-possible "
            "negative evidence and declare NO-GO unless the visible evidence already unambiguously supports GO. "
            "3) NO-GO also if: matcher RED_FLAG without derogations for each HIGH anomaly; "
            "regulatory RED_FLAG or derogation_needed>0 without justification; detective SUSPICIOUS; "
            "auditor NON_COMPLIANT. "
            "GO only if all HIGH-severity anomalies are JUSTIFICATION_SIGNALS or there are none. "
            'Output JSON: {"final_decision":"GO"|"NO-GO","vdd_explanation":"short professional paragraph"}'
        )
    )

    # Default budgets are intentionally larger than the old 1800/800/700 because
    # compact_report() guarantees critical fields stay in-budget and drops the
    # rest; the LLM benefits from seeing more context when it fits.
    pre_max = int(os.getenv("LEAD_ASSESSOR_PRE_ISA_CHARS", "4000"))
    rep_max = int(os.getenv("LEAD_ASSESSOR_REPORT_CHARS", "1200"))
    auth_max = int(os.getenv("LEAD_ASSESSOR_AUTH_CHARS", "700"))

    pre_isa_blob = _compact_pre_isa_blob(state.get("pre_isa_report") or {}, pre_max)
    matcher_blob = _compact_matcher_blob(state.get("matcher_report") or {}, rep_max)

    derog_blob = _compact_generic_blob(
        state.get("derogation_report"), rep_max, must_keep=("overall", "matches", "summary")
    )
    auditor_blob = _compact_generic_blob(
        state.get("auditor_report"),
        rep_max,
        must_keep=("overall_assessment", "compliance_score", "risks", "recommendations"),
    )
    detective_blob = _compact_generic_blob(
        state.get("detective_report"),
        rep_max,
        must_keep=("status", "severity", "reason", "red_flags"),
    )
    regulatory_blob = _compact_generic_blob(
        state.get("regulatory_report"),
        rep_max,
        must_keep=("status", "derogation_needed", "rationale", "clauses"),
    )
    auth_blob = _truncate_auth_text((state.get("authorization_text") or "").strip(), auth_max)

    # Explicit truncation awareness in the prompt payload.
    any_truncated = any(
        TRUNCATION_MARKER in b
        for b in (pre_isa_blob, matcher_blob, derog_blob, auditor_blob, detective_blob, regulatory_blob, auth_blob)
    )
    truncation_banner = (
        f"TRUNCATION FLAG: one or more payloads above were compacted. Marker='{TRUNCATION_MARKER}'. "
        "If the visible evidence is insufficient to prove GO, declare NO-GO."
    ) if any_truncated else "TRUNCATION FLAG: none — all evidence fully visible."

    user_message = HumanMessage(content=f"""
    Pre-ISA Report (consolidated — read first): {pre_isa_blob}
    Traceability Matcher Report: {matcher_blob}
    Derogation Context Scan: {derog_blob}
    Auditor Report: {auditor_blob}
    Detective Report: {detective_blob}
    Regulatory Report: {regulatory_blob}
    Authorization / waiver text (may be empty): {auth_blob}
    {truncation_banner}
    """)

    try:
        response = invoke_chat_groq([system_prompt, user_message])
    except Exception as exc:
        return _deterministic_fallback(state, reason=f"Groq call failed ({type(exc).__name__}): {exc!s}")

    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
        if not isinstance(report, dict):
            raise ValueError("LLM returned non-dict JSON")
        report["mode"] = "llm"
    except Exception:
        return _deterministic_fallback(state, reason="Parsing error in Assessor LLM response")

    # Post-LLM safety override: even if the LLM said GO, if the raw untruncated
    # verdict list contains a HIGH RED_FLAG, force NO-GO. This is the final
    # independent safety gate and is deliberately not dependent on prompt size
    # or LLM behaviour.
    pre_isa = state.get("pre_isa_report") or {}
    raw_verdicts = pre_isa.get("verdict_per_anomaly") or []
    if has_high_red_flag(raw_verdicts) and str(report.get("final_decision", "")).upper() != "NO-GO":
        report["final_decision"] = "NO-GO"
        prior_expl = str(report.get("vdd_explanation") or "").strip()
        report["vdd_explanation"] = (
            "Safety override: at least one HIGH-severity anomaly has verdict=RED_FLAG in "
            "verdict_per_anomaly. Final decision forced to NO-GO regardless of LLM output. "
            f"Prior LLM rationale: {prior_expl}"
        ).strip()
        report["safety_override"] = True

    return {"assessor_report": report}


def _deterministic_fallback(state: GraphState, *, reason: str) -> dict:
    """
    LLM-independent NO-GO/GO decision. This is deliberately simple: we inspect
    the ORIGINAL, untruncated per-anomaly verdict list and any explicit status
    gates from upstream nodes. No character limits apply here.
    """
    print(f"Lead Assessor: {reason}", flush=True)

    auditor = state.get("auditor_report", {}) or {}
    detective = state.get("detective_report", {}) or {}
    regulatory = state.get("regulatory_report", {}) or {}
    matcher = state.get("matcher_report", {}) or {}
    derogation = state.get("derogation_report", {}) or {}
    pre_isa = state.get("pre_isa_report", {}) or {}

    auditor_assessment = str(auditor.get("overall_assessment", "PARTIAL")).upper()
    detective_status = str(detective.get("status", "SUSPICIOUS")).upper()
    regulatory_status = str(regulatory.get("status", "RED_FLAG")).upper()
    derogation_needed = int(regulatory.get("derogation_needed", 0) or 0)
    matcher_status = str(matcher.get("status", "WARNING")).upper()
    derog_overall = str(derogation.get("overall", "NO_SIGNALS")).upper()
    pre_isa_overall = str(pre_isa.get("overall", "REVIEW_REQUIRED")).upper()
    pre_isa_summary = str(pre_isa.get("summary_for_vdd", "")).strip()

    raw_verdicts = pre_isa.get("verdict_per_anomaly") or []
    any_high_red_flag = has_high_red_flag(raw_verdicts)
    worst_seen = worst_verdict_in(raw_verdicts) or "NONE"

    no_go = (
        any_high_red_flag
        or matcher_status == "RED_FLAG"
        or regulatory_status == "RED_FLAG"
        or derogation_needed > 0
        or detective_status == "SUSPICIOUS"
        or auditor_assessment == "NON_COMPLIANT"
    )

    report = {
        "final_decision": "NO-GO" if no_go else "GO",
        "vdd_explanation": (
            f"Deterministic fallback decision ({reason}). "
            f"Pre-ISA overall={pre_isa_overall}. {pre_isa_summary} "
            f"Raw gates: matcher={matcher_status}, derogation_scan={derog_overall}, "
            f"auditor={auditor_assessment}, detective={detective_status}, "
            f"regulatory_status={regulatory_status}, derogation_needed={derogation_needed}, "
            f"any_high_red_flag={any_high_red_flag}, worst_verdict={worst_seen}."
        ),
        "mode": "deterministic_fallback",
        "safety_override": any_high_red_flag,
    }
    return {"assessor_report": report}
