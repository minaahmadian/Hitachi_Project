import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import invoke_chat_groq
from core.state import GraphState

# Severity ordering used when sorting anomalies before serialisation.
# HIGH-severity entries appear first so naïve character truncation discards
# LOW-severity tail content rather than hiding the entries that drive NO-GO.
_SEVERITY_ORDER: dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_UNKNOWN_SEVERITY_RANK = 3

# Verdict ordering: worst signals first.
_VERDICT_ORDER: dict[str, int] = {"RED_FLAG": 0, "REVIEW": 1, "JUSTIFICATION_SIGNALS": 2, "TRACKED": 3}
_UNKNOWN_VERDICT_RANK = 4


def _truncate_blob(text: str, max_chars: int) -> str:
    """Naïve prefix truncation — only use for fields where order doesn't matter."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _anomaly_sort_key(item: dict) -> tuple[int, int]:
    sev = _SEVERITY_ORDER.get(str(item.get("severity", "")).upper(), _UNKNOWN_SEVERITY_RANK)
    ver = _VERDICT_ORDER.get(str(item.get("verdict", "")).upper(), _UNKNOWN_VERDICT_RANK)
    return (sev, ver)


def _verdict_sort_key(item: dict) -> tuple[int, int]:
    sev = _SEVERITY_ORDER.get(str(item.get("matcher_severity", "")).upper(), _UNKNOWN_SEVERITY_RANK)
    ver = _VERDICT_ORDER.get(str(item.get("verdict", "")).upper(), _UNKNOWN_VERDICT_RANK)
    return (sev, ver)


def _critical_pre_isa_blob(report: dict, max_chars: int) -> str:
    """
    Build a compact, priority-ordered JSON blob for the pre_isa_report so that
    naive character truncation NEVER silently hides negative evidence.

    Field priority (front-loaded, always fully visible):
      1. overall / release_readiness          — top-level GO/NO-GO signal
      2. inputs_digest                        — all status gate values in one compact dict
      3. verdict_per_anomaly (HIGH-first)     — per-requirement RED_FLAG / REVIEW verdicts
      4. summary_for_vdd excerpt              — human-readable narrative (trimmed to fit)

    Deliberately omitted (metadata, not decision-relevant):
      schema_version, mode, compiled_at, fingerprints, evidence_chain_text, citations
    """
    if not isinstance(report, dict):
        return _truncate_blob(json.dumps(report, ensure_ascii=False), max_chars)

    verdicts = [v for v in (report.get("verdict_per_anomaly") or []) if isinstance(v, dict)]
    verdicts_sorted = sorted(verdicts, key=_verdict_sort_key)

    compact: dict = {
        "overall": report.get("overall"),
        "release_readiness": report.get("release_readiness"),
        "inputs_digest": report.get("inputs_digest"),
        "verdict_per_anomaly": verdicts_sorted,
    }

    base = json.dumps(compact, ensure_ascii=False)
    remaining = max_chars - len(base)
    if remaining > 80:
        vdd = str(report.get("summary_for_vdd") or "")
        # Leave a small margin for the JSON key + quotes + close brace
        compact["summary_for_vdd_excerpt"] = vdd[: max(0, remaining - 40)]
        base = json.dumps(compact, ensure_ascii=False)

    return base[:max_chars]


def _critical_matcher_blob(report: dict, max_chars: int) -> str:
    """
    Serialize the matcher_report with anomalies sorted HIGH-severity first so
    that if the list must be truncated, LOW-severity tail entries are discarded,
    not the HIGH-severity entries that feed the NO-GO gate.
    """
    if not isinstance(report, dict):
        return _truncate_blob(json.dumps(report, ensure_ascii=False), max_chars)

    anomalies = [a for a in (report.get("anomalies") or []) if isinstance(a, dict)]
    anomalies_sorted = sorted(anomalies, key=_anomaly_sort_key)

    reordered = {
        "status": report.get("status"),
        "summary": report.get("summary"),
        "anomalies": anomalies_sorted,
    }
    # append any remaining keys that aren't already included
    for k, v in report.items():
        if k not in reordered:
            reordered[k] = v

    serialized = json.dumps(reordered, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return serialized
    return serialized[:max_chars] + "\n...[truncated]"


def lead_assessor_node(state: GraphState):
    print("Lead Assessor: Cross-referencing reports and making final decision (VDD)...")

    system_prompt = SystemMessage(
        content=(
            "Lead SIL4 safety assessor. Read Pre-ISA JSON first (overall, per-anomaly verdicts). "
            "NO-GO if: any verdict_per_anomaly entry has verdict=RED_FLAG; "
            "or matcher RED_FLAG without documented derogations for each HIGH anomaly; "
            "or regulatory RED_FLAG / derogation_needed>0 without justification; "
            "or detective SUSPICIOUS; or auditor NON_COMPLIANT. "
            "GO only if all HIGH-severity anomalies are JUSTIFICATION_SIGNALS or no HIGH anomalies exist. "
            'Output JSON: {"final_decision":"GO"|"NO-GO","vdd_explanation":"short professional paragraph"}'
        )
    )

    # Groq free/low tiers often enforce ~6k TPM and small per-request bodies; full JSON dumps
    # caused 413 Payload Too Large. Tune via env if your tier allows larger prompts.
    pre_max = int(os.getenv("LEAD_ASSESSOR_PRE_ISA_CHARS", "1800"))
    rep_max = int(os.getenv("LEAD_ASSESSOR_REPORT_CHARS", "800"))
    auth_max = int(os.getenv("LEAD_ASSESSOR_AUTH_CHARS", "700"))

    # Use priority-aware serializers for the two reports that carry negative-evidence
    # fields near the end of their JSON structure.
    pre_isa_blob = _critical_pre_isa_blob(
        state.get("pre_isa_report") or {},
        pre_max,
    )
    matcher_blob = _critical_matcher_blob(
        state.get("matcher_report") or {},
        rep_max,
    )

    # These reports are smaller and place their status key first in the dict;
    # naïve truncation is safe here.
    derog_blob = _truncate_blob(
        json.dumps(state.get("derogation_report") or {}, ensure_ascii=False),
        rep_max,
    )
    auditor_blob = _truncate_blob(
        json.dumps(state.get("auditor_report") or {}, ensure_ascii=False),
        rep_max,
    )
    detective_blob = _truncate_blob(
        json.dumps(state.get("detective_report") or {}, ensure_ascii=False),
        rep_max,
    )
    regulatory_blob = _truncate_blob(
        json.dumps(state.get("regulatory_report") or {}, ensure_ascii=False),
        rep_max,
    )
    auth_blob = _truncate_blob(
        (state.get("authorization_text") or "").strip(),
        auth_max,
    )

    user_message = HumanMessage(content=f"""
    Pre-ISA Report (consolidated — read first): {pre_isa_blob}
    Traceability Matcher Report: {matcher_blob}
    Derogation Context Scan: {derog_blob}
    Auditor Report: {auditor_blob}
    Detective Report: {detective_blob}
    Regulatory Report: {regulatory_blob}
    Authorization / waiver text (may be empty): {auth_blob}
    """)

    try:
        response = invoke_chat_groq([system_prompt, user_message])
    except Exception as exc:
        # Deterministic fallback when LLM is unavailable (rate limit, auth, network, etc.).
        print(
            f"Lead Assessor: Groq call failed ({type(exc).__name__}): {exc!s}",
            flush=True,
        )
        auditor = state.get("auditor_report", {})
        detective = state.get("detective_report", {})
        regulatory = state.get("regulatory_report", {})
        matcher = state.get("matcher_report", {})
        derogation = state.get("derogation_report", {})
        pre_isa = state.get("pre_isa_report") or {}

        auditor_assessment = str(auditor.get("overall_assessment", "PARTIAL")).upper()
        detective_status = str(detective.get("status", "SUSPICIOUS")).upper()
        regulatory_status = str(regulatory.get("status", "RED_FLAG")).upper()
        derogation_needed = int(regulatory.get("derogation_needed", 0) or 0)
        matcher_status = str(matcher.get("status", "WARNING")).upper()
        derog_overall = str(derogation.get("overall", "NO_SIGNALS")).upper()
        pre_isa_overall = str(pre_isa.get("overall", "REVIEW_REQUIRED")).upper()
        pre_isa_summary = str(pre_isa.get("summary_for_vdd", "")).strip()

        # Also check per-anomaly verdicts directly from state (never truncated here).
        any_high_red_flag = any(
            str(v.get("matcher_severity", "")).upper() == "HIGH"
            and str(v.get("verdict", "")).upper() == "RED_FLAG"
            for v in (pre_isa.get("verdict_per_anomaly") or [])
            if isinstance(v, dict)
        )

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
                "Deterministic fallback decision was used due to LLM connectivity issues. "
                f"Pre-ISA overall={pre_isa_overall}. {pre_isa_summary} "
                f"Raw gates: matcher={matcher_status}, derogation_scan={derog_overall}, "
                f"auditor={auditor_assessment}, detective={detective_status}, "
                f"regulatory_status={regulatory_status}, derogation_needed={derogation_needed}, "
                f"any_high_red_flag={any_high_red_flag}."
            ),
            "mode": "deterministic_fallback",
        }
        return {"assessor_report": report}

    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
        if isinstance(report, dict):
            report["mode"] = "llm"
    except Exception:
        report = {
            "final_decision": "ERROR",
            "vdd_explanation": "Parsing error in Assessor.",
            "mode": "deterministic_fallback",
        }

    return {"assessor_report": report}
