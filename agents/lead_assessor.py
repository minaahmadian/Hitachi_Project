import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import invoke_chat_groq
from core.state import GraphState


def _truncate_blob(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def lead_assessor_node(state: GraphState):
    print("Lead Assessor: Cross-referencing reports and making final decision (VDD)...")
    
    system_prompt = SystemMessage(
        content=(
            "Lead SIL4 safety assessor. Read Pre-ISA JSON first (overall, per-anomaly). "
            "NO-GO if: matcher RED_FLAG without documented derogations for each HIGH anomaly; "
            "or regulatory RED_FLAG / derogation_needed>0 without justification; "
            "or detective SUSPICIOUS; or auditor NON_COMPLIANT. "
            "GO only if evidence supports compliance. "
            'Output JSON: {"final_decision":"GO"|"NO-GO","vdd_explanation":"short professional paragraph"}'
        )
    )
    
    # Groq free/low tiers often enforce ~6k TPM and small per-request bodies; full JSON dumps
    # caused 413 Payload Too Large. Tune via env if your tier allows larger prompts.
    pre_max = int(os.getenv("LEAD_ASSESSOR_PRE_ISA_CHARS", "1800"))
    rep_max = int(os.getenv("LEAD_ASSESSOR_REPORT_CHARS", "800"))
    auth_max = int(os.getenv("LEAD_ASSESSOR_AUTH_CHARS", "700"))

    pre_isa_blob = _truncate_blob(
        json.dumps(state.get("pre_isa_report") or {}, ensure_ascii=False),
        pre_max,
    )
    matcher_blob = _truncate_blob(
        json.dumps(state.get("matcher_report") or {}, ensure_ascii=False),
        rep_max,
    )
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

        no_go = (
            matcher_status == "RED_FLAG"
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
                f"regulatory_status={regulatory_status}, derogation_needed={derogation_needed}."
            ),
            "mode": "deterministic_fallback",
        }
        return {"assessor_report": report}
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
        if isinstance(report, dict):
            report["mode"] = "llm"
    except:
        report = {
            "final_decision": "ERROR",
            "vdd_explanation": "Parsing error in Assessor.",
            "mode": "deterministic_fallback",
        }
        
    return {"assessor_report": report}