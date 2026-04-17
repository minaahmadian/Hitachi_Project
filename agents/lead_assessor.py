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
    
    system_prompt = SystemMessage(content="""
    You are the Lead Safety Assessor for a SIL 4 Railway project.
    Your job is to read:
    - the 'Auditor Report' (formal safety/compliance findings),
    - the 'Detective Report' (semantic evidence from communications),
    - the 'Regulatory Report' (deterministic CEI EN 50128 rule-check results),
    - the 'Traceability Matcher Report' (deterministic requirements ↔ test design ↔ log consistency),
    - the 'Derogation Context Scan' (deterministic governance / waiver language near anomalies in emails + authorizations),
    - and the 'Pre-ISA Report' (single consolidated JSON: overall gate, per-anomaly verdicts, and citations for VDD / audit)
    and make a final release decision.
    
    Rules:
    1. Read the Pre-ISA Report first: use ``overall`` (RED_FLAG / REVIEW_REQUIRED / CLEAR) and ``verdict_per_anomaly`` as the structured summary of traceability + derogation posture before reconciling with the other reports.
    2. If Traceability Matcher status is "RED_FLAG", final decision should be "NO-GO" unless emails or authorizations clearly document an approved derogation for every high-severity anomaly. Use the Derogation Context Scan overall field (STRONG_SIGNALS / WEAK_SIGNALS / NO_SIGNALS) as a hint, not a substitute for human ISA judgement.
    3. If Regulatory Report has status "RED_FLAG" OR derogation_needed > 0, final decision should be "NO-GO"
       unless there is a clear documented justification in evidence.
    4. If Detective status is "SUSPICIOUS", strongly bias to "NO-GO".
    5. If Auditor overall_assessment is "NON_COMPLIANT", final decision should be "NO-GO".
    6. Return "GO" only when the combined evidence supports compliance without unresolved high-risk gaps.
    
    You MUST output a valid JSON object matching this schema exactly:
    {
        "final_decision": "GO" or "NO-GO",
        "vdd_explanation": "A professional paragraph for the Version Description Document explaining WHY the release is approved or rejected, citing the Pre-ISA Report, Traceability Matcher, Derogation Scan, Auditor, Detective, and Regulatory outputs where relevant."
    }
    """)
    
    # Groq free/low tiers often enforce ~6k TPM and small per-request bodies; full JSON dumps
    # caused 413 Payload Too Large. Tune via env if your tier allows larger prompts.
    pre_max = int(os.getenv("LEAD_ASSESSOR_PRE_ISA_CHARS", "2500"))
    rep_max = int(os.getenv("LEAD_ASSESSOR_REPORT_CHARS", "1200"))
    auth_max = int(os.getenv("LEAD_ASSESSOR_AUTH_CHARS", "1000"))

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