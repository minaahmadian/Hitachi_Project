import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import get_chat_groq
from core.state import GraphState

def lead_assessor_node(state: GraphState):
    print("Lead Assessor: Cross-referencing reports and making final decision (VDD)...")
    
    system_prompt = SystemMessage(content="""
    You are the Lead Safety Assessor for a SIL 4 Railway project.
    Your job is to read:
    - the 'Auditor Report' (formal safety/compliance findings),
    - the 'Detective Report' (semantic evidence from communications),
    - the 'Regulatory Report' (deterministic CEI EN 50128 rule-check results),
    - the 'Traceability Matcher Report' (deterministic requirements ↔ test design ↔ log consistency),
    - and the 'Derogation Context Scan' (deterministic governance / waiver language near anomalies in emails + authorizations)
    and make a final release decision.
    
    Rules:
    1. If Traceability Matcher status is "RED_FLAG", final decision should be "NO-GO" unless emails or authorizations clearly document an approved derogation for every high-severity anomaly. Use the Derogation Context Scan overall field (STRONG_SIGNALS / WEAK_SIGNALS / NO_SIGNALS) as a hint, not a substitute for human ISA judgement.
    2. If Regulatory Report has status "RED_FLAG" OR derogation_needed > 0, final decision should be "NO-GO"
       unless there is a clear documented justification in evidence.
    3. If Detective status is "SUSPICIOUS", strongly bias to "NO-GO".
    4. If Auditor overall_assessment is "NON_COMPLIANT", final decision should be "NO-GO".
    5. Return "GO" only when the combined evidence supports compliance without unresolved high-risk gaps.
    
    You MUST output a valid JSON object matching this schema exactly:
    {
        "final_decision": "GO" or "NO-GO",
        "vdd_explanation": "A professional paragraph for the Version Description Document explaining WHY the release is approved or rejected, citing specific findings from the Traceability Matcher, Auditor, Detective, and Regulatory outputs where relevant."
    }
    """)
    
    user_message = HumanMessage(content=f"""
    Traceability Matcher Report: {json.dumps(state.get('matcher_report', {}))}
    Derogation Context Scan: {json.dumps(state.get('derogation_report', {}))}
    Auditor Report: {json.dumps(state.get('auditor_report', {}))}
    Detective Report: {json.dumps(state.get('detective_report', {}))}
    Regulatory Report: {json.dumps(state.get('regulatory_report', {}))}
    Authorization / waiver text (may be empty): {(state.get('authorization_text') or '')[:4000]}
    """)
    
    try:
        response = get_chat_groq().invoke([system_prompt, user_message])
    except Exception:
        # Deterministic fallback when LLM is unavailable.
        auditor = state.get("auditor_report", {})
        detective = state.get("detective_report", {})
        regulatory = state.get("regulatory_report", {})
        matcher = state.get("matcher_report", {})
        derogation = state.get("derogation_report", {})

        auditor_assessment = str(auditor.get("overall_assessment", "PARTIAL")).upper()
        detective_status = str(detective.get("status", "SUSPICIOUS")).upper()
        regulatory_status = str(regulatory.get("status", "RED_FLAG")).upper()
        derogation_needed = int(regulatory.get("derogation_needed", 0) or 0)
        matcher_status = str(matcher.get("status", "WARNING")).upper()
        derog_overall = str(derogation.get("overall", "NO_SIGNALS")).upper()

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
                f"Inputs: matcher={matcher_status}, derogation_scan={derog_overall}, "
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