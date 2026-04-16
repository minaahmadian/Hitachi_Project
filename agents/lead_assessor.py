import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import GraphState

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}} 
)

def lead_assessor_node(state: GraphState):
    print("Lead Assessor: Cross-referencing reports and making final decision (VDD)...")
    
    system_prompt = SystemMessage(content="""
    You are the Lead Safety Assessor for a SIL 4 Railway project.
    Your job is to read:
    - the 'Auditor Report' (formal safety/compliance findings),
    - the 'Detective Report' (semantic evidence from communications),
    - and the 'Regulatory Report' (deterministic CEI EN 50128 rule-check results)
    and make a final release decision.
    
    Rules:
    1. If Regulatory Report has status "RED_FLAG" OR derogation_needed > 0, final decision should be "NO-GO"
       unless there is a clear documented justification in evidence.
    2. If Detective status is "SUSPICIOUS", strongly bias to "NO-GO".
    3. If Auditor overall_assessment is "NON_COMPLIANT", final decision should be "NO-GO".
    4. Return "GO" only when the combined evidence supports compliance without unresolved high-risk gaps.
    
    You MUST output a valid JSON object matching this schema exactly:
    {
        "final_decision": "GO" or "NO-GO",
        "vdd_explanation": "A professional paragraph for the Version Description Document explaining WHY the release is approved or rejected, citing specific findings from the Auditor and Detective."
    }
    """)
    
    user_message = HumanMessage(content=f"""
    Auditor Report: {json.dumps(state.get('auditor_report', {}))}
    Detective Report: {json.dumps(state.get('detective_report', {}))}
    Regulatory Report: {json.dumps(state.get('regulatory_report', {}))}
    """)
    
    try:
        response = llm.invoke([system_prompt, user_message])
    except Exception:
        # Deterministic fallback when LLM is unavailable.
        auditor = state.get("auditor_report", {})
        detective = state.get("detective_report", {})
        regulatory = state.get("regulatory_report", {})

        auditor_assessment = str(auditor.get("overall_assessment", "PARTIAL")).upper()
        detective_status = str(detective.get("status", "SUSPICIOUS")).upper()
        regulatory_status = str(regulatory.get("status", "RED_FLAG")).upper()
        derogation_needed = int(regulatory.get("derogation_needed", 0) or 0)

        no_go = (
            regulatory_status == "RED_FLAG"
            or derogation_needed > 0
            or detective_status == "SUSPICIOUS"
            or auditor_assessment == "NON_COMPLIANT"
        )

        report = {
            "final_decision": "NO-GO" if no_go else "GO",
            "vdd_explanation": (
                "Deterministic fallback decision was used due to LLM connectivity issues. "
                f"Inputs: auditor={auditor_assessment}, detective={detective_status}, "
                f"regulatory_status={regulatory_status}, derogation_needed={derogation_needed}."
            ),
        }
        return {"assessor_report": report}
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
    except:
        report = {"final_decision": "ERROR", "vdd_explanation": "Parsing error in Assessor."}
        
    return {"assessor_report": report}