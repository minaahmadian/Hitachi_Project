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
    Your job is to read the 'Auditor Report' (based on logs) and the 'Detective Report' (based on emails) and make a final release decision.
    
    Rules:
    1. If Auditor status is "GO" AND Detective status is "CLEAN", the final decision is "GO".
    2. If EITHER the Auditor says "NO-GO" OR the Detective says "SUSPICIOUS", the final decision is "NO-GO".
    
    You MUST output a valid JSON object matching this schema exactly:
    {
        "final_decision": "GO" or "NO-GO",
        "vdd_explanation": "A professional paragraph for the Version Description Document explaining WHY the release is approved or rejected, citing specific findings from the Auditor and Detective."
    }
    """)
    
    user_message = HumanMessage(content=f"""
    Auditor Report: {json.dumps(state.get('auditor_report', {}))}
    Detective Report: {json.dumps(state.get('detective_report', {}))}
    """)
    
    response = llm.invoke([system_prompt, user_message])
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = json.loads(content)
    except:
        report = {"final_decision": "ERROR", "vdd_explanation": "Parsing error in Assessor."}
        
    return {"assessor_report": report}