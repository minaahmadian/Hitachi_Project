import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import invoke_chat_groq
from core.state import GraphState

def context_detective_node(state: GraphState):
    print("Detective: Semantic analysis of team communications in progress...")
    system_prompt = SystemMessage(content="""
    You are a Context Detective for a SIL 4 Railway system.
    Your job is to read informal team communications and find hidden safety risks, procedural violations, or unauthorized workarounds.
    You are also given the Traceability Matcher output (deterministic CSV ↔ test evidence ↔ logs cross-check) and the Derogation Context Scan (deterministic keyword scan of emails + authorizations near those anomalies). Use both to prioritize emails that might justify or worsen a matcher anomaly (e.g. failed MQTT test, formal deviation, workaround language).
    
    Look specifically for:
    - Bypassed hardware tests (e.g., HIL) or disabled sensors.
    - Rushed releases ("just to pass the build").
    - Discrepancies between formal logs and actual actions.
    
    You MUST output a valid JSON object matching this schema exactly:
    {
        "status": "CLEAN" or "SUSPICIOUS",
        "severity": "LOW", "MEDIUM", "HIGH", or "CRITICAL",
        "reason": "Detailed explanation of the anomalies found",
        "red_flags": ["list", "of", "exact", "quotes", "or", "issues"]
    }
    """)
    matcher_blob = ""
    mr = state.get("matcher_report")
    if isinstance(mr, dict) and mr:
        matcher_blob = json.dumps(mr, ensure_ascii=False)[:6000]

    derog_blob = ""
    dr = state.get("derogation_report")
    if isinstance(dr, dict) and dr:
        derog_blob = json.dumps(dr, ensure_ascii=False)[:3500]

    user_message = HumanMessage(
        content=(
            f"Traceability Matcher Report (JSON, may be truncated):\n{matcher_blob or '{}'}"
            f"\n\nDerogation Context Scan (JSON, may be truncated):\n{derog_blob or '{}'}"
            f"\n\nEmails:\n{state['email_threads']}"
        )
    )
    
    try:
        response = invoke_chat_groq([system_prompt, user_message])
    except Exception as exc:
        # Fallback mode when external LLM call is unavailable.
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
    except:
        report = {"status": "ERROR", "mode": "deterministic_fallback"}
    return {"detective_report": report}