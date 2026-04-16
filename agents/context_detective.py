import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import GraphState

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm

def context_detective_node(state: GraphState):
    print("Detective: Semantic analysis of team communications in progress...")
    system_prompt = SystemMessage(content="""
    You are a Context Detective for a SIL 4 Railway system.
    Your job is to read informal team communications and find hidden safety risks, procedural violations, or unauthorized workarounds.
    
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
    user_message = HumanMessage(content=f"Emails:\n{state['email_threads']}")
    
    try:
        response = _get_llm().invoke([system_prompt, user_message])
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