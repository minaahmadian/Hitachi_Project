import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import invoke_chat_groq
from core.state import GraphState

def context_detective_node(state: GraphState):
    print("Detective: Semantic analysis of team communications in progress...")
    system_prompt = SystemMessage(
        content=(
            "SIL4 communications triage. Use matcher + derogation JSON + emails. "
            "Flag bypasses, rushed release, log discrepancies. "
            'JSON only: {"status":"CLEAN"|"SUSPICIOUS","severity":"LOW"|"MEDIUM"|"HIGH"|"CRITICAL",'
            '"reason":"...","red_flags":["..."]}'
        )
    )
    mc = int(os.getenv("DETECTIVE_MATCHER_JSON_CHARS", "2200"))
    dc = int(os.getenv("DETECTIVE_DEROG_JSON_CHARS", "1200"))
    ec = int(os.getenv("DETECTIVE_EMAIL_CHARS", "3200"))

    matcher_blob = ""
    mr = state.get("matcher_report")
    if isinstance(mr, dict) and mr:
        matcher_blob = json.dumps(mr, ensure_ascii=False)[:mc]

    derog_blob = ""
    dr = state.get("derogation_report")
    if isinstance(dr, dict) and dr:
        derog_blob = json.dumps(dr, ensure_ascii=False)[:dc]

    emails = str(state.get("email_threads") or "")
    if len(emails) > ec:
        emails = emails[:ec] + "\n...[emails truncated]"

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