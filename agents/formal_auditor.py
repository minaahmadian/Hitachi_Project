import json
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import GraphState

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}} 
)


def _extract_text_from_docx_content(node: Any) -> list[str]:
    snippets: list[str] = []

    if node is None:
        return snippets

    if isinstance(node, str):
        cleaned = node.strip()
        if cleaned:
            snippets.append(cleaned)
        return snippets

    if isinstance(node, (int, float, bool)):
        snippets.append(str(node))
        return snippets

    if isinstance(node, list):
        for item in node:
            snippets.extend(_extract_text_from_docx_content(item))
        return snippets

    if isinstance(node, dict):
        preferred_text_keys = {
            "text",
            "content",
            "value",
            "title",
            "heading",
            "paragraph",
            "requirement",
            "description",
            "note",
        }

        for key, value in node.items():
            if key.lower() in preferred_text_keys:
                snippets.extend(_extract_text_from_docx_content(value))

        for value in node.values():
            snippets.extend(_extract_text_from_docx_content(value))

        return snippets

    return snippets


def _normalize_auditor_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    assessment = raw_report.get("overall_assessment", "PARTIAL")
    if assessment not in {"COMPLIANT", "PARTIAL", "NON_COMPLIANT"}:
        assessment = "PARTIAL"

    score = raw_report.get("compliance_score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0
    score = max(0, min(100, score))

    requirements = raw_report.get("requirements_found", [])
    if not isinstance(requirements, list):
        requirements = [str(requirements)]

    risks = raw_report.get("risks", [])
    if not isinstance(risks, list):
        risks = [str(risks)]

    recommendations = raw_report.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]

    return {
        "overall_assessment": assessment,
        "requirements_found": [str(item) for item in requirements if str(item).strip()],
        "compliance_score": score,
        "risks": [str(item) for item in risks if str(item).strip()],
        "recommendations": [str(item) for item in recommendations if str(item).strip()],
    }


def formal_auditor_node(state: GraphState):
    print("Auditor: Analyzing safety requirements and compliance evidence from DOCX...")

    docx_content = state.get("docx_content")
    extracted_snippets = _extract_text_from_docx_content(docx_content)
    combined_text = "\n".join(dict.fromkeys(extracted_snippets))

    if not combined_text.strip():
        return {
            "auditor_report": {
                "overall_assessment": "NON_COMPLIANT",
                "requirements_found": [],
                "compliance_score": 0,
                "risks": ["No parsable DOCX content found in state['docx_content']"],
                "recommendations": [
                    "Ensure the RSSOM_APCS_FIT.docx file is parsed and loaded into state['docx_content'] before running the auditor"
                ],
            }
        }

    def _build_focused_content(docx: Any) -> str:
        parts = []
        
        title = docx.get("title", "") if isinstance(docx, dict) else ""
        headings = docx.get("headings", []) if isinstance(docx, dict) else []
        if title:
            parts.append(f"DOCUMENT TITLE: {title}")
        if headings:
            parts.append("\nKEY SECTIONS:")
            for h in headings[:20]:
                if isinstance(h, dict):
                    parts.append(f"  {'  ' * (h.get('level', 1) - 1)}{h.get('text', '')}")
        
        paragraphs = docx.get("paragraphs", []) if isinstance(docx, dict) else []
        important_paras = []
        keywords = ["shall", "must", "requirement", "safety", "compliance", "verify", "test", "hazard", "risk"]
        for p in paragraphs:
            if isinstance(p, str) and any(kw in p.lower() for kw in keywords):
                important_paras.append(p)
                if len(important_paras) >= 30:
                    break
        
        if important_paras:
            parts.append("\nKEY REQUIREMENTS AND OBLIGATIONS:")
            parts.extend(important_paras[:30])
        
        tables = docx.get("tables", []) if isinstance(docx, dict) else []
        if tables:
            parts.append("\nREQUIREMENT TABLES (samples):")
            for i, table in enumerate(tables[:5]):
                if isinstance(table, list) and len(table) > 0:
                    parts.append(f"\nTable {i+1}:")
                    for row in table[:5]:
                        if isinstance(row, list):
                            parts.append("  |  ".join(str(cell)[:100] for cell in row))
        
        return "\n".join(parts)
    
    if isinstance(docx_content, dict):
        analysis_text = _build_focused_content(docx_content)
    else:
        analysis_text = str(docx_content)[:4000]
    
    max_chars = 4000
    if len(analysis_text) > max_chars:
        analysis_text = analysis_text[:max_chars] + "\n[Content truncated due to size limits]"

    system_prompt = SystemMessage(content="""
You are a Formal Safety Compliance Auditor for a SIL 4 railway software process.

Task:
Analyze the provided parsed document text (from RSSOM_APCS_FIT.docx) and extract meaningful compliance intelligence.

You MUST identify and reason about:
1) Safety requirements explicitly or implicitly stated
2) Compliance criteria / acceptance criteria
3) Test coverage and verification requirements
4) Mandatory obligations containing SHALL / MUST (or equivalent strict obligation language)
5) Risks, gaps, ambiguities, or concerns

Scoring guidance:
- 85-100 COMPLIANT: requirements are clear, testability is strong, obligations are concrete, and risks are low/mitigated
- 50-84 PARTIAL: some compliance evidence exists but there are gaps, ambiguity, or weak verification detail
- 0-49 NON_COMPLIANT: major missing requirements, weak/no verification criteria, or significant unresolved risks

Output MUST be valid JSON and MUST match this schema exactly:
{
  "overall_assessment": "COMPLIANT" | "PARTIAL" | "NON_COMPLIANT",
  "requirements_found": [
    "Concise requirement statement including MUST/SHALL wording when present"
  ],
  "compliance_score": 0,
  "risks": ["Potential issue or concern"],
  "recommendations": ["Concrete action to improve compliance"]
}

Rules:
- Use only evidence from the provided text
- Do not invent requirements that are not present
- Keep requirements_found, risks, and recommendations specific and actionable
- Return at least 3 items in requirements_found when possible
""")

    user_message = HumanMessage(content=f"""
Document source: RSSOM_APCS_FIT.docx (parsed content)

Document text for analysis:
{analysis_text}
""")

    response = llm.invoke([system_prompt, user_message])
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = _normalize_auditor_report(json.loads(content))
    except Exception:
        report = {
            "overall_assessment": "PARTIAL",
            "requirements_found": [],
            "compliance_score": 0,
            "risks": ["Unable to parse LLM output into expected auditor JSON schema"],
            "recommendations": [
                "Retry analysis with cleaner parsed DOCX content and verify LLM JSON response formatting"
            ],
        }

    return {"auditor_report": report}