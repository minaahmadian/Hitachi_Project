import json
import os
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import invoke_chat_groq
from core.state import GraphState


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
        "mode": str(raw_report.get("mode", "llm")),
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

    def _build_focused_content(docx: Any, max_chars: int) -> str:
        """
        Prefer full body paragraph text (all non-empty paragraphs), then append flattened
        table rows until max_chars. This avoids missing prose-only obligations that never
        appeared in the old keyword-filter + small table sample.
        """
        if not isinstance(docx, dict):
            return ""

        parts: list[str] = []
        title = (docx.get("title") or "").strip()
        headings = docx.get("headings", []) or []
        if title:
            parts.append(f"DOCUMENT TITLE: {title}")
        if headings:
            parts.append("\nKEY SECTIONS:")
            for h in headings[:30]:
                if isinstance(h, dict):
                    parts.append(f"  {'  ' * (h.get('level', 1) - 1)}{h.get('text', '')}")

        paragraphs = [p.strip() for p in (docx.get("paragraphs") or []) if isinstance(p, str) and p.strip()]
        para_text = "\n\n".join(paragraphs)

        tables = docx.get("tables", []) or []
        table_lines: list[str] = []
        for row in tables:
            if isinstance(row, list) and row:
                table_lines.append("  |  ".join(str(cell)[:400] for cell in row))
        table_text = "\n".join(table_lines)

        header = "\n".join(parts)
        prefix = header + "\n\nFULL PARAGRAPH TEXT (all non-empty paragraphs):\n"
        budget = max_chars - len(prefix) - 40
        if budget <= 0:
            return (prefix + "\n[truncated]")[:max_chars]

        if len(para_text) <= budget:
            out = prefix + para_text
            rest = max_chars - len(out) - 30
            if rest > 0 and table_text:
                tbl_snip = table_text[:rest]
                out += "\n\nTABLES (all rows, flattened):\n" + tbl_snip
                if len(table_text) > len(tbl_snip):
                    out += "\n[truncated tables]"
            return out[:max_chars]
        out = prefix + para_text[:budget] + "\n[truncated paragraphs]"
        return out[:max_chars]

    max_chars = int(os.getenv("AUDITOR_MAX_DOC_CHARS", "12000"))
    if isinstance(docx_content, dict):
        analysis_text = _build_focused_content(docx_content, max_chars)
    else:
        analysis_text = str(docx_content)[:max_chars]
    if len(analysis_text) > max_chars:
        analysis_text = analysis_text[:max_chars] + "\n[truncated]"

    system_prompt = SystemMessage(
        content=(
            "SIL4 railway compliance auditor. From the excerpt only: find SHALL/MUST obligations, "
            "verification/test gaps, risks. "
            "Scores: 85+ COMPLIANT; 50-84 PARTIAL; 0-49 NON_COMPLIANT. "
            "Output JSON only:\n"
            '{"overall_assessment":"COMPLIANT"|"PARTIAL"|"NON_COMPLIANT",'
            '"requirements_found":["..."],"compliance_score":0,'
            '"risks":["..."],"recommendations":["..."]}'
        )
    )

    user_message = HumanMessage(content=f"""
Document source: RSSOM_APCS_FIT.docx (parsed content)

Document text for analysis:
{analysis_text}
""")

    try:
        response = invoke_chat_groq([system_prompt, user_message])
    except Exception as exc:
        # Fallback mode when external LLM call is unavailable.
        return {
            "auditor_report": {
                "overall_assessment": "PARTIAL",
                "requirements_found": extracted_snippets[:5],
                "compliance_score": 50,
                "risks": [f"LLM unreachable in auditor node: {exc}"],
                "recommendations": [
                    "Retry with network access for full semantic audit",
                    "Use deterministic regulatory assessor output as interim gate",
                ],
                "mode": "deterministic_fallback",
            }
        }
    try:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        report = _normalize_auditor_report(json.loads(content))
        report["mode"] = "llm"
    except Exception:
        report = {
            "overall_assessment": "PARTIAL",
            "requirements_found": [],
            "compliance_score": 0,
            "risks": ["Unable to parse LLM output into expected auditor JSON schema"],
            "recommendations": [
                "Retry analysis with cleaner parsed DOCX content and verify LLM JSON response formatting"
            ],
            "mode": "deterministic_fallback",
        }

    return {"auditor_report": report}