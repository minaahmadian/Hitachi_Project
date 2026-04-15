from typing import TypedDict, Dict, Any
from core.docx_parser import ParsedDocx

class GraphState(TypedDict):
    docx_content: ParsedDocx
    email_threads: str
    auditor_report: Dict[str, Any]
    detective_report: Dict[str, Any]
    assessor_report: Dict[str, Any]