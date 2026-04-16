from typing import Any, TypedDict

from core.docx_parser import ParsedDocx


class GraphState(TypedDict):
    docx_content: ParsedDocx
    email_threads: str
    requirements_records: list[dict[str, Any]]
    test_logs: dict[str, Any]
    test_evidence_corpus: str
    authorization_text: str
    matcher_report: dict[str, Any]
    derogation_report: dict[str, Any]
    auditor_report: dict[str, Any]
    detective_report: dict[str, Any]
    regulatory_report: dict[str, Any]
    pre_isa_report: dict[str, Any]
    assessor_report: dict[str, Any]