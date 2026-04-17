import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from core.audit_export import write_vdd_audit_artifact
from core.state import GraphState
from core.io_utils import load_local_data
from core.docx_parser import parse_docx
from core.project_ingestion import (
    build_test_evidence_corpus,
    load_authorization_text,
    load_requirements_csv,
    load_test_logs_json,
)
from agents.formal_auditor import formal_auditor_node
from agents.derogation_context import derogation_context_node
from agents.context_detective import context_detective_node
from agents.traceability_matcher import traceability_matcher_node
from agents.regulatory_assessor import regulatory_assessor_node
from agents.pre_isa_compiler import pre_isa_compiler_node
from agents.lead_assessor import lead_assessor_node
from vdd.render_vdd_docx import try_render_vdd_docx

if __name__ == "__main__":
    print("Initializing Multi-Agent Network (LangGraph)...\n")
    
    _, emails = load_local_data()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(base_dir)
    docx_path = os.path.join(base_dir, "data", "RSSOM_APCS_FIT.docx")
    parsed_docx = parse_docx(docx_path)

    req_csv = Path(os.getenv("REQUIREMENTS_TRACE_CSV", str(repo_root / "data" / "requirements_trace.csv")))
    logs_json = Path(os.getenv("PROJECT_TEST_LOGS_JSON", str(repo_root / "data" / "test_logs.json")))
    auth_path = Path(os.getenv("AUTHORIZATION_TEXT_PATH", str(repo_root / "data" / "authorizations.txt")))

    requirements_records = load_requirements_csv(req_csv)
    test_logs = load_test_logs_json(logs_json)
    req_ids_for_corpus = [
        str(r.get("requirement_id", "")).strip()
        for r in requirements_records
        if isinstance(r, dict) and str(r.get("requirement_id", "")).strip()
    ]
    test_evidence_corpus = build_test_evidence_corpus(
        parsed_docx,
        repo_root,
        requirement_ids=req_ids_for_corpus,
    )
    authorization_text = load_authorization_text(auth_path)

    workflow = StateGraph(GraphState)

    workflow.add_node("traceability_matcher", traceability_matcher_node)
    workflow.add_node("formal_auditor", formal_auditor_node)
    workflow.add_node("derogation_context", derogation_context_node)
    workflow.add_node("context_detective", context_detective_node)
    workflow.add_node("regulatory_assessor", regulatory_assessor_node)
    workflow.add_node("pre_isa_compiler", pre_isa_compiler_node)
    workflow.add_node("lead_assessor", lead_assessor_node)

    workflow.set_entry_point("traceability_matcher")
    workflow.add_edge("traceability_matcher", "formal_auditor")
    workflow.add_edge("formal_auditor", "derogation_context")
    workflow.add_edge("derogation_context", "context_detective")
    workflow.add_edge("context_detective", "regulatory_assessor")
    workflow.add_edge("regulatory_assessor", "pre_isa_compiler")
    workflow.add_edge("pre_isa_compiler", "lead_assessor")
    workflow.add_edge("lead_assessor", END)
    
    app = workflow.compile()
    
    initial_state = GraphState(
        docx_content=parsed_docx,
        email_threads=emails,
        requirements_records=requirements_records,
        test_logs=test_logs,
        test_evidence_corpus=test_evidence_corpus,
        authorization_text=authorization_text,
        matcher_report={},
        derogation_report={},
        auditor_report={},
        detective_report={},
        regulatory_report={},
        pre_isa_report={},
        assessor_report={},
    )
    
    final_state = app.invoke(initial_state)

    audit_path = write_vdd_audit_artifact(repo_root=repo_root, final_state=final_state)
    if audit_path is not None:
        print(f"\nVDD audit bundle written to: {audit_path}")

    vdd_docx_path = try_render_vdd_docx(repo_root=repo_root, final_state=final_state)
    if vdd_docx_path is not None:
        print(f"\nVDD Word draft (Phase 4) written to: {vdd_docx_path}")
    
    mr = final_state.get("matcher_report") or {}
    if isinstance(mr, dict) and mr:
        ms = mr.get("summary") or {}
        print("\n" + "-" * 50)
        print("TRACEABILITY MATCHER (Phase 2)")
        print("-" * 50)
        print(f"Status            : {mr.get('status')}")
        print(f"Requirements      : {ms.get('total_requirements')} (doc hits: {ms.get('with_document_hit')})")
        print(f"Anomalies         : {ms.get('anomalies_count')} (HIGH: {ms.get('high_severity_count')})")

    dr = final_state.get("derogation_report") or {}
    if isinstance(dr, dict) and dr:
        print("-" * 50)
        print("DEROGATION CONTEXT SCAN (Phase 3 — deterministic)")
        print("-" * 50)
        print(f"Overall           : {dr.get('overall')}")
        print(f"Summary           : {dr.get('summary_text', '')[:200]}")

    pr = final_state.get("pre_isa_report") or {}
    if isinstance(pr, dict) and pr:
        print("-" * 50)
        print("PRE-ISA REPORT (Step D — consolidated)")
        print("-" * 50)
        print(f"Overall           : {pr.get('overall')}")
        _s = str(pr.get("summary_for_vdd", ""))
        print(f"VDD summary       : {_s[:220]}{'…' if len(_s) > 220 else ''}")

    print("\n" + "="*50)
    print("DRAFT VDD (Version Description Document)")
    print("="*50)
    ar = final_state.get("assessor_report") or {}
    print(f"LEAD ASSESSOR MODE: {ar.get('mode', 'unknown')}  (llm = Groq wrote rationale)")
    print(f"RELEASE VERDICT   : {ar.get('final_decision')}")
    print(f"RATIONALE         : {ar.get('vdd_explanation')}")
    print("="*50)