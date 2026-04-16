import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from core.audit_export import write_vdd_audit_artifact
from core.state import GraphState
from core.io_utils import load_local_data
from core.docx_parser import parse_docx
from agents.formal_auditor import formal_auditor_node
from agents.context_detective import context_detective_node
from agents.regulatory_assessor import regulatory_assessor_node
from agents.lead_assessor import lead_assessor_node

if __name__ == "__main__":
    print("Initializing Multi-Agent Network (LangGraph)...\n")
    
    _, emails = load_local_data()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    docx_path = os.path.join(base_dir, "data", "RSSOM_APCS_FIT.docx")
    parsed_docx = parse_docx(docx_path)
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("formal_auditor", formal_auditor_node)
    workflow.add_node("context_detective", context_detective_node)
    workflow.add_node("regulatory_assessor", regulatory_assessor_node)
    workflow.add_node("lead_assessor", lead_assessor_node)
    
    workflow.set_entry_point("formal_auditor")
    workflow.add_edge("formal_auditor", "context_detective")
    workflow.add_edge("context_detective", "regulatory_assessor")
    workflow.add_edge("regulatory_assessor", "lead_assessor")
    workflow.add_edge("lead_assessor", END)
    
    app = workflow.compile()
    
    initial_state = GraphState(
        docx_content=parsed_docx,
        email_threads=emails,
        auditor_report={},
        detective_report={},
        regulatory_report={},
        assessor_report={}
    )
    
    final_state = app.invoke(initial_state)

    repo_root = Path(__file__).resolve().parent
    audit_path = write_vdd_audit_artifact(repo_root=repo_root, final_state=final_state)
    if audit_path is not None:
        print(f"\nVDD audit bundle written to: {audit_path}")
    
    print("\n" + "="*50)
    print("DRAFT VDD (Version Description Document)")
    print("="*50)
    print(f"RELEASE VERDICT   : {final_state['assessor_report'].get('final_decision')}")
    print(f"RATIONALE         : {final_state['assessor_report'].get('vdd_explanation')}")
    print("="*50)