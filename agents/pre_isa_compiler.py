from __future__ import annotations

from core.state import GraphState
from traceability.pre_isa_report import build_pre_isa_report


def pre_isa_compiler_node(state: GraphState):
    print("Pre-ISA compiler: Building consolidated pre_isa_report for assessor and VDD templates...")

    report = build_pre_isa_report(state)
    print(f"   -> pre_isa_report.overall={report.get('overall')} (schema {report.get('schema_version')})")

    return {"pre_isa_report": report}
