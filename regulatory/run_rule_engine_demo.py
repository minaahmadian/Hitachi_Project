from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regulatory.rule_engine import EvidenceItem, RegulatoryRuleEngine


def main() -> None:
    rules_path = REPO_ROOT / "data" / "regulatory" / "cei_en_50128_rules.json"

    engine = RegulatoryRuleEngine(rules_path=rules_path)

    anomaly = "failed test release justification"
    evidence = [
        EvidenceItem(
            evidence_id="email-001",
            source_type="email",
            text=(
                "The MQTT integration test is failing because serviceAccount permissions are missing. "
                "A manual workaround was applied in manifests and release is requested for timeline reasons."
            ),
        ),
        EvidenceItem(
            evidence_id="test-log-001",
            source_type="test_log",
            text=(
                "Software Integration Test Report: test case MQTT-INT-07 failed in CI; "
                "no updated verification report attached in release package."
            ),
        ),
        EvidenceItem(
            evidence_id="release-note-001",
            source_type="release_note",
            text=(
                "Release note includes known discrepancy on MQTT test. "
                "Temporary mitigation documented; formal approval pending."
            ),
        ),
    ]

    summary = engine.evaluate(anomaly_text=anomaly, evidence=evidence, top_k_rules=20)

    print("=== Deterministic Rule Engine Demo ===")
    print(f"Anomaly: {anomaly}")
    print(
        f"Rules checked={summary.total_rules_checked} "
        f"pass={summary.passed} fail={summary.failed} warning={summary.warning} "
        f"derogation_needed={summary.derogation_needed}"
    )
    print("\nTop findings:")
    for finding in summary.findings[:8]:
        print(
            f"- {finding.rule_id} clause={finding.clause_id} modality={finding.modality} "
            f"status={finding.status} derogation={finding.needs_derogation}"
        )
        print(f"  matched={finding.matched_evidence_ids}")
        print(f"  rationale={finding.rationale}")


if __name__ == "__main__":
    main()
