#!/usr/bin/env python3
"""
Compare a pipeline audit JSON (from VDD_AUDIT=1) or any compatible snapshot
against a golden scenario file.

Usage (from repo root):
  export VDD_AUDIT=1
  python main.py
  python scripts/evaluate_against_golden.py --audit output/vdd_last_run.json \\
      --golden data/eval/golden/scenario_baseline.json

Exit code 0 if all deterministic expectations match; 1 otherwise.
Soft expectations are reported but do not fail the exit code by default.
Use --strict-soft to fail on soft mismatches too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _get_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _norm(v: Any) -> str:
    return str(v).strip().upper() if v is not None else ""


def _anomaly_requirement_ids_from_audit(audit: dict[str, Any]) -> set[str]:
    """Uppercased requirement_id values attached to matcher anomalies."""
    mr = audit.get("matcher_report")
    if not isinstance(mr, dict):
        return set()
    out: set[str] = set()
    for item in mr.get("anomalies") or []:
        if not isinstance(item, dict):
            continue
        rid = item.get("requirement_id")
        if rid is not None and str(rid).strip():
            out.add(str(rid).strip().upper())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare audit JSON to golden expectations.")
    ap.add_argument("--audit", required=True, type=Path, help="Path to vdd_last_run.json or audit export")
    ap.add_argument("--golden", required=True, type=Path, help="Golden scenario JSON")
    ap.add_argument(
        "--strict-soft",
        action="store_true",
        help="Exit non-zero if any soft expectation mismatches",
    )
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    gold = json.loads(args.golden.read_text(encoding="utf-8"))

    det = gold.get("expected_deterministic") or {}
    soft = gold.get("expected_soft") or {}

    if not isinstance(det, dict):
        det = {}
    if not isinstance(soft, dict):
        soft = {}

    def run_block(label: str, expected: dict[str, Any], *, fail_on_fail: bool) -> tuple[int, int]:
        ok = 0
        bad = 0
        print(f"\n=== {label} ===")
        for path, exp in expected.items():
            got = _get_nested(audit, path)
            exp_s = _norm(exp)
            got_s = _norm(got)
            match = exp_s == got_s or (exp_s == "" and got_s == "")
            if match:
                ok += 1
                print(f"  OK   {path!r}: {got!r}")
            else:
                bad += 1
                print(f"  MISS {path!r}: expected {exp!r} got {got!r}")
        return ok, bad

    d_ok, d_bad = run_block("deterministic", det, fail_on_fail=True)
    s_ok, s_bad = run_block("soft (LLM / non-frozen)", soft, fail_on_fail=args.strict_soft)

    arc = gold.get("anomaly_requirement_id_checks")
    a_ok, a_bad = 0, 0
    n_anomaly_checks = 0
    if isinstance(arc, dict):
        must_in = [str(x).strip().upper() for x in (arc.get("must_include") or []) if str(x).strip()]
        must_out = [str(x).strip().upper() for x in (arc.get("must_not_include") or []) if str(x).strip()]
        n_anomaly_checks = len(must_in) + len(must_out)
        seen = _anomaly_requirement_ids_from_audit(audit)
        print("\n=== anomaly requirement_id checks (matcher_report.anomalies) ===")
        for rid in must_in:
            if rid in seen:
                a_ok += 1
                print(f"  OK   must_include {rid!r} (present in anomaly requirement_ids)")
            else:
                a_bad += 1
                print(f"  MISS must_include {rid!r} — seen={sorted(seen)}")
        for rid in must_out:
            if rid not in seen:
                a_ok += 1
                print(f"  OK   must_not_include {rid!r} (absent)")
            else:
                a_bad += 1
                print(f"  MISS must_not_include {rid!r} — still present in anomalies")

    total_det = len(det) + n_anomaly_checks
    det_hits = d_ok + a_ok
    total_soft = len(soft)
    det_acc = (det_hits / total_det) if total_det else 1.0
    soft_acc = (s_ok / total_soft) if total_soft else 1.0

    print("\n--- Summary ---")
    print(
        f"  Deterministic + anomaly-id accuracy: {det_hits}/{total_det} = {det_acc:.2%}"
        if total_det
        else "  (no deterministic expectations)"
    )
    print(f"  Soft accuracy:         {s_ok}/{total_soft} = {soft_acc:.2%}" if total_soft else "  (no soft expectations)")

    if d_bad > 0 or a_bad > 0:
        print("\nDeterministic or anomaly-id mismatch — fix inputs or golden labels.")
        return 1
    if args.strict_soft and s_bad > 0:
        print("\nSoft mismatch (--strict-soft).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
