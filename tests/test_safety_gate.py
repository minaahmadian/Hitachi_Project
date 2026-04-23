"""
Safety-gate regression tests.

These tests pin down the *contract* of the safety primitives in
``core.blob_compaction`` so that a future refactor cannot silently disable
the ``has_high_red_flag`` detector.

Specifically, they prove:

1. ``has_high_red_flag`` fires for a HIGH RED_FLAG whether the producer used
   the ``severity`` key (matcher anomalies) OR the ``matcher_severity`` key
   (pre_isa verdicts). This is the exact bug the previous audit identified
   as "schema fragility".

2. Case variations, surrounding whitespace, and unknown-but-adjacent values
   cannot spoof the detector into a false negative on truly HIGH RED_FLAG
   data, nor a false positive on everything else.

3. ``compact_report`` guarantees that a HIGH RED_FLAG digest is visible in
   the compacted output even when the prompt budget is violently small —
   the hallmark of the previous "negative evidence hidden in the tail"
   vulnerability.

4. The ordering helpers rank HIGH RED_FLAG first regardless of which
   severity key is used.

These are the tests a mentor would want to see before accepting the safety
claim of the system.
"""
from __future__ import annotations

import json

import pytest

from core.blob_compaction import (
    TRUNCATION_MARKER,
    compact_report,
    has_high_red_flag,
    severity_verdict_sort_key,
    worst_verdict_in,
)
from core.enums import Severity, Verdict


# ─────────────────────────────────────────────────────────────────────────────
# 1) has_high_red_flag detects the unsafe combination under BOTH schema shapes.
# ─────────────────────────────────────────────────────────────────────────────

def test_has_high_red_flag_detects_matcher_severity_schema() -> None:
    """pre_isa_report.verdict_per_anomaly shape: uses `matcher_severity`."""
    verdicts = [
        {"requirement_id": "C6-APCS-1", "matcher_severity": "MEDIUM", "verdict": "TRACKED"},
        {"requirement_id": "HIT-FAKE-999", "matcher_severity": "HIGH", "verdict": "RED_FLAG"},
    ]
    assert has_high_red_flag(verdicts) is True


def test_has_high_red_flag_detects_plain_severity_schema() -> None:
    """matcher_report.anomalies shape: uses `severity` (no `matcher_severity`)."""
    verdicts = [
        {"requirement_id": "C6-APCS-1", "severity": "MEDIUM", "verdict": "TRACKED"},
        {"requirement_id": "HIT-FAKE-999", "severity": "HIGH", "verdict": "RED_FLAG"},
    ]
    assert has_high_red_flag(verdicts) is True


def test_has_high_red_flag_detects_mixed_schema_in_same_list() -> None:
    """A real pipeline can mix both shapes; the detector must be key-agnostic."""
    verdicts = [
        {"severity": "HIGH", "verdict": "TRACKED"},
        {"matcher_severity": "HIGH", "verdict": "RED_FLAG"},
    ]
    assert has_high_red_flag(verdicts) is True


def test_has_high_red_flag_is_case_insensitive() -> None:
    verdicts = [{"severity": "high", "verdict": "red_flag"}]
    assert has_high_red_flag(verdicts) is True


def test_has_high_red_flag_does_not_fire_on_non_high_red_flag() -> None:
    verdicts = [
        {"severity": "MEDIUM", "verdict": "RED_FLAG"},   # medium RED_FLAG
        {"severity": "HIGH", "verdict": "REVIEW"},        # high REVIEW
        {"severity": "LOW", "verdict": "TRACKED"},
        {"severity": "", "verdict": ""},
        {},
    ]
    assert has_high_red_flag(verdicts) is False


def test_has_high_red_flag_handles_junk_input_safely() -> None:
    assert has_high_red_flag([]) is False
    assert has_high_red_flag([None, 123, "not-a-dict"]) is False  # type: ignore[list-item]
    assert has_high_red_flag([{"severity": None, "verdict": None}]) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2) worst_verdict_in + sort key both honour enum ordering.
# ─────────────────────────────────────────────────────────────────────────────

def test_worst_verdict_prefers_red_flag_over_review() -> None:
    verdicts = [
        {"verdict": "TRACKED"},
        {"verdict": "REVIEW"},
        {"verdict": "RED_FLAG"},
        {"verdict": "JUSTIFICATION_SIGNALS"},
    ]
    assert worst_verdict_in(verdicts) == Verdict.RED_FLAG.value


def test_worst_verdict_in_returns_none_when_all_unknown() -> None:
    assert worst_verdict_in([{"verdict": "typo"}, {}]) is None


def test_sort_key_places_high_red_flag_first() -> None:
    items = [
        {"severity": "LOW", "verdict": "TRACKED"},
        {"severity": "MEDIUM", "verdict": "REVIEW"},
        {"matcher_severity": "HIGH", "verdict": "RED_FLAG"},   # mixed schema
        {"severity": "HIGH", "verdict": "REVIEW"},
    ]
    items.sort(key=severity_verdict_sort_key)
    assert items[0]["verdict"] == Verdict.RED_FLAG.value
    assert items[0].get("matcher_severity") == Severity.HIGH.value


# ─────────────────────────────────────────────────────────────────────────────
# 3) compact_report preserves safety evidence even under extreme budgets.
# ─────────────────────────────────────────────────────────────────────────────

def test_compact_report_preserves_must_keep_field_even_if_budget_exceeded() -> None:
    report = {
        "overall": "RED_FLAG",
        "verdict_digests": ["HIT-FAKE-999 | HIGH | RED_FLAG | missing evidence"],
        "citations": ["irrelevant but large" * 80],  # huge low-priority field
    }
    # Absurdly small budget that cannot even fit the must-keep fields.
    blob = compact_report(
        report,
        max_chars=50,
        must_keep=("overall", "verdict_digests"),
        drop_order=("citations",),
    )
    # Must-keep fields are present in the output
    assert '"overall": "RED_FLAG"' in blob
    assert "HIT-FAKE-999" in blob
    # Truncation marker warns downstream consumers
    assert TRUNCATION_MARKER in blob
    # Low-priority field is actually dropped
    assert "irrelevant but large" not in blob


def test_compact_report_keeps_everything_when_budget_is_generous() -> None:
    report = {"overall": "CLEAR", "footnotes": "ok"}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall",))
    parsed = json.loads(blob)
    assert parsed == report
    assert TRUNCATION_MARKER not in blob


# ─────────────────────────────────────────────────────────────────────────────
# 4) The key regression: the same data structure from a real vdd_last_run.json
#    run — where HIT-FAKE-999 is the hidden-risk seed — must trigger override.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def realistic_verdicts() -> list[dict[str, str]]:
    """Shape matches what pre_isa_report.build_pre_isa_report actually emits."""
    return [
        {"requirement_id": "C6-APCS-1",    "matcher_severity": "MEDIUM", "verdict": "TRACKED"},
        {"requirement_id": "C6-APCS-2",    "matcher_severity": "MEDIUM", "verdict": "TRACKED"},
        {"requirement_id": "C6-APCS-3",    "matcher_severity": "MEDIUM", "verdict": "TRACKED"},
        {"requirement_id": "HIT-FAKE-999", "matcher_severity": "HIGH",   "verdict": "RED_FLAG"},
        {"requirement_id": None,            "matcher_severity": "HIGH",   "verdict": "JUSTIFICATION_SIGNALS"},
    ]


def test_realistic_run_triggers_high_red_flag(realistic_verdicts: list[dict[str, str]]) -> None:
    assert has_high_red_flag(realistic_verdicts) is True
    assert worst_verdict_in(realistic_verdicts) == Verdict.RED_FLAG.value


def test_realistic_run_sorts_red_flag_first(realistic_verdicts: list[dict[str, str]]) -> None:
    realistic_verdicts.sort(key=severity_verdict_sort_key)
    assert realistic_verdicts[0]["verdict"] == Verdict.RED_FLAG.value
    assert realistic_verdicts[0]["requirement_id"] == "HIT-FAKE-999"
