"""
Unit tests for ``core.enums``.

These tests pin down the contract of every enum:

* ``.value`` matches the exact string the pipeline wrote to JSON before the
  refactor, so existing audit artifacts remain schema-compatible.
* ``.normalize()`` is case-insensitive and never raises — unknown / None
  input maps to ``UNKNOWN``.
* str-subclass behaviour: ``Severity.HIGH == "HIGH"`` and JSON round-trips
  preserve the raw string.
* ``Severity.order`` and ``Verdict.order`` encode the intended priority
  (HIGH first, RED_FLAG first).
* ``ReleaseDecision.normalize`` accepts every common spelling variant.
"""
from __future__ import annotations

import json

import pytest

from core.enums import (
    AuditorAssessment,
    DerogationOverall,
    DerogationStrength,
    DetectiveStatus,
    ReleaseDecision,
    Severity,
    StatusGate,
    Verdict,
)


# ─────────────────────────────────────────────────────────────────────────────
# str-subclass compatibility: these enums must be drop-in for plain strings.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "member, expected",
    [
        (Severity.HIGH, "HIGH"),
        (Severity.MEDIUM, "MEDIUM"),
        (Severity.LOW, "LOW"),
        (Severity.UNKNOWN, "UNKNOWN"),
        (Verdict.RED_FLAG, "RED_FLAG"),
        (Verdict.REVIEW, "REVIEW"),
        (Verdict.JUSTIFICATION_SIGNALS, "JUSTIFICATION_SIGNALS"),
        (Verdict.TRACKED, "TRACKED"),
        (StatusGate.CLEAR, "CLEAR"),
        (StatusGate.WARNING, "WARNING"),
        (StatusGate.RED_FLAG, "RED_FLAG"),
        (StatusGate.REVIEW_REQUIRED, "REVIEW_REQUIRED"),
        (DetectiveStatus.SUSPICIOUS, "SUSPICIOUS"),
        (AuditorAssessment.NON_COMPLIANT, "NON_COMPLIANT"),
        (DerogationOverall.STRONG_SIGNALS, "STRONG_SIGNALS"),
        (DerogationStrength.STRONG, "strong"),  # Lowercase — intentional, historical.
        (ReleaseDecision.GO, "GO"),
        (ReleaseDecision.NO_GO, "NO-GO"),
    ],
)
def test_value_matches_canonical_json_string(member, expected: str) -> None:
    assert member.value == expected
    assert member == expected
    assert isinstance(member, str)
    assert json.loads(json.dumps(member)) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Normalization: case-insensitive, None-safe, typo-safe.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value", ["HIGH", "high", "High", " HIGH ", "\thigh\n"]
)
def test_severity_normalize_is_case_insensitive_and_trim(value: str) -> None:
    assert Severity.normalize(value) is Severity.HIGH


@pytest.mark.parametrize(
    "value", [None, "", "  ", "not-a-severity", "HIGGH", "CRITICAL", 123, object()]
)
def test_severity_normalize_unknown_is_defensive(value) -> None:
    assert Severity.normalize(value) is Severity.UNKNOWN


def test_verdict_normalize_roundtrip() -> None:
    for member in Verdict:
        assert Verdict.normalize(member.value) is member
        assert Verdict.normalize(member.value.lower()) is member


def test_severity_order_is_high_first() -> None:
    ordered = sorted(Severity, key=lambda s: s.order)
    assert ordered[:3] == [Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    assert ordered[-1] is Severity.UNKNOWN


def test_verdict_order_is_red_flag_first() -> None:
    ordered = sorted(Verdict, key=lambda v: v.order)
    assert ordered[0] is Verdict.RED_FLAG
    assert ordered[-1] is Verdict.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# ReleaseDecision: all NO-GO spellings must map to the same enum.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spelling",
    ["NO-GO", "NO_GO", "NOGO", "no-go", "no go", "No Go", " no_go ", "nO-Go"],
)
def test_release_decision_normalize_accepts_all_no_go_spellings(spelling: str) -> None:
    assert ReleaseDecision.normalize(spelling) is ReleaseDecision.NO_GO


@pytest.mark.parametrize(
    "spelling",
    ["GO", "go", " go ", "Go"],
)
def test_release_decision_normalize_accepts_all_go_spellings(spelling: str) -> None:
    assert ReleaseDecision.normalize(spelling) is ReleaseDecision.GO


def test_release_decision_normalize_unknown_is_unknown() -> None:
    assert ReleaseDecision.normalize(None) is ReleaseDecision.UNKNOWN
    assert ReleaseDecision.normalize("MAYBE") is ReleaseDecision.UNKNOWN
    assert ReleaseDecision.normalize("") is ReleaseDecision.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# JSON round-trip: serialising a report that contains enums must produce the
# exact same bytes as the pre-refactor pipeline, so audit artifacts stay stable.
# ─────────────────────────────────────────────────────────────────────────────

def test_enum_values_survive_json_round_trip_identically() -> None:
    report = {
        "overall": StatusGate.RED_FLAG,
        "severity": Severity.HIGH,
        "verdict": Verdict.RED_FLAG,
        "detective": DetectiveStatus.SUSPICIOUS,
        "auditor": AuditorAssessment.PARTIAL,
        "derogation": DerogationOverall.STRONG_SIGNALS,
        "strength": DerogationStrength.STRONG,
        "final_decision": ReleaseDecision.NO_GO,
    }
    # Enums subclass str, so default json handler treats them as strings.
    payload = json.dumps(report)
    parsed = json.loads(payload)
    assert parsed == {
        "overall": "RED_FLAG",
        "severity": "HIGH",
        "verdict": "RED_FLAG",
        "detective": "SUSPICIOUS",
        "auditor": "PARTIAL",
        "derogation": "STRONG_SIGNALS",
        "strength": "strong",
        "final_decision": "NO-GO",
    }
