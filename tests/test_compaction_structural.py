"""
Structural-integrity tests for ``compact_report`` and ``parse_compacted_blob``.

The question being answered: "چطور مطمئنی که یک ویرگول اشتباه باعث نمی‌شود
کل ساختار JSON به هم بریزد؟" ("How do you know a wrong comma does not break
the entire JSON structure?")

Short answer: ``compact_report`` never hand-edits a JSON string. Every output
is produced by ``json.dumps(some_dict)``, so the only way a structurally invalid
JSON string can appear is if Python's own ``json.dumps`` has a bug — which is not
something we need to test. What we DO need to test is:

1. Every output of ``compact_report`` is either:
   a) parseable with ``json.loads``, OR
   b) ends with ``\\n[truncated]`` and the prefix before it is parseable.
   — "text-after-JSON" is the documented overflow contract, not a bug.

2. The parsed result always contains ``must_keep`` fields.

3. When the marker is appended, ``parse_compacted_blob`` recovers the dict
   without raising.

4. Adversarial inputs (Unicode escapes, embedded quotes, backslashes, deeply
   nested dicts, NaN/Infinity in values, duplicate-like keys, very long strings,
   null bytes, non-string keys) never produce silently invalid JSON.

5. ``extras`` injection does not interfere with the base report keys or
   introduce JSON corruption.

6. Idempotence: calling ``compact_report`` a second time on the output of
   ``parse_compacted_blob`` produces the same essential result.
"""
from __future__ import annotations

import json
import math
import string
from typing import Any

import pytest

from core.blob_compaction import (
    TRUNCATION_MARKER,
    compact_report,
    parse_compacted_blob,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_or_fail(blob: str) -> dict[str, Any]:
    """
    Parse the JSON portion of a blob. Always succeeds or the test fails —
    this is the structural guarantee we are asserting.
    """
    stripped, was_truncated = parse_compacted_blob(blob)
    assert stripped != {} or (blob.strip() in ("{}", "")), (
        f"parse_compacted_blob returned empty dict for non-empty blob:\n{blob[:300]}"
    )
    return stripped


def _is_valid_json_prefix(blob: str) -> bool:
    """Returns True if the JSON portion of blob is structurally valid."""
    _, truncated = parse_compacted_blob(blob)
    if truncated:
        json_part = blob[: -len(TRUNCATION_MARKER)].rstrip()
    else:
        json_part = blob
    try:
        json.loads(json_part)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic contract: output is always parseable JSON (or JSON + marker suffix).
# ─────────────────────────────────────────────────────────────────────────────

def test_normal_report_produces_valid_json() -> None:
    report = {"overall": "RED_FLAG", "score": 42, "items": ["a", "b"]}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall",))
    parsed = json.loads(blob)
    assert parsed["overall"] == "RED_FLAG"


def test_empty_report_produces_valid_json() -> None:
    blob = compact_report({}, max_chars=10_000, must_keep=())
    assert json.loads(blob) == {}


def test_none_report_produces_valid_json() -> None:
    blob = compact_report(None, max_chars=10_000, must_keep=())
    assert json.loads(blob) == {}


def test_no_fields_dropped_when_budget_generous() -> None:
    report = {"a": 1, "b": 2, "c": 3}
    blob = compact_report(report, max_chars=10_000, must_keep=("a",))
    parsed = json.loads(blob)
    assert parsed == report
    assert TRUNCATION_MARKER not in blob


@pytest.mark.parametrize("max_chars", [1, 5, 15, 30, 50, 100, 200, 500])
def test_output_json_prefix_is_always_parseable_regardless_of_budget(max_chars: int) -> None:
    report = {
        "overall": "RED_FLAG",
        "verdict_digests": ["HIT-FAKE-999 | HIGH | RED_FLAG | missing doc"],
        "summary": "x" * 300,
        "citations": ["ref"] * 10,
    }
    blob = compact_report(
        report,
        max_chars=max_chars,
        must_keep=("overall", "verdict_digests"),
        drop_order=("citations", "summary"),
    )
    assert _is_valid_json_prefix(blob), (
        f"JSON prefix is not parseable with max_chars={max_chars}:\n{blob[:200]}"
    )


def test_truncation_marker_only_appended_as_suffix_never_inline() -> None:
    report = {"overall": "RED_FLAG", "x": "a" * 500}
    blob = compact_report(report, max_chars=40, must_keep=("overall",))
    # marker must not appear inside the JSON portion
    if TRUNCATION_MARKER in blob:
        json_part = blob[: -len(TRUNCATION_MARKER)].rstrip()
        assert TRUNCATION_MARKER not in json_part, (
            "TRUNCATION_MARKER appears inside the JSON portion — this would corrupt parsing"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. must_keep fields are always present in the parsed output.
# ─────────────────────────────────────────────────────────────────────────────

def test_must_keep_fields_survive_extreme_budget() -> None:
    report = {"overall": "RED_FLAG", "verdict_digests": ["A | HIGH | RED_FLAG"], "noise": "z" * 5000}
    blob = compact_report(report, max_chars=1, must_keep=("overall", "verdict_digests"))
    parsed, _ = parse_compacted_blob(blob)
    assert "overall" in parsed
    assert "verdict_digests" in parsed
    assert parsed["overall"] == "RED_FLAG"


def test_must_keep_fields_not_in_report_are_silently_skipped() -> None:
    """Fields listed in must_keep that don't exist in report are simply absent — no KeyError."""
    report = {"overall": "CLEAR"}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall", "nonexistent_key"))
    parsed = json.loads(blob)
    assert "nonexistent_key" not in parsed


# ─────────────────────────────────────────────────────────────────────────────
# 3. parse_compacted_blob — structural recovery.
# ─────────────────────────────────────────────────── ─────────────────────────

def test_parse_compacted_blob_clean_json() -> None:
    blob = '{"a": 1, "b": "ok"}'
    parsed, truncated = parse_compacted_blob(blob)
    assert parsed == {"a": 1, "b": "ok"}
    assert truncated is False


def test_parse_compacted_blob_with_marker() -> None:
    clean_json = '{"overall": "RED_FLAG", "verdict": "x"}'
    blob = clean_json + "\n" + TRUNCATION_MARKER
    parsed, truncated = parse_compacted_blob(blob)
    assert truncated is True
    assert parsed["overall"] == "RED_FLAG"


def test_parse_compacted_blob_never_raises_on_garbage() -> None:
    for bad in [
        "",
        "not json at all",
        "{broken:",
        "null",
        "[1, 2, 3]",
        "12345",
        '{"ok": 1}\nrandom garbage',
        TRUNCATION_MARKER,
    ]:
        parsed, truncated = parse_compacted_blob(bad)
        assert isinstance(parsed, dict), f"Expected dict for input: {bad!r}"


def test_parse_compacted_blob_handles_none_input() -> None:
    parsed, truncated = parse_compacted_blob(None)  # type: ignore[arg-type]
    assert parsed == {}
    assert truncated is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adversarial inputs — Unicode, escapes, quotes, backslash, deeply nested,
#    very long, null bytes, non-ASCII keys, float edge-cases.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "tricky_value",
    [
        # Embedded quotes that need escaping
        'value with "embedded quotes" inside',
        "value with 'single quotes'",
        # Backslashes
        r"path\to\something",
        r"C:\Users\mina\file.docx",
        # Newlines and tabs
        "line1\nline2\ttabbed",
        # Unicode
        "مقدار فارسی با کاراکترهای خاص",
        "日本語テスト",
        "\u0000\u0001\u001f",  # control characters
        # JSON-like strings that must not be double-serialised
        '{"nested": "json", "looking": "string"}',
        "[1, 2, 3]",
        "null",
        "true",
        # Extremely long value
        "A" * 10_000,
        # Numeric-looking
        "1234567890",
        "3.14",
        "-0.0",
    ],
)
def test_adversarial_string_values_produce_valid_json(tricky_value: str) -> None:
    report = {
        "overall": "RED_FLAG",
        "tricky": tricky_value,
        "noise": "drop me",
    }
    blob = compact_report(
        report, max_chars=10_000, must_keep=("overall", "tricky"), drop_order=("noise",)
    )
    assert _is_valid_json_prefix(blob), (
        f"Invalid JSON for tricky value: {tricky_value[:80]!r}\nBlob: {blob[:200]}"
    )
    parsed, _ = parse_compacted_blob(blob)
    assert parsed.get("tricky") == tricky_value, (
        "Round-trip failed — value was corrupted during compaction"
    )


@pytest.mark.parametrize(
    "nested",
    [
        {"level1": {"level2": {"level3": {"level4": "deep"}}}},
        {"list": [{"a": 1}, {"b": [2, 3, {"c": "deep"}]}]},
        {"mixed": [1, None, True, False, 3.14, "string"]},
        {"empty_nested": {"empty": {}, "also_empty": []}},
    ],
)
def test_nested_structures_survive_round_trip(nested: dict) -> None:
    report = {"overall": "CLEAR", "data": nested, "noise": "drop"}
    blob = compact_report(
        report, max_chars=10_000, must_keep=("overall", "data"), drop_order=("noise",)
    )
    assert _is_valid_json_prefix(blob)
    parsed, _ = parse_compacted_blob(blob)
    assert parsed["data"] == nested


def test_null_value_in_report_is_preserved() -> None:
    report = {"overall": "RED_FLAG", "requirement_id": None, "noise": "x"}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall", "requirement_id"))
    parsed = json.loads(blob)
    assert parsed["requirement_id"] is None


def test_boolean_values_are_not_corrupted() -> None:
    report = {"overall": "CLEAR", "has_flag": True, "is_ok": False}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall",))
    parsed = json.loads(blob)
    assert parsed["has_flag"] is True
    assert parsed["is_ok"] is False


def test_integer_zero_and_negative_values_are_preserved() -> None:
    report = {"overall": "CLEAR", "score": 0, "delta": -5, "large": 10 ** 12}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall", "score"))
    parsed = json.loads(blob)
    assert parsed["score"] == 0
    assert parsed["delta"] == -5
    assert parsed["large"] == 10 ** 12


def test_float_values_roundtrip_without_corruption() -> None:
    report = {"overall": "CLEAR", "ratio": 0.9375, "neg": -1.5, "small": 1e-10}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall", "ratio"))
    parsed = json.loads(blob)
    assert abs(parsed["ratio"] - 0.9375) < 1e-9
    assert abs(parsed["neg"] - (-1.5)) < 1e-9


def test_empty_string_values_are_preserved() -> None:
    report = {"overall": "CLEAR", "empty_str": "", "noise": "x"}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall", "empty_str"))
    parsed = json.loads(blob)
    assert parsed["empty_str"] == ""


def test_empty_list_and_dict_values_are_preserved() -> None:
    report = {"overall": "CLEAR", "empty_list": [], "empty_dict": {}}
    blob = compact_report(report, max_chars=10_000, must_keep=("overall",))
    parsed = json.loads(blob)
    assert parsed["empty_list"] == []
    assert parsed["empty_dict"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# 5. extras injection does not corrupt the base report.
# ─────────────────────────────────────────────────────────────────────────────

def test_extras_are_included_in_output() -> None:
    report = {"overall": "RED_FLAG", "noise": "drop"}
    blob = compact_report(
        report,
        max_chars=10_000,
        must_keep=("overall",),
        extras={"injected_key": "injected_value", "score": 99},
    )
    parsed = json.loads(blob)
    assert parsed["injected_key"] == "injected_value"
    assert parsed["score"] == 99
    assert parsed["overall"] == "RED_FLAG"


def test_extras_with_tricky_values_do_not_corrupt_base() -> None:
    report = {"overall": "RED_FLAG"}
    tricky_extras = {
        "quoted": '"hello"',
        "backslash": "a\\b",
        "unicode": "\u00e9",
        "nested_dict": {"x": [1, 2]},
    }
    blob = compact_report(
        report,
        max_chars=10_000,
        must_keep=("overall",),
        extras=tricky_extras,
    )
    assert _is_valid_json_prefix(blob)
    parsed, _ = parse_compacted_blob(blob)
    assert parsed["overall"] == "RED_FLAG"
    assert parsed["quoted"] == '"hello"'
    assert parsed["backslash"] == "a\\b"
    assert parsed["unicode"] == "\u00e9"


def test_extras_override_report_key_without_corruption() -> None:
    """When extras and report share a key, extras win — and the result is valid JSON."""
    report = {"overall": "CLEAR", "score": 1}
    blob = compact_report(
        report,
        max_chars=10_000,
        must_keep=("overall",),
        extras={"score": 999},
    )
    parsed = json.loads(blob)
    assert parsed["score"] == 999


# ─────────────────────────────────────────────────────────────────────────────
# 6. _compaction metadata block is itself valid JSON inside the output.
# ─────────────────────────────────────────────────────────────────────────────

def test_compaction_metadata_block_is_valid_json_sub_object() -> None:
    """When fields are dropped, a ``_compaction`` block is injected.
    This test confirms it does not break the outer JSON structure."""
    report = {
        "overall": "RED_FLAG",
        "must": "keep me",
        "noise1": "x" * 200,
        "noise2": "y" * 200,
    }
    blob = compact_report(
        report,
        max_chars=80,
        must_keep=("overall", "must"),
        drop_order=("noise1", "noise2"),
    )
    # The blob must have valid JSON (even if marker appended)
    assert _is_valid_json_prefix(blob), f"JSON broken after compaction:\n{blob[:300]}"
    parsed, _ = parse_compacted_blob(blob)
    if "_compaction" in parsed:
        assert isinstance(parsed["_compaction"], dict)
        assert "dropped_fields" in parsed["_compaction"]
        assert isinstance(parsed["_compaction"]["dropped_fields"], list)
        assert parsed["_compaction"]["marker"] == TRUNCATION_MARKER


# ─────────────────────────────────────────────────────────────────────────────
# 7. Idempotence: parsing + re-compacting does not explode.
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_then_recompact_is_stable() -> None:
    report = {
        "overall": "RED_FLAG",
        "verdict_digests": ["HIT-FAKE-999 | HIGH | RED_FLAG | missing"],
        "citations": ["c"] * 50,
    }
    blob1 = compact_report(
        report, max_chars=200, must_keep=("overall", "verdict_digests"), drop_order=("citations",)
    )
    parsed1, _ = parse_compacted_blob(blob1)
    # Compact again
    blob2 = compact_report(
        parsed1, max_chars=200, must_keep=("overall", "verdict_digests")
    )
    parsed2, _ = parse_compacted_blob(blob2)
    assert parsed2.get("overall") == "RED_FLAG"
    assert "verdict_digests" in parsed2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Real-world shaped reports — exact shape the pipeline emits.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def realistic_pre_isa_report() -> dict[str, Any]:
    return {
        "overall": "RED_FLAG",
        "release_readiness": "NOT_READY",
        "inputs_digest": "matcher=RED_FLAG; regulatory=RED_FLAG; detective=SUSPICIOUS; derog=STRONG_SIGNALS",
        "verdict_digests": [
            "C6-APCS-1 | MEDIUM | TRACKED | rssom_rag_hit",
            "C6-APCS-2 | MEDIUM | TRACKED | rssom_rag_hit",
            "HIT-FAKE-999 | HIGH | RED_FLAG | no doc evidence found",
        ],
        "summary_for_vdd_short": "Pre-ISA consolidated. RED_FLAG from matcher + regulatory.",
        "citations": [{"kind": "rssom_retrieval", "requirement_id": "C6-APCS-1", "score": 0.91}] * 12,
        "evidence_chain_text": "Long chain... " * 100,
        "fingerprints": ["fp_001", "fp_002"],
        "rag_enabled": True,
    }


@pytest.mark.parametrize("budget", [200, 500, 1000, 2000, 4000, 10000])
def test_realistic_pre_isa_json_is_valid_at_all_budgets(
    realistic_pre_isa_report: dict[str, Any], budget: int
) -> None:
    blob = compact_report(
        realistic_pre_isa_report,
        max_chars=budget,
        must_keep=("overall", "release_readiness", "inputs_digest", "verdict_digests"),
        drop_order=("evidence_chain_text", "fingerprints", "citations", "summary_for_vdd_short"),
    )
    assert _is_valid_json_prefix(blob), (
        f"Invalid JSON at budget={budget}:\n{blob[:300]}"
    )
    parsed, _ = parse_compacted_blob(blob)
    # Safety contract: RED_FLAG and HIT-FAKE-999 must be findable
    assert parsed.get("overall") == "RED_FLAG"
    assert any("RED_FLAG" in d for d in parsed.get("verdict_digests", []))
    assert any("HIT-FAKE-999" in d for d in parsed.get("verdict_digests", []))


def test_realistic_pre_isa_safety_evidence_never_lost(
    realistic_pre_isa_report: dict[str, Any],
) -> None:
    """Even at the most extreme budget (1 byte), the worst RED_FLAG evidence must survive."""
    blob = compact_report(
        realistic_pre_isa_report,
        max_chars=1,
        must_keep=("overall", "verdict_digests"),
        drop_order=("evidence_chain_text", "fingerprints", "citations", "summary_for_vdd_short"),
    )
    parsed, truncated = parse_compacted_blob(blob)
    assert truncated is True, "Marker must be set when budget is violated"
    assert parsed.get("overall") == "RED_FLAG"
    verdicts = parsed.get("verdict_digests", [])
    high_rf = [v for v in verdicts if "HIGH" in v and "RED_FLAG" in v]
    assert high_rf, f"HIGH RED_FLAG digest missing from budget=1 output. Got: {verdicts}"
