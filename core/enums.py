"""
Typed enums for safety-critical string values.

Why this module exists
----------------------
Across the pipeline, status/severity/verdict fields are compared as plain
strings (e.g. ``v.get("verdict") == "RED_FLAG"``). That is fragile in three
specific ways that matter for a SIL4-adjacent safety gate:

1. A typo in a producer (``"REDFLAG"``) silently disables the consumer check.
2. Two different keys can encode the same field (``severity`` vs
   ``matcher_severity``); a caller that reads only one silently misses the
   other.
3. Renaming a value requires grepping every string occurrence — a missed
   occurrence is invisible at runtime and manifests only as "safety gate did
   nothing".

These enums fix all three by:

- Subclassing ``str`` (so ``Severity.HIGH == "HIGH"`` stays true and JSON
  round-trips are identical — no artifact schema change).
- Exposing a ``normalize(value)`` classmethod that accepts ``str | Enum |
  None``, is case-insensitive, and returns an ``UNKNOWN`` sentinel rather
  than raising for unrecognised input.
- Providing canonical priority orderings (``Severity.order``,
  ``Verdict.order``) so every safety comparison uses the same ranking.

Usage
-----
>>> Severity.normalize("high") is Severity.HIGH
True
>>> Verdict.normalize(None) is Verdict.UNKNOWN
True
>>> Severity.HIGH == "HIGH"              # str-subclass: drop-in compatible
True
>>> json.dumps(Severity.HIGH)            # serialises as the plain string
'"HIGH"'

Safety guarantees proven by ``tests/test_enums.py``:

* Unknown or ``None`` input always maps to ``UNKNOWN`` — never raises.
* ``Severity.normalize`` and ``Verdict.normalize`` are case-insensitive.
* ``ReleaseDecision.normalize`` accepts ``"NO-GO"``, ``"NO_GO"``,
  ``"NOGO"``, and ``"no go"`` as equivalent.
* ``.value`` of every member equals the exact string used in current
  pipeline JSON output — so migrating internal comparisons to enums does
  not change any artifact on disk.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class _StrEnum(str, Enum):
    """Base class: str-backed Enum with case-insensitive ``normalize()``.

    Subclasses MUST declare an ``UNKNOWN`` member; it is returned for
    unrecognised input so the safety path never crashes on bad data.
    """

    @classmethod
    def normalize(cls, value: Any) -> "_StrEnum":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls._unknown_member()
        try:
            s = str(value).strip()
        except Exception:
            return cls._unknown_member()
        if not s:
            return cls._unknown_member()
        s_up = s.upper()
        for member in cls:
            if member.value.upper() == s_up:
                return member
        return cls._unknown_member()

    @classmethod
    def _unknown_member(cls) -> "_StrEnum":
        unknown = getattr(cls, "UNKNOWN", None)
        if unknown is not None:
            return unknown
        # Fallback: last-declared member. All concrete enums below declare
        # UNKNOWN explicitly, so this path is defensive only.
        return list(cls)[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly severity (HIGH > MEDIUM > LOW)
# ─────────────────────────────────────────────────────────────────────────────

class Severity(_StrEnum):
    """Severity of a traceability anomaly. Order: HIGH (0) → UNKNOWN (3)."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"

    @property
    def order(self) -> int:
        return _SEVERITY_ORDER[self]


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.HIGH: 0,
    Severity.MEDIUM: 1,
    Severity.LOW: 2,
    Severity.UNKNOWN: 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-anomaly verdict (RED_FLAG is worst)
# ─────────────────────────────────────────────────────────────────────────────

class Verdict(_StrEnum):
    """Per-anomaly verdict produced by the pre-ISA compiler."""

    RED_FLAG = "RED_FLAG"
    REVIEW = "REVIEW"
    JUSTIFICATION_SIGNALS = "JUSTIFICATION_SIGNALS"
    TRACKED = "TRACKED"
    UNKNOWN = "UNKNOWN"

    @property
    def order(self) -> int:
        return _VERDICT_ORDER[self]


_VERDICT_ORDER: dict[Verdict, int] = {
    Verdict.RED_FLAG: 0,
    Verdict.REVIEW: 1,
    Verdict.JUSTIFICATION_SIGNALS: 2,
    Verdict.TRACKED: 3,
    Verdict.UNKNOWN: 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Upstream status gates (matcher / regulatory / pre-ISA overall)
# ─────────────────────────────────────────────────────────────────────────────

class StatusGate(_StrEnum):
    """Status emitted by matcher, regulatory, and pre-ISA ``overall`` fields."""

    CLEAR = "CLEAR"
    WARNING = "WARNING"
    RED_FLAG = "RED_FLAG"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Context Detective triage status
# ─────────────────────────────────────────────────────────────────────────────

class DetectiveStatus(_StrEnum):
    CLEAR = "CLEAR"
    SUSPICIOUS = "SUSPICIOUS"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Formal Auditor overall assessment
# ─────────────────────────────────────────────────────────────────────────────

class AuditorAssessment(_StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Derogation language scan overall
# ─────────────────────────────────────────────────────────────────────────────

class DerogationOverall(_StrEnum):
    NO_SIGNALS = "NO_SIGNALS"
    WEAK_SIGNALS = "WEAK_SIGNALS"
    STRONG_SIGNALS = "STRONG_SIGNALS"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Derogation hit strength (lowercase — matches existing JSON)
# ─────────────────────────────────────────────────────────────────────────────

class DerogationStrength(_StrEnum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Final release decision
# ─────────────────────────────────────────────────────────────────────────────

class ReleaseDecision(_StrEnum):
    """Final GO / NO-GO output from the Lead Assessor.

    ``normalize`` accepts every common spelling variant so a typo in a
    downstream consumer (``"NO_GO"``, ``"NOGO"``, ``"no go"``) cannot
    misclassify the decision.
    """

    GO = "GO"
    NO_GO = "NO-GO"  # Canonical JSON value keeps the hyphen (legacy).
    CONDITIONAL_GO = "CONDITIONAL_GO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def normalize(cls, value: Any) -> "ReleaseDecision":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        try:
            s = str(value).strip().upper()
        except Exception:
            return cls.UNKNOWN
        # Normalize separators: "NO GO", "NO_GO", "NOGO", "NO-GO" all map the same.
        canon = s.replace(" ", "").replace("_", "").replace("-", "")
        if canon == "NOGO":
            return cls.NO_GO
        if canon == "GO":
            return cls.GO
        if canon == "CONDITIONALGO":
            return cls.CONDITIONAL_GO
        return cls.UNKNOWN


__all__ = [
    "Severity",
    "Verdict",
    "StatusGate",
    "DetectiveStatus",
    "AuditorAssessment",
    "DerogationOverall",
    "DerogationStrength",
    "ReleaseDecision",
]
