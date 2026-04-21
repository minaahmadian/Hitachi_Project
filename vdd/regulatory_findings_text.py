"""Plain-text formatting for CEI EN 50128 per-rule PASS/FAIL/WARNING (terminal, file, Word)."""
from __future__ import annotations

from typing import Any


def format_regulatory_findings_plain(
    regulatory: dict[str, Any] | None,
    *,
    rationale_max: int = 240,
) -> str:
    """
    Human-readable breakdown of each rule evaluated in the last regulatory pass.

    PASS/FAIL here means keyword overlap between rule text and pipeline evidence blobs
    (not semantic proof of compliance).
    """
    if not isinstance(regulatory, dict):
        return ""
    raw = regulatory.get("findings")
    if not isinstance(raw, list) or not raw:
        raw = regulatory.get("top_findings") or []
    if not isinstance(raw, list) or not raw:
        return ""
    rac = max(40, int(rationale_max))

    def _bucket(status: str) -> list[dict[str, Any]]:
        return [f for f in raw if isinstance(f, dict) and str(f.get("status", "")).upper() == status]

    passes = _bucket("PASS")
    fails = _bucket("FAIL")
    warns = _bucket("WARNING")

    blocks: list[str] = []

    blocks.append(
        f"PASS ({len(passes)}): overlap between rule keywords and at least one evidence item "
        f"(matcher / derogation / auditor / detective)."
    )
    for i, f in enumerate(passes, 1):
        cid = f.get("clause_id") or f.get("rule_id") or "?"
        ev_ids = ", ".join(f.get("matched_evidence_ids") or []) or "(none)"
        rat = str(f.get("rationale") or "")[:rac]
        blocks.append(
            f"  {i}. [{cid}] modality={f.get('modality')} severity={f.get('severity')}\n"
            f"     matched_evidence_ids: {ev_ids}\n"
            f"     {rat}"
        )

    blocks.append(
        f"\nFAIL ({len(fails)}): mandatory (MUST/SHALL) rules with insufficient overlap — evidence gap / derogation."
    )
    for i, f in enumerate(fails, 1):
        cid = f.get("clause_id") or f.get("rule_id") or "?"
        rat = str(f.get("rationale") or "")[:rac]
        blocks.append(
            f"  {i}. [{cid}] modality={f.get('modality')} severity={f.get('severity')} "
            f"needs_derogation={f.get('needs_derogation')}\n"
            f"     {rat}"
        )

    if warns:
        blocks.append(f"\nWARNING ({len(warns)}): SHOULD rules without overlap.")
        for i, f in enumerate(warns, 1):
            cid = f.get("clause_id") or f.get("rule_id") or "?"
            rat = str(f.get("rationale") or "")[:rac]
            blocks.append(f"  {i}. [{cid}] modality={f.get('modality')}\n     {rat}")

    return "\n\n".join(blocks)
