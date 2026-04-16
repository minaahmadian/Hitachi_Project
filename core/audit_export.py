from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.state import GraphState


def build_vdd_audit_payload(final_state: GraphState) -> dict[str, Any]:
    """Structured, JSON-serializable snapshot of graph outputs for audit / CI."""
    dc = final_state.get("docx_content")
    docx_meta: dict[str, Any] = {}
    if isinstance(dc, dict):
        docx_meta = {
            "title": dc.get("title", ""),
            "headings_count": len(dc.get("headings") or []),
            "paragraphs_count": len(dc.get("paragraphs") or []),
            "tables_count": len(dc.get("tables") or []),
        }

    raw_emails = final_state.get("email_threads") or ""
    email_text = raw_emails if isinstance(raw_emails, str) else str(raw_emails)
    max_email = 12_000
    if len(email_text) > max_email:
        email_text = email_text[:max_email] + "\n...[truncated]"

    reqs = final_state.get("requirements_records") or []
    req_summary: dict[str, Any] = {"count": len(reqs) if isinstance(reqs, list) else 0}
    if isinstance(reqs, list) and reqs:
        req_summary["sample_ids"] = [
            str(r.get("requirement_id", "")).strip()
            for r in reqs[:12]
            if isinstance(r, dict) and str(r.get("requirement_id", "")).strip()
        ]

    logs = final_state.get("test_logs") if isinstance(final_state.get("test_logs"), dict) else {}
    logs_preview = json.dumps(logs, indent=2, ensure_ascii=False) if logs else "{}"
    max_logs = 6_000
    if len(logs_preview) > max_logs:
        logs_preview = logs_preview[:max_logs] + "\n...[truncated]"

    corpus = final_state.get("test_evidence_corpus") or ""
    corpus_preview = corpus if isinstance(corpus, str) else str(corpus)
    max_corpus = 8_000
    if len(corpus_preview) > max_corpus:
        corpus_preview = corpus_preview[:max_corpus] + "\n...[truncated]"

    auth = final_state.get("authorization_text") or ""
    auth_text = auth if isinstance(auth, str) else str(auth)
    if len(auth_text) > 4_000:
        auth_text = auth_text[:4_000] + "\n...[truncated]"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matcher_report": final_state.get("matcher_report"),
        "derogation_report": final_state.get("derogation_report"),
        "auditor_report": final_state.get("auditor_report"),
        "detective_report": final_state.get("detective_report"),
        "regulatory_report": final_state.get("regulatory_report"),
        "pre_isa_report": final_state.get("pre_isa_report"),
        "assessor_report": final_state.get("assessor_report"),
        "inputs": {
            "docx": docx_meta,
            "requirements": req_summary,
            "test_logs": logs_preview,
            "test_evidence_corpus_preview": corpus_preview,
            "authorization_text": auth_text,
            "email_threads": email_text,
        },
    }


def write_vdd_audit_artifact(*, repo_root: Path, final_state: GraphState) -> Path | None:
    """
    Persist audit bundle when enabled.

    - Set ``VDD_AUDIT_PATH`` to a relative (from repo root) or absolute JSON path.
    - Or set ``VDD_AUDIT=1`` to write ``<repo>/output/vdd_last_run.json``.
    """
    path_raw = os.getenv("VDD_AUDIT_PATH", "").strip()
    audit_flag = os.getenv("VDD_AUDIT", "0").strip().lower()

    if path_raw.lower() in {"-", "0", "false", "no", "off", "none"}:
        return None

    if path_raw:
        out_path = Path(path_raw).expanduser()
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
    elif audit_flag in {"1", "true", "yes", "on"}:
        out_path = (repo_root / "output" / "vdd_last_run.json").resolve()
    else:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_vdd_audit_payload(final_state)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
