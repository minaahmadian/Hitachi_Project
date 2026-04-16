from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.docx_parser import ParsedDocx


def load_requirements_csv(path: Path) -> list[dict[str, str]]:
    """Load customer / internal requirements trace rows (UTF-8)."""
    if not path.is_file():
        return []

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        for raw in reader:
            if not raw:
                continue
            row = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in raw.items() if k}
            req_id = row.get("requirement_id") or row.get("Requirement ID") or row.get("id") or ""
            if not req_id or req_id.startswith("#"):
                continue
            if "requirement_id" not in row and req_id:
                row["requirement_id"] = req_id
            rows.append(row)
    return rows


def load_test_logs_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _corpus_covers_requirement_ids(corpus: str, requirement_ids: Sequence[str]) -> bool:
    blob = corpus.lower()
    for rid in requirement_ids:
        token = str(rid).strip()
        if not token:
            continue
        if token.lower() not in blob:
            return False
    return True


def build_test_evidence_corpus(
    parsed_docx: ParsedDocx,
    repo_root: Path,
    *,
    requirement_ids: Sequence[str] | None = None,
) -> str:
    """
    Single searchable corpus for Phase 2: parsed Word body + flattened tables,
    optionally augmented with ``data/test_design_traceability.txt`` when the DOCX
    is thin **or** when known requirement IDs from the CSV do not appear in the DOCX text.
    """
    paragraphs = parsed_docx.get("paragraphs") or []
    tables = parsed_docx.get("tables") or []
    table_lines = [" | ".join(str(c) for c in row) for row in tables if row]
    parts: list[str] = []
    body = (parsed_docx.get("full_text") or "").strip()
    if body:
        parts.append(body)
    elif paragraphs:
        parts.append("\n".join(str(p) for p in paragraphs if str(p).strip()))
    if table_lines:
        parts.append("\n".join(table_lines))

    combined = "\n\n".join(p for p in parts if p).strip()
    min_chars = int(os.getenv("TRACEABILITY_DOCX_MIN_CHARS", "80") or "80")
    sidecar = repo_root / "data" / "test_design_traceability.txt"

    need_sidecar = len(combined) < min_chars
    if not need_sidecar and requirement_ids and sidecar.is_file():
        need_sidecar = not _corpus_covers_requirement_ids(combined, requirement_ids)

    if sidecar.is_file() and need_sidecar:
        extra = sidecar.read_text(encoding="utf-8")
        combined = f"{combined}\n\n--- sidecar: {sidecar.name} ---\n{extra}".strip()

    return combined


def load_authorization_text(path: Path) -> str:
    """Optional plain-text authorizations / waivers archive."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

