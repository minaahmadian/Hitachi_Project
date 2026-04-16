from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict

from pdfminer.high_level import extract_text

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional fallback import
    PdfReader = None


NOISE_PATTERNS = (
    r"RIPRODUZIONE SU LICENZA CEI AD ESCLUSIVO USO AZIENDALE",
    r"Copia concessa a .*? CEI-Comitato Elettrotecnico Italiano",
    r"^Norma Tecnica$",
    r"^NORMA TECNICA$",
    r"^CEI EN 50128(:2002-04)?$",
    r"^Pagina [ivx\d ]+(di \d+)?$",
    r"^-- \d+ of \d+ --$",
)

MAIN_CLAUSE_RE = re.compile(r"^(?P<id>\d+)\s+(?P<title>[A-Z][A-Z0-9 /\-(),]+)$")
SUBCLAUSE_RE = re.compile(r"^(?P<id>\d+(?:\.\d+)+)\s+(?P<title>.+)$")


@dataclass(slots=True)
class ClauseRecord:
    clause_id: str
    title: str
    text: str
    section: str
    page_start: int | None
    source_document: str


@dataclass(slots=True)
class CleanupReport:
    before_total: int
    after_total: int
    duplicate_ids_before: int
    duplicate_extra_rows_before: int
    duplicates_resolved: int
    merged_clause_ids: list[str]
    removed_invalid_titles: int
    removed_empty_text: int


def _strip_noise(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.replace("\t", " ").split()).strip()
        if not line:
            lines.append("")
            continue

        if any(re.search(pattern, line) for pattern in NOISE_PATTERNS):
            continue

        # Remove page marker remnants and decorative front matter.
        if line.startswith("© ") or line.startswith("Comitato Europeo di"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_page_number(page_text: str, page_index: int) -> int | None:
    match = re.search(r"--\s+(\d+)\s+of\s+\d+\s+--", page_text)
    if match:
        return int(match.group(1))
    return page_index + 1


def _load_pdf_pages(pdf_path: Path) -> list[tuple[int | None, str]]:
    if PdfReader is not None:
        try:
            reader = PdfReader(str(pdf_path))
            output: list[tuple[int | None, str]] = []
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_no = _extract_page_number(text, idx)
                output.append((page_no, _strip_noise(text)))
            return output
        except Exception:
            pass

    # Fallback for malformed/encrypted PDFs that PyPDF2 cannot open reliably.
    full_text = extract_text(str(pdf_path))
    raw_pages = re.split(r"\n\s*--\s+\d+\s+of\s+\d+\s+--\s*\n", full_text)
    output = []
    for idx, page_text in enumerate(raw_pages):
        if not page_text.strip():
            continue
        output.append((idx + 1, _strip_noise(page_text)))
    return output


def _collect_english_body(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """
    Keep only the useful English regulation body.

    Strategy:
    - start at the first occurrence of the English clause body
      (fallback extractor may not preserve '1 SCOPE' on one line)
    - drop known front-matter/index fragments
    - stop before annexes/bibliography if encountered
    """
    started = False
    collected: list[tuple[int | None, str]] = []

    for page_no, text in pages:
        if not text.strip():
            continue

        if not started:
            anchors = []
            for pattern in (
                r"3\.1\s+Assessment\s+Valutazione\s+Process of analysis",
                r"3\.1\s+Assessment\s+Process of analysis",
                r"3\.1\s+Assessment",
            ):
                match = re.search(pattern, text, flags=re.MULTILINE)
                if match:
                    anchors.append(match.start())
            anchors = [pos for pos in anchors if pos != -1]
            if not anchors:
                continue
            text = text[min(anchors) :]
            started = True

        if "ANNEX/ALLEGATO A" in text or "ANNEX A" in text or "BIBLIOGRAPHY" in text:
            text = (
                text.split("ANNEX/ALLEGATO A", 1)[0]
                .split("ANNEX A", 1)[0]
                .split("BIBLIOGRAPHY", 1)[0]
            )
            if text.strip():
                collected.append((page_no, text))
            break

        collected.append((page_no, text))

    return collected


def _normalize_lines(body_pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    normalized: list[tuple[int | None, str]] = []
    for page_no, page_text in body_pages:
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Skip obvious Italian-only headings that can reappear before English text.
            if stripped in {
                "CAMPO D’APPLICAZIONE",
                "RIFERIMENTI NORMATIVI",
                "DEFINIZIONI",
                "OBIETTIVI E CONFORMITÀ",
                "PERSONALE E RESPONSABILITÀ",
            }:
                continue

            normalized.append((page_no, stripped))
    return normalized


def _parse_clauses(lines: list[tuple[int | None, str]], source_document: str) -> list[ClauseRecord]:
    records: list[ClauseRecord] = []
    current_id: str | None = None
    current_title = ""
    current_section = ""
    current_page: int | None = None
    text_parts: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, current_section, current_page, text_parts
        if not current_id:
            return
        text = " ".join(text_parts).strip()
        if text:
            records.append(
                ClauseRecord(
                    clause_id=current_id,
                    title=current_title,
                    text=text,
                    section=current_section or current_title,
                    page_start=current_page,
                    source_document=source_document,
                )
            )
        text_parts = []

    idx = 0
    total = len(lines)
    while idx < total:
        page_no, line = lines[idx]
        main_match = MAIN_CLAUSE_RE.match(line)
        sub_match = SUBCLAUSE_RE.match(line)
        standalone_clause_id = re.fullmatch(r"\d+(?:\.\d+)+", line)
        standalone_main_id = re.fullmatch(r"\d+", line)

        if main_match and not "." in main_match.group("id"):
            flush()
            current_id = main_match.group("id")
            current_title = main_match.group("title").strip()
            current_section = current_title
            current_page = page_no
            idx += 1
            continue

        if sub_match:
            clause_id = sub_match.group("id").strip()
            title = sub_match.group("title").strip()

            # Ignore table-of-contents leader rows.
            if "...." in title:
                continue

            flush()
            current_id = clause_id
            current_title = title
            current_page = page_no
            idx += 1
            continue

        if standalone_clause_id and idx + 1 < total:
            next_page, next_line = lines[idx + 1]
            if next_page == page_no and next_line and not re.fullmatch(r"\d+(?:\.\d+)+", next_line):
                flush()
                current_id = line
                current_title = next_line
                current_page = page_no
                idx += 2
                continue

        if standalone_main_id and idx + 1 < total:
            next_page, next_line = lines[idx + 1]
            if next_page == page_no and re.fullmatch(r"[A-Z][A-Z0-9 /\-(),]+", next_line):
                flush()
                current_id = line
                current_title = next_line
                current_section = next_line
                current_page = page_no
                idx += 2
                continue

        if not current_id:
            idx += 1
            continue

        # Skip recurring non-body fragments.
        if line in {"CONTENTS", "INDICE", "INTRODUCTION"}:
            idx += 1
            continue
        if line.startswith("Publication") or line.startswith("Titolo"):
            idx += 1
            continue

        text_parts.append(line)
        idx += 1

    flush()
    return records


_DOT_LEADER_RE = re.compile(r"\.{5,}")
_ISOLATED_SYMBOL_LINE_RE = re.compile(r"^[^\w]{1,6}$")
_CLAUSE_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _looks_meaningful_title(title: str) -> bool:
    t = " ".join(title.split()).strip()
    if not t:
        return False
    if re.fullmatch(r"[\d\W_]+", t):
        return False
    if _DOT_LEADER_RE.search(t):
        return False
    if len(re.findall(r"[A-Za-z]", t)) < 2:
        return False
    return True


def _clean_line(line: str) -> str:
    line = _DOT_LEADER_RE.sub(" ", line)
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
    line = line.replace("", " ")
    line = " ".join(line.split()).strip()
    if not line:
        return ""
    if _ISOLATED_SYMBOL_LINE_RE.match(line):
        return ""
    # Remove typical OCR bullets/noise rows
    if line in {"♦", "•", ""}:
        return ""
    return line


def _extract_english_only(text: str) -> str:
    """
    Keep mostly-English lines for consistent downstream rule extraction.
    """
    kept: list[str] = []
    italian_markers = (
        " il ",
        " della ",
        " delle ",
        " deve ",
        " requisito ",
        " requisiti ",
        " documento ",
        " software ",
        " validazione ",
    )
    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        lower = f" {line.lower()} "

        # Keep lines with strong English signal.
        english_hits = sum(1 for token in (" shall ", " should ", " must ", " the ", " and ", " of ") if token in lower)
        italian_hits = sum(1 for token in italian_markers if token in lower)
        ascii_letters = sum(1 for ch in line if "a" <= ch.lower() <= "z")
        if ascii_letters < 8:
            continue
        if english_hits == 0 and italian_hits > 0:
            continue
        kept.append(line)
    return " ".join(kept).strip()


def _merge_clause_group(group: list[ClauseRecord]) -> ClauseRecord:
    """
    Merge duplicates by keeping the richest title/body and combining distinct text blocks.
    """
    # Prefer meaningful longest title.
    titles = [item.title.strip() for item in group if _looks_meaningful_title(item.title)]
    title = max(titles, key=len) if titles else max((item.title.strip() for item in group), key=len)

    # Use section from best-matching record.
    section = ""
    for item in group:
        if item.title.strip() == title and item.section.strip():
            section = item.section.strip()
            break
    if not section:
        section = title

    page_candidates = [item.page_start for item in group if item.page_start is not None]
    page_start = min(page_candidates) if page_candidates else None

    cleaned_parts: list[str] = []
    seen_parts: set[str] = set()
    for item in sorted(group, key=lambda x: len(x.text), reverse=True):
        cleaned = _extract_english_only(item.text)
        if not cleaned:
            continue
        if cleaned in seen_parts:
            continue
        seen_parts.add(cleaned)
        cleaned_parts.append(cleaned)

    merged_text = " ".join(cleaned_parts).strip()
    return ClauseRecord(
        clause_id=group[0].clause_id,
        title=title,
        text=merged_text,
        section=section,
        page_start=page_start,
        source_document=group[0].source_document,
    )


def _cleanup_clauses(raw_clauses: list[ClauseRecord]) -> tuple[list[ClauseRecord], CleanupReport]:
    grouped: dict[str, list[ClauseRecord]] = defaultdict(list)
    for clause in raw_clauses:
        cid = clause.clause_id.strip()
        if not _CLAUSE_ID_RE.match(cid):
            continue
        grouped[cid].append(
            ClauseRecord(
                clause_id=cid,
                title=" ".join(clause.title.split()).strip(),
                text=clause.text,
                section=" ".join(clause.section.split()).strip(),
                page_start=clause.page_start,
                source_document=clause.source_document,
            )
        )

    before_total = len(raw_clauses)
    duplicate_ids_before = sum(1 for _, rows in grouped.items() if len(rows) > 1)
    duplicate_extra_rows_before = sum(len(rows) - 1 for rows in grouped.values() if len(rows) > 1)

    merged_clause_ids: list[str] = []
    cleaned: list[ClauseRecord] = []
    removed_invalid_titles = 0
    removed_empty_text = 0

    for cid in sorted(grouped.keys(), key=lambda x: [int(p) for p in x.split(".")]):
        rows = grouped[cid]
        if len(rows) > 1:
            merged_clause_ids.append(cid)
        merged = _merge_clause_group(rows)

        if not _looks_meaningful_title(merged.title):
            removed_invalid_titles += 1
            continue
        if not merged.text.strip():
            removed_empty_text += 1
            continue
        cleaned.append(merged)

    report = CleanupReport(
        before_total=before_total,
        after_total=len(cleaned),
        duplicate_ids_before=duplicate_ids_before,
        duplicate_extra_rows_before=duplicate_extra_rows_before,
        duplicates_resolved=duplicate_ids_before,
        merged_clause_ids=merged_clause_ids,
        removed_invalid_titles=removed_invalid_titles,
        removed_empty_text=removed_empty_text,
    )
    return cleaned, report


def normalize_regulation(pdf_path: Path, output_path: Path) -> tuple[list[ClauseRecord], CleanupReport]:
    pages = _load_pdf_pages(pdf_path)
    body_pages = _collect_english_body(pages)
    lines = _normalize_lines(body_pages)
    raw_clauses = _parse_clauses(lines, source_document=pdf_path.name)
    clauses, report = _cleanup_clauses(raw_clauses)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(clause) for clause in clauses], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return clauses, report


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = repo_root / "data" / "CEI EN 50128.pdf"
    output_path = repo_root / "data" / "regulatory" / "cei_en_50128_clauses.json"

    clauses, report = normalize_regulation(pdf_path=pdf_path, output_path=output_path)
    print(f"Normalized {len(clauses)} clauses")
    print(f"Wrote: {output_path}")
    if clauses:
        print(f"First clause: {clauses[0].clause_id} - {clauses[0].title}")
        print(f"Last clause: {clauses[-1].clause_id} - {clauses[-1].title}")
    print(
        "Cleanup report:",
        json.dumps(
            {
                "before_total": report.before_total,
                "after_total": report.after_total,
                "duplicate_ids_before": report.duplicate_ids_before,
                "duplicate_extra_rows_before": report.duplicate_extra_rows_before,
                "duplicates_resolved": report.duplicates_resolved,
                "removed_invalid_titles": report.removed_invalid_titles,
                "removed_empty_text": report.removed_empty_text,
                "merged_clause_ids_sample": report.merged_clause_ids[:15],
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
