from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

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


def normalize_regulation(pdf_path: Path, output_path: Path) -> list[ClauseRecord]:
    pages = _load_pdf_pages(pdf_path)
    body_pages = _collect_english_body(pages)
    lines = _normalize_lines(body_pages)
    clauses = _parse_clauses(lines, source_document=pdf_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(clause) for clause in clauses], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return clauses


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = repo_root / "data" / "CEI EN 50128.pdf"
    output_path = repo_root / "data" / "regulatory" / "cei_en_50128_clauses.json"

    clauses = normalize_regulation(pdf_path=pdf_path, output_path=output_path)
    print(f"Normalized {len(clauses)} clauses")
    print(f"Wrote: {output_path}")
    if clauses:
        print(f"First clause: {clauses[0].clause_id} - {clauses[0].title}")
        print(f"Last clause: {clauses[-1].clause_id} - {clauses[-1].title}")


if __name__ == "__main__":
    main()
