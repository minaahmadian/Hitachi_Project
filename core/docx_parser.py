import os
from importlib import import_module
from typing import TypedDict


class HeadingItem(TypedDict):
    level: int
    text: str


class ParsedDocx(TypedDict):
    title: str
    headings: list[HeadingItem]
    tables: list[list[str]]
    paragraphs: list[str]
    full_text: str


def _empty_result() -> ParsedDocx:
    return {
        "title": "",
        "headings": [],
        "tables": [],
        "paragraphs": [],
        "full_text": "",
    }


def _extract_heading_level(style_name: str) -> int | None:
    if not style_name:
        return None

    style_name_lower = style_name.strip().lower()
    if not style_name_lower.startswith("heading"):
        return None

    suffix = style_name_lower.replace("heading", "", 1).strip()
    if suffix.isdigit():
        return int(suffix)

    return None


def parse_docx(file_path: str) -> ParsedDocx:
    """Parse a .docx file and return structured document content.

    Returns an empty structure when the file does not exist.
    """
    result = _empty_result()

    if not os.path.exists(file_path):
        return result

    try:
        document_factory = import_module("docx").Document
        document = document_factory(file_path)
    except Exception:
        return result

    paragraphs: list[str] = []
    headings: list[HeadingItem] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        paragraphs.append(text)

        heading_level = _extract_heading_level(paragraph.style.name if paragraph.style else "")
        if heading_level is not None:
            headings.append({"level": heading_level, "text": text})

    tables: list[list[str]] = []
    for table in document.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            tables.append(row_cells)

    title = headings[0]["text"] if headings else os.path.splitext(os.path.basename(file_path))[0]

    table_text_lines = [" | ".join(row) for row in tables]

    result["title"] = title
    result["headings"] = headings
    result["tables"] = tables
    result["paragraphs"] = paragraphs
    result["full_text"] = "\n".join(paragraphs + table_text_lines)

    return result