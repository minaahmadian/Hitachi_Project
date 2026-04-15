from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email import policy
from email.message import Message
from email.parser import BytesParser, Parser
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import BytesIO
import importlib
from pathlib import Path
from typing_extensions import override

from .models import Document, DocumentMetadata, DocumentSource, EmailDocument


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    @override
    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(part.strip() for part in self._parts if part.strip())


def _normalize_datetime(value: str | datetime | None) -> datetime | None:
    if isinstance(value, str):
        try:
            parsed = parsedate_to_datetime(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _is_existing_path(source: str | Path) -> bool:
    try:
        return Path(source).exists()
    except (OSError, ValueError):
        return False


def _detect_encoding(sample: bytes) -> str:
    if not sample:
        return "utf-8"

    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff"):
        return "utf-16"

    try:
        chardet_module = importlib.import_module("chardet")
        detect_fn = getattr(chardet_module, "detect", None)
        if callable(detect_fn):
            result = detect_fn(sample)
            if isinstance(result, dict):
                encoding = result.get("encoding")
                if isinstance(encoding, str) and encoding:
                    return encoding
    except Exception:
        pass

    return "utf-8"


def _read_text_file_streaming(filepath: str | Path, chunk_size: int = 65536) -> str:
    path = Path(filepath)
    with path.open("rb") as file_obj:
        sample = file_obj.read(65536)
        encoding = _detect_encoding(sample)
        _ = file_obj.seek(0)

        chunks: list[str] = []
        while True:
            raw_chunk = file_obj.read(chunk_size)
            if not raw_chunk:
                break
            chunks.append(raw_chunk.decode(encoding, errors="replace"))
    return "".join(chunks)


def _extract_email_payload(msg: Message) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = (part.get_content_disposition() or "").lower()
            content_type = (part.get_content_type() or "").lower()

            if content_disposition == "attachment":
                filename = part.get_filename()
                if filename:
                    attachments.append(filename)
                continue

            payload_raw = part.get_payload(decode=True)
            payload = payload_raw if isinstance(payload_raw, bytes) else b""
            charset = part.get_content_charset() or _detect_encoding(payload[:65536])
            decoded = payload.decode(charset, errors="replace") if payload else ""

            if content_type == "text/plain" and decoded.strip():
                text_parts.append(decoded)
            elif content_type == "text/html" and decoded.strip():
                html_parts.append(decoded)
    else:
        payload_raw = msg.get_payload(decode=True)
        payload = payload_raw if isinstance(payload_raw, bytes) else b""
        charset = msg.get_content_charset() or _detect_encoding(payload[:65536])
        decoded = payload.decode(charset, errors="replace") if payload else ""
        if (msg.get_content_type() or "").lower() == "text/html":
            html_parts.append(decoded)
        else:
            text_parts.append(decoded)

    if text_parts:
        body = "\n\n".join(part.strip() for part in text_parts if part.strip())
        return body, attachments

    if html_parts:
        extractor = _HTMLTextExtractor()
        extractor.feed("\n".join(html_parts))
        return extractor.text(), attachments

    return "", attachments


class BaseParser(ABC):
    @abstractmethod
    def parse(self, source: str | Path | bytes | Message) -> Document:
        pass


class TextFileParser(BaseParser):
    @override
    def parse(self, source: str | Path | bytes | Message) -> Document:
        if isinstance(source, Message):
            content = str(source)
            return Document(
                content=content,
                metadata=DocumentMetadata(
                    source_type=DocumentSource.MANUAL,
                    source_id="raw:text",
                    title="raw_text",
                ),
            )

        if isinstance(source, (str, Path)) and _is_existing_path(source):
            path = Path(source)
            content = _read_text_file_streaming(path)
            metadata = DocumentMetadata(
                source_type=DocumentSource.FILE,
                source_id=str(path.resolve()),
                title=path.name,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                extra_fields={"extension": path.suffix.lower()},
            )
            return Document(content=content, metadata=metadata)

        if isinstance(source, bytes):
            content = source.decode(_detect_encoding(source[:65536]), errors="replace")
        else:
            content = str(source)

        return Document(
            content=content,
            metadata=DocumentMetadata(
                source_type=DocumentSource.MANUAL,
                source_id="raw:text",
                title="raw_text",
            ),
        )


class EmailParser(BaseParser):
    @override
    def parse(self, source: str | Path | bytes | Message) -> Document:
        message: Message
        source_id = "raw:email"

        if isinstance(source, Message):
            message = source
        elif isinstance(source, (str, Path)) and _is_existing_path(source):
            path = Path(source)
            source_id = str(path.resolve())
            with path.open("rb") as file_obj:
                message = BytesParser(policy=policy.default).parse(file_obj)
        elif isinstance(source, bytes):
            message = BytesParser(policy=policy.default).parsebytes(source)
        else:
            message = Parser(policy=policy.default).parsestr(str(source))

        subject = message.get("Subject")
        from_addr = message.get("From")
        to_header = message.get("To", "")
        to_addr = [item.strip() for item in to_header.split(",") if item.strip()] if to_header else []
        created_at = _normalize_datetime(message.get("Date"))
        body, attachments = _extract_email_payload(message)

        return EmailDocument.from_email_fields(
            body=body,
            source_id=source_id,
            subject=subject,
            from_addr=from_addr,
            to_addr=to_addr,
            attachments=attachments,
            created_at=created_at,
            extra_fields={"message_id": message.get("Message-ID")},
            tags=["email"],
        )


class PDFParser(BaseParser):
    @override
    def parse(self, source: str | Path | bytes | Message) -> Document:
        if isinstance(source, Message):
            raise ValueError("PDFParser accepts file path or raw PDF bytes.")

        try:
            pdf_module = importlib.import_module("PyPDF2")
            pdf_reader_ctor = getattr(pdf_module, "PdfReader")
        except Exception:
            fallback_content = ""
            if isinstance(source, bytes):
                fallback_content = source.decode(_detect_encoding(source[:65536]), errors="replace")
            return Document(
                content=fallback_content,
                metadata=DocumentMetadata(
                    source_type=DocumentSource.FILE,
                    source_id=str(source) if not isinstance(source, bytes) else "raw:pdf",
                    title=Path(source).name if isinstance(source, (str, Path)) else "raw_pdf",
                    extra_fields={"parser_unavailable": "PyPDF2"},
                ),
            )

        source_id = "raw:pdf"
        pages_text: list[str] = []

        if isinstance(source, (str, Path)) and _is_existing_path(source):
            path = Path(source)
            source_id = str(path.resolve())
            with path.open("rb") as file_obj:
                reader = pdf_reader_ctor(file_obj)
                for page in reader.pages:
                    extracted = page.extract_text()
                    pages_text.append(str(extracted or "").strip())
            metadata = DocumentMetadata(
                source_type=DocumentSource.FILE,
                source_id=source_id,
                title=path.name,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                extra_fields={"extension": ".pdf", "pages": len(pages_text)},
            )
            return Document(content="\n\n".join(filter(None, pages_text)), metadata=metadata)

        if isinstance(source, bytes):
            reader = pdf_reader_ctor(BytesIO(source))
            for page in reader.pages:
                extracted = page.extract_text()
                pages_text.append(str(extracted or "").strip())
        else:
            raise ValueError("PDFParser accepts file path or raw PDF bytes.")

        return Document(
            content="\n\n".join(filter(None, pages_text)),
            metadata=DocumentMetadata(
                source_type=DocumentSource.API,
                source_id=source_id,
                title="raw_pdf",
                extra_fields={"pages": len(pages_text)},
            ),
        )


class ParserFactory:
    @staticmethod
    def get_parser(source: str | Path | bytes | Message, content_type: str | None = None) -> BaseParser:
        if content_type:
            normalized = content_type.lower()
            if "message/rfc822" in normalized or normalized.startswith("email/"):
                return EmailParser()
            if "pdf" in normalized:
                return PDFParser()
            if normalized.startswith("text/"):
                return TextFileParser()

        if isinstance(source, (str, Path)):
            path = Path(source)
            suffix = path.suffix.lower()
            if suffix in {".eml"}:
                return EmailParser()
            if suffix in {".pdf"}:
                return PDFParser()
            if suffix in {".txt", ".md"}:
                return TextFileParser()

            source_str = str(source)
            if "@" in source_str and "\n" in source_str:
                return EmailParser()

        if isinstance(source, bytes):
            if source.startswith(b"%PDF"):
                return PDFParser()
            if b"Subject:" in source or b"From:" in source:
                return EmailParser()
            return TextFileParser()

        return TextFileParser()


def create_document_from_file(filepath: str | Path) -> Document:
    parser = ParserFactory.get_parser(filepath)
    return parser.parse(filepath)


def create_document_from_email(email_data: str | bytes | Message) -> Document:
    return EmailParser().parse(email_data)


def extract_text_from_bytes(content_bytes: bytes, mime_type: str) -> str:
    parser = ParserFactory.get_parser(content_bytes, content_type=mime_type)
    document = parser.parse(content_bytes)
    return document.content