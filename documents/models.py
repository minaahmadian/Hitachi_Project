from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class DocumentSource(str, Enum):
    """Normalized source categories for all ingested content."""

    FILE = "file"
    EMAIL = "email"
    API = "api"
    MANUAL = "manual"


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata attached to each normalized document."""

    source_type: DocumentSource
    source_id: str
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    extra_fields: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Unified document payload used before vectorDB storage."""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: DocumentMetadata = field(
        default_factory=lambda: DocumentMetadata(source_type=DocumentSource.MANUAL, source_id="manual")
    )
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class EmailDocument(Document):
    """Document variant for email inputs with email-specific fields."""

    subject: str | None = None
    from_addr: str | None = None
    to_addr: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    @classmethod
    def from_email_fields(
        cls,
        *,
        body: str,
        source_id: str,
        subject: str | None = None,
        from_addr: str | None = None,
        to_addr: list[str] | None = None,
        attachments: list[str] | None = None,
        created_at: datetime | None = None,
        author: str | None = None,
        extra_fields: dict[str, object] | None = None,
        tags: list[str] | None = None,
    ) -> "EmailDocument":
        recipient_list = to_addr or []
        attachment_list = attachments or []
        metadata = DocumentMetadata(
            source_type=DocumentSource.EMAIL,
            source_id=source_id,
            title=subject,
            author=author or from_addr,
            created_at=created_at,
            tags=tags or [],
            extra_fields={
                "subject": subject,
                "from_addr": from_addr,
                "to_addr": recipient_list,
                "attachments": attachment_list,
                **(extra_fields or {}),
            },
        )
        return cls(
            content=body,
            metadata=metadata,
            subject=subject,
            from_addr=from_addr,
            to_addr=recipient_list,
            attachments=attachment_list,
        )