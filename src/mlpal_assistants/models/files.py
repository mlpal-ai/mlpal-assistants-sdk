"""File handling with factory methods."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

FileType = Literal["image", "audio", "document"]


@dataclass(frozen=True, slots=True)
class File:
    """
    Immutable file attachment for chat messages.

    Use factory methods to create:
        File.from_path("image.jpg")
        File.from_bytes(data, mime_type="image/png")
        File.from_url("https://...")

    Convenience factories:
        File.image("photo.jpg")
        File.audio("recording.mp3")
        File.document("report.pdf")
    """

    type: FileType
    mime_type: str | None
    _url: str | None = None
    _base64: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        type: FileType | None = None,
        mime_type: str | None = None,
    ) -> File:
        """Create from local file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = base64.b64encode(path.read_bytes()).decode()
        mime = mime_type or mimetypes.guess_type(str(path))[0]

        return cls(
            type=type or cls._infer_type(mime),
            mime_type=mime,
            _base64=content,
        )

    @classmethod
    def from_bytes(
        cls,
        content: bytes | BinaryIO,
        *,
        mime_type: str,
        type: FileType | None = None,
    ) -> File:
        """Create from bytes with explicit MIME type."""
        if hasattr(content, "read"):
            content = content.read()

        encoded = base64.b64encode(content).decode()
        return cls(
            type=type or cls._infer_type(mime_type),
            mime_type=mime_type,
            _base64=encoded,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        type: FileType | None = None,
        mime_type: str | None = None,
    ) -> File:
        """Create from URL (passed through to API, not downloaded)."""
        mime = mime_type or mimetypes.guess_type(url)[0]
        return cls(
            type=type or cls._infer_type(mime),
            mime_type=mime,
            _url=url,
        )

    # Convenience factories
    @classmethod
    def image(cls, source: str | Path | bytes, *, mime_type: str | None = None) -> File:
        """Create an image file attachment."""
        if isinstance(source, bytes):
            return cls.from_bytes(source, type="image", mime_type=mime_type or "image/png")
        return cls.from_path(source, type="image", mime_type=mime_type)

    @classmethod
    def audio(cls, source: str | Path | bytes, *, mime_type: str | None = None) -> File:
        """Create an audio file attachment."""
        if isinstance(source, bytes):
            return cls.from_bytes(source, type="audio", mime_type=mime_type or "audio/mpeg")
        return cls.from_path(source, type="audio", mime_type=mime_type)

    @classmethod
    def document(cls, source: str | Path | bytes, *, mime_type: str | None = None) -> File:
        """Create a document file attachment."""
        if isinstance(source, bytes):
            return cls.from_bytes(source, type="document", mime_type=mime_type or "application/octet-stream")
        return cls.from_path(source, type="document", mime_type=mime_type)

    @staticmethod
    def _infer_type(mime: str | None) -> FileType:
        """Infer file type from MIME type."""
        if not mime:
            return "document"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        return "document"

    def to_api_format(self) -> dict[str, str]:
        """Serialize for API request."""
        result: dict[str, str] = {"type": self.type}

        if self._url:
            result["url"] = self._url
        elif self._base64:
            result["base64"] = self._base64

        if self.mime_type:
            result["mime_type"] = self.mime_type

        return result

    def __repr__(self) -> str:
        if self._url:
            return f"File(type={self.type!r}, url={self._url!r})"
        return f"File(type={self.type!r}, mime_type={self.mime_type!r})"
