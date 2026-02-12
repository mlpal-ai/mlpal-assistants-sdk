"""Message types for the SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from mlpal_assistants.models.files import File


class FileAttachment(TypedDict, total=False):
    """File attachment in API format."""

    type: Literal["image", "audio", "document"]
    url: str | None
    base64: str | None
    mime_type: str | None


class Message(TypedDict, total=False):
    """A message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    files: list[File | FileAttachment | dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


def normalize_message(message: Message | dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a message for API submission.

    Converts File objects to API format and ensures proper structure.

    Args:
        message: Message dict with optional File objects

    Returns:
        Message dict ready for API submission
    """
    from mlpal_assistants.models.files import File

    result: dict[str, Any] = dict(message)

    if "files" in result and result["files"]:
        normalized_files = []
        for f in result["files"]:
            if isinstance(f, File):
                normalized_files.append(f.to_api_format())
            elif isinstance(f, dict):
                normalized_files.append(f)
            else:
                raise TypeError(f"Invalid file type: {type(f)}")
        result["files"] = normalized_files

    return result


def normalize_messages(messages: list[Message | dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a list of messages for API submission."""
    return [normalize_message(m) for m in messages]
