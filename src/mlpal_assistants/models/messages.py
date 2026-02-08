"""Message types for chat conversations."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from mlpal_assistants.models.files import File

Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict, total=False):
    """A message in a chat conversation."""

    role: Role  # Required
    content: str | None
    files: list[File | dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


def normalize_message(message: Message | dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a message for API request.

    Converts File objects to API format and ensures proper structure.
    """
    result: dict[str, Any] = {
        "role": message.get("role", "user"),
    }

    if "content" in message:
        result["content"] = message["content"]

    # Convert files to API format
    if files := message.get("files"):
        result["files"] = [
            f.to_api_format() if isinstance(f, File) else f
            for f in files
        ]

    # Pass through other fields
    if "tool_calls" in message:
        result["tool_calls"] = message["tool_calls"]
    if "tool_call_id" in message:
        result["tool_call_id"] = message["tool_call_id"]
    if "name" in message:
        result["name"] = message["name"]

    return result


def normalize_messages(
    messages: list[Message | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a list of messages for API request."""
    return [normalize_message(m) for m in messages]
