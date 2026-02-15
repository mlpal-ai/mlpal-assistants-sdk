"""Tool-related types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """Function call details from model response."""

    name: str
    arguments: str  # JSON string

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> FunctionCall:
        """Create from API response."""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", "{}"),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call made by the model."""

    id: str
    type: str
    function: FunctionCall
    thought_signature: str | None = None  # Gemini-specific field

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ToolCall:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=FunctionCall.from_api(data.get("function", {})),
            thought_signature=data.get("thought_signature"),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    name: str
    result: str
    success: bool
    error: str | None = None

    def to_message(self) -> dict[str, Any]:
        """Convert to tool message format."""
        content = self.result if self.success else f"Error: {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content,
        }
