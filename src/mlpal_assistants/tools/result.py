"""Tool execution result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Result of executing a tool call.

    Attributes:
        tool_call_id: ID of the tool call this result is for
        name: Name of the tool that was executed
        result: String result from the tool (JSON for structured data)
        success: Whether execution succeeded
        error: Error message if execution failed
        latency_ms: Execution time in milliseconds
    """

    tool_call_id: str
    name: str
    result: str
    success: bool
    error: str | None = None
    latency_ms: float | None = None

    def to_message(self) -> dict[str, Any]:
        """Convert to tool message format for chat API."""
        content = self.result if self.success else f"Error: {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content,
        }

    @classmethod
    def success_result(
        cls,
        tool_call_id: str,
        name: str,
        result: str,
        latency_ms: float | None = None,
    ) -> ToolResult:
        """Create a successful result."""
        return cls(
            tool_call_id=tool_call_id,
            name=name,
            result=result,
            success=True,
            latency_ms=latency_ms,
        )

    @classmethod
    def error_result(
        cls,
        tool_call_id: str,
        name: str,
        error: str,
        latency_ms: float | None = None,
    ) -> ToolResult:
        """Create an error result."""
        return cls(
            tool_call_id=tool_call_id,
            name=name,
            result="",
            success=False,
            error=error,
            latency_ms=latency_ms,
        )
