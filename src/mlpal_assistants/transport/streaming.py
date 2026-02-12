"""SSE streaming support for HTTP transport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    import httpx


@dataclass
class StreamEvent:
    """A single server-sent event."""

    event: str | None
    data: str
    id: str | None = None
    retry: int | None = None

    def json(self) -> dict[str, Any]:
        """Parse data as JSON."""
        result: dict[str, Any] = json.loads(self.data)
        return result


async def iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """
    Iterate over lines from an SSE response.

    Handles chunked encoding and line buffering.
    """
    buffer = ""

    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line


async def parse_sse_stream(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    """
    Parse SSE stream from an HTTP response.

    Yields StreamEvent objects for each complete event.
    """
    event_type: str | None = None
    event_data: list[str] = []
    event_id: str | None = None
    retry: int | None = None

    async for line in iter_sse_lines(response):
        # Empty line = dispatch event
        if not line:
            if event_data:
                yield StreamEvent(
                    event=event_type,
                    data="\n".join(event_data),
                    id=event_id,
                    retry=retry,
                )
            # Reset for next event
            event_type = None
            event_data = []
            event_id = None
            retry = None
            continue

        # Comment line (starts with :)
        if line.startswith(":"):
            continue

        # Parse field: value
        if ":" in line:
            field, _, value = line.partition(":")
            # Remove leading space from value (per SSE spec)
            if value.startswith(" "):
                value = value[1:]
        else:
            field = line
            value = ""

        # Handle known fields
        if field == "event":
            event_type = value
        elif field == "data":
            event_data.append(value)
        elif field == "id":
            event_id = value
        elif field == "retry":
            try:
                retry = int(value)
            except ValueError:
                pass

    # Dispatch any remaining event
    if event_data:
        yield StreamEvent(
            event=event_type,
            data="\n".join(event_data),
            id=event_id,
            retry=retry,
        )


@dataclass(frozen=True, slots=True)
class ChatStreamDelta:
    """A single delta in a streaming chat response."""

    content: str | None = None
    """Text content delta."""

    role: str | None = None
    """Role (usually only in first delta)."""

    finish_reason: str | None = None
    """Set when streaming is complete (e.g., 'stop', 'tool_calls')."""

    tool_calls: tuple[dict[str, Any], ...] | None = None
    """Tool call deltas."""

    raw: dict[str, Any] | None = None
    """Raw API response data."""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ChatStreamDelta:
        """Parse streaming API response."""
        choices = data.get("choices", [])
        if not choices:
            return cls(raw=data)

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Parse tool calls if present
        tool_calls = None
        if raw_tool_calls := delta.get("tool_calls"):
            tool_calls = tuple(raw_tool_calls)

        return cls(
            content=delta.get("content"),
            role=delta.get("role"),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=data,
        )


@dataclass
class ChatStream:
    """
    Async iterator for streaming chat responses.

    Yields ChatStreamDelta objects with incremental content.

    Example:
        async for delta in stream:
            if delta.content:
                print(delta.content, end="", flush=True)
    """

    _response: httpx.Response
    _collected_content: list[str]
    _finish_reason: str | None

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._collected_content = []
        self._finish_reason = None

    async def __aiter__(self) -> AsyncIterator[ChatStreamDelta]:
        """Iterate over stream deltas."""
        async for event in parse_sse_stream(self._response):
            # Skip [DONE] marker
            if event.data == "[DONE]":
                break

            try:
                data = event.json()
            except json.JSONDecodeError:
                continue

            delta = ChatStreamDelta.from_api(data)

            # Collect content for final response
            if delta.content:
                self._collected_content.append(delta.content)

            if delta.finish_reason:
                self._finish_reason = delta.finish_reason

            yield delta

    @property
    def collected_content(self) -> str:
        """Get all collected content so far."""
        return "".join(self._collected_content)

    @property
    def finish_reason(self) -> str | None:
        """Get the finish reason (available after iteration completes)."""
        return self._finish_reason

    async def collect(self) -> str:
        """Consume the stream and return collected content."""
        async for _ in self:
            pass
        return self.collected_content
