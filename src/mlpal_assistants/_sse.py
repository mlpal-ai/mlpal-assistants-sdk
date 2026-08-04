"""Server-Sent Events decoding for the native (Anthropic-wire) stream.

The gateway streams `/v2/messages` as faithful Anthropic SSE — `event:` +
`data:` line pairs separated by blank lines, with `: ping` heartbeat comments
interleaved during long reasoning/tool phases. This decoder is fed one decoded
line at a time (from httpx's `iter_lines`/`aiter_lines`, which already strips
newlines) and emits one `ServerSentEvent` per complete record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerSentEvent:
    """One decoded SSE record. `data` is the parsed JSON payload when the frame
    carried JSON (the common case for Anthropic events); otherwise the raw
    string is kept on `raw_data` and `data` is None."""

    event: str | None = None
    data: Any = None
    raw_data: str | None = None

    @property
    def type(self) -> str | None:
        """The Anthropic event type. Prefer the payload's own `type` (present on
        every Anthropic event) and fall back to the SSE `event:` field."""
        if isinstance(self.data, dict) and isinstance(self.data.get("type"), str):
            return self.data["type"]
        return self.event


@dataclass
class SSEDecoder:
    """Line-oriented, stateful SSE decoder. `push` returns a completed event on
    the blank line that terminates a record, else None."""

    _event: str | None = None
    _data: list[str] = field(default_factory=list)

    def push(self, line: str) -> ServerSentEvent | None:
        # A blank line dispatches the buffered record (SSE framing).
        if line == "":
            if self._event is None and not self._data:
                return None
            raw = "\n".join(self._data)
            evt = ServerSentEvent(event=self._event, raw_data=raw, data=_maybe_json(raw))
            self._event = None
            self._data = []
            return evt

        # Comment line (`:` prefix) — heartbeats arrive as `: ping`. Ignore.
        if line.startswith(":"):
            return None

        field_name, _, value = line.partition(":")
        # Per the SSE spec a single leading space after the colon is stripped.
        if value.startswith(" "):
            value = value[1:]

        if field_name == "event":
            self._event = value
        elif field_name == "data":
            self._data.append(value)
        # Other fields (id, retry) are unused by the Anthropic wire.
        return None


def _maybe_json(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
