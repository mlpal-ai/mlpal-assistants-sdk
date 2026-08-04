"""`client.models` — capability advertisement for the native inference surface.

Lists the models this key can serve on `/v2/messages`, with per-model
capability flags (tools/vision/reasoning/caching). Filtered by the key's model
policy, so it reflects what the caller can actually route to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlpal_assistants._wire import ModelInfo

if TYPE_CHECKING:
    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_PATH = "/v2/messages/models"


def _parse(data: object) -> list[ModelInfo]:
    rows = data.get("data", []) if isinstance(data, dict) else []
    return [ModelInfo.model_validate(r) for r in rows]


class Models:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(self, *, timeout: float | None = None) -> list[ModelInfo]:
        data, _ = self._transport.request("GET", _PATH, timeout=timeout)
        return _parse(data)


class AsyncModels:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(self, *, timeout: float | None = None) -> list[ModelInfo]:
        data, _ = await self._transport.request("GET", _PATH, timeout=timeout)
        return _parse(data)
