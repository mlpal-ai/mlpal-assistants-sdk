"""`client.catalog` — the curated model view at `/v2/catalog`.

Returns the tier ladder for a curation profile with live cost and availability.
Use `catalog.tiers` + `catalog.routing_ladder` to route a subtask by complexity
without hardcoding model tags. Filtered to the models this key can serve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlpal_assistants._wire import Catalog

if TYPE_CHECKING:
    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_PATH = "/v2/catalog"


class CatalogResource:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def retrieve(self, *, profile: str = "coding", timeout: float | None = None) -> Catalog:
        data, _ = self._transport.request(
            "GET", _PATH, params={"profile": profile}, timeout=timeout
        )
        return Catalog.model_validate(data)


class AsyncCatalogResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def retrieve(self, *, profile: str = "coding", timeout: float | None = None) -> Catalog:
        data, _ = await self._transport.request(
            "GET", _PATH, params={"profile": profile}, timeout=timeout
        )
        return Catalog.model_validate(data)
