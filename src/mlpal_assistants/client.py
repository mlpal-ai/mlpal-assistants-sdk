"""The v2 MLPal clients — one surface for inference and management.

`MLPal` (sync) and `AsyncMLPal` (async) are the entry points app teams
standardize on. They speak the native Anthropic wire for inference
(`client.messages`), list routable models (`client.models`), and administer
keys/policies/budgets (`client.admin`). Point `base_url` at the managed service
or a self-hosted gateway — the surface is identical.

    from mlpal_assistants import MLPal

    client = MLPal(api_key="mlpal_sk_...")
    msg = client.messages.create(
        model="mlpal",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(msg.text)
"""

from __future__ import annotations

from mlpal_assistants._resources.admin import Admin, AsyncAdmin
from mlpal_assistants._resources.catalog import AsyncCatalogResource, CatalogResource
from mlpal_assistants._resources.feedback import AsyncFeedback, Feedback
from mlpal_assistants._resources.messages import AsyncMessages, Messages
from mlpal_assistants._resources.models import AsyncModels, Models
from mlpal_assistants._resources.usage import AsyncUsageResource, UsageResource
from mlpal_assistants._transport import AsyncTransport, SyncTransport
from mlpal_assistants.config import Config


class MLPal:
    """Synchronous MLPal client."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: object | None = None,
    ) -> None:
        self._config = Config.from_env(api_key=api_key, base_url=base_url, timeout=timeout)
        self._transport = SyncTransport(self._config, http_client=http_client)  # type: ignore[arg-type]
        self.messages = Messages(self._transport)
        self.models = Models(self._transport)
        self.catalog = CatalogResource(self._transport)
        self.feedback = Feedback(self._transport)
        self.usage = UsageResource(self._transport)
        self.admin = Admin(self._transport)

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> MLPal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncMLPal:
    """Asynchronous MLPal client."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: object | None = None,
    ) -> None:
        self._config = Config.from_env(api_key=api_key, base_url=base_url, timeout=timeout)
        self._transport = AsyncTransport(self._config, http_client=http_client)  # type: ignore[arg-type]
        self.messages = AsyncMessages(self._transport)
        self.models = AsyncModels(self._transport)
        self.catalog = AsyncCatalogResource(self._transport)
        self.feedback = AsyncFeedback(self._transport)
        self.usage = AsyncUsageResource(self._transport)
        self.admin = AsyncAdmin(self._transport)

    @property
    def base_url(self) -> str:
        return self._config.base_url

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncMLPal:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
