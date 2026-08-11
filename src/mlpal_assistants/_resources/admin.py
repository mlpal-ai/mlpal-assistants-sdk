"""`client.admin` — the management (control-plane) surface at `/admin/v1/*`.

Issue and govern API keys, including their **model policy** (allow/deny globs)
and **spend budgets** (usd/cu caps per window). The caller must present an
admin-scoped principal — a Cognito JWT on the managed service, or an
`admin`-scoped API key when self-hosting (`MLPAL_AUTH_BACKEND=local`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mlpal_assistants._wire import (
    KeyUsageSummary,
    ManagedKey,
    ManagedKeyList,
    ManagedKeyWithSecret,
)

if TYPE_CHECKING:
    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_KEYS = "/admin/v1/keys"


def _create_body(
    name: str,
    description: str | None,
    permissions: Iterable[str] | None,
    expires_at: datetime | str | None,
    tags: Mapping[str, Any] | None,
    model_policy: Mapping[str, Any] | None,
    budgets: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name}
    if description is not None:
        body["description"] = description
    if permissions is not None:
        body["permissions"] = list(permissions)
    if expires_at is not None:
        body["expires_at"] = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    if tags is not None:
        body["tags"] = dict(tags)
    if model_policy is not None:
        body["model_policy"] = dict(model_policy)
    if budgets is not None:
        body["budgets"] = [dict(b) for b in budgets]
    return body


def _update_body(
    model_policy: Mapping[str, Any] | None,
    budgets: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    # Sentinel-free convention (mirrors APIKeyUpdate): omit to keep, send {}/[]
    # to clear. Callers pass None to keep a field untouched.
    body: dict[str, Any] = {}
    if model_policy is not None:
        body["model_policy"] = dict(model_policy)
    if budgets is not None:
        body["budgets"] = [dict(b) for b in budgets]
    return body


class Keys:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        permissions: Iterable[str] | None = None,
        expires_at: datetime | str | None = None,
        tags: Mapping[str, Any] | None = None,
        model_policy: Mapping[str, Any] | None = None,
        budgets: Iterable[Mapping[str, Any]] | None = None,
    ) -> ManagedKeyWithSecret:
        """Create a key. The returned `.secret` is shown once — store it now."""
        body = _create_body(name, description, permissions, expires_at, tags, model_policy, budgets)
        data, _ = self._transport.request("POST", _KEYS, json=body)
        return ManagedKeyWithSecret.model_validate(data)

    def list(self) -> ManagedKeyList:
        data, _ = self._transport.request("GET", _KEYS)
        return ManagedKeyList.model_validate(data)

    def retrieve(self, key_id: int) -> ManagedKey:
        data, _ = self._transport.request("GET", f"{_KEYS}/{key_id}")
        return ManagedKey.model_validate(data)

    def update(
        self,
        key_id: int,
        *,
        model_policy: Mapping[str, Any] | None = None,
        budgets: Iterable[Mapping[str, Any]] | None = None,
    ) -> ManagedKey:
        """Replace policy/budgets. Omit a field to keep it; pass `{}`/`[]` to clear."""
        data, _ = self._transport.request(
            "PATCH", f"{_KEYS}/{key_id}", json=_update_body(model_policy, budgets)
        )
        return ManagedKey.model_validate(data)

    def delete(self, key_id: int) -> None:
        self._transport.request("DELETE", f"{_KEYS}/{key_id}")

    def usage(self, key_id: int) -> KeyUsageSummary:
        data, _ = self._transport.request("GET", f"{_KEYS}/{key_id}/usage")
        return KeyUsageSummary.model_validate(data)


class AsyncKeys:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        name: str,
        description: str | None = None,
        permissions: Iterable[str] | None = None,
        expires_at: datetime | str | None = None,
        tags: Mapping[str, Any] | None = None,
        model_policy: Mapping[str, Any] | None = None,
        budgets: Iterable[Mapping[str, Any]] | None = None,
    ) -> ManagedKeyWithSecret:
        body = _create_body(name, description, permissions, expires_at, tags, model_policy, budgets)
        data, _ = await self._transport.request("POST", _KEYS, json=body)
        return ManagedKeyWithSecret.model_validate(data)

    async def list(self) -> ManagedKeyList:
        data, _ = await self._transport.request("GET", _KEYS)
        return ManagedKeyList.model_validate(data)

    async def retrieve(self, key_id: int) -> ManagedKey:
        data, _ = await self._transport.request("GET", f"{_KEYS}/{key_id}")
        return ManagedKey.model_validate(data)

    async def update(
        self,
        key_id: int,
        *,
        model_policy: Mapping[str, Any] | None = None,
        budgets: Iterable[Mapping[str, Any]] | None = None,
    ) -> ManagedKey:
        data, _ = await self._transport.request(
            "PATCH", f"{_KEYS}/{key_id}", json=_update_body(model_policy, budgets)
        )
        return ManagedKey.model_validate(data)

    async def delete(self, key_id: int) -> None:
        await self._transport.request("DELETE", f"{_KEYS}/{key_id}")

    async def usage(self, key_id: int) -> KeyUsageSummary:
        data, _ = await self._transport.request("GET", f"{_KEYS}/{key_id}/usage")
        return KeyUsageSummary.model_validate(data)


class Admin:
    def __init__(self, transport: SyncTransport) -> None:
        self.keys = Keys(transport)


class AsyncAdmin:
    def __init__(self, transport: AsyncTransport) -> None:
        self.keys = AsyncKeys(transport)
