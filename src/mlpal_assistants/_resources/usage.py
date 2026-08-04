"""`client.usage` — this account's own usage at `/v1/usage`.

Scoped to the calling key's account (per-key breakdowns for an admin live under
`client.admin.keys.usage`). `summary` aggregates a window; `daily` returns a
per-day series (empty days are not materialized — zero-fill client-side if a
contiguous series is needed).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from mlpal_assistants._wire import DailyUsage, UsageSummary

if TYPE_CHECKING:
    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_SUMMARY = "/v1/usage/summary"
_DAILY = "/v1/usage/daily"


def _window(start_date: datetime | str | None, end_date: datetime | str | None) -> dict[str, str]:
    params: dict[str, str] = {}
    if start_date is not None:
        params["start_date"] = start_date.isoformat() if isinstance(start_date, datetime) else start_date
    if end_date is not None:
        params["end_date"] = end_date.isoformat() if isinstance(end_date, datetime) else end_date
    return params


class UsageResource:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def summary(
        self,
        *,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        timeout: float | None = None,
    ) -> UsageSummary:
        data, _ = self._transport.request(
            "GET", _SUMMARY, params=_window(start_date, end_date), timeout=timeout
        )
        return UsageSummary.model_validate(data)

    def daily(self, *, days: int = 30, timeout: float | None = None) -> DailyUsage:
        data, _ = self._transport.request("GET", _DAILY, params={"days": days}, timeout=timeout)
        return DailyUsage.model_validate(data)


class AsyncUsageResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def summary(
        self,
        *,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        timeout: float | None = None,
    ) -> UsageSummary:
        data, _ = await self._transport.request(
            "GET", _SUMMARY, params=_window(start_date, end_date), timeout=timeout
        )
        return UsageSummary.model_validate(data)

    async def daily(self, *, days: int = 30, timeout: float | None = None) -> DailyUsage:
        data, _ = await self._transport.request(
            "GET", _DAILY, params={"days": days}, timeout=timeout
        )
        return DailyUsage.model_validate(data)
