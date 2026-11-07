"""HTTP transport for the v2 client — sync and async over httpx.

Owns exactly three concerns: auth/headers, HTTP-status → typed-exception
mapping, and bounded retries on transient failures. It does no request shaping
or response parsing beyond JSON; resources own that. Sync and async share every
pure decision (header build, error mapping, retry policy) and differ only in the
I/O loop, so the two paths cannot drift.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from mlpal_assistants.config import Config
from mlpal_assistants.exceptions import (
    AssistantsError,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    ModelNotAvailableError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    TimeoutError,
)

# Anthropic requires a version header on the Messages wire; the gateway forwards
# it as-is. Overridable per request via extra_headers.
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


def _user_agent() -> str:
    from mlpal_assistants._version import __version__

    return f"mlpal-assistants-python/{__version__}"


def build_headers(config: Config, extra: Mapping[str, str] | None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _user_agent(),
        "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
        # One base URL, two shapes on /v1/chat/completions and /v1/models: this
        # SDK speaks OpenAI's (choices[], chat.completion.chunk) and says so
        # explicitly, so the gateway's flat-shape default never applies to it.
        "X-MLPal-Wire": "openai",
    }
    if extra:
        # Caller overrides win (e.g. anthropic-version, anthropic-beta).
        headers.update({k: v for k, v in extra.items() if v is not None})
    return headers


def _request_id(headers: Mapping[str, str]) -> str | None:
    return headers.get("request-id") or headers.get("x-request-id")


def _extract_message(body: Any, fallback: str) -> str:
    """Pull a human message out of the various error envelopes the gateway can
    return: Anthropic `{"error": {"message"}}`, FastAPI `{"detail"}`, or a bare
    string."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return str(detail)
        if body.get("message"):
            return str(body["message"])
    if isinstance(body, str) and body:
        return body
    return fallback


def raise_for_status(status: int, body: Any, headers: Mapping[str, str]) -> None:
    """Map an error response to a typed exception. Never called for 2xx."""
    request_id = _request_id(headers)
    message = _extract_message(body, f"HTTP {status}")
    kw: dict[str, Any] = {"status_code": status, "request_id": request_id}
    kw["response"] = body if isinstance(body, dict) else None

    if status == 400:
        raise BadRequestError(message, **kw)
    if status == 401:
        raise AuthenticationError(message, **kw)
    if status == 403:
        raise PermissionDeniedError(message, **kw)
    if status == 404:
        exc = ModelNotAvailableError if "model" in message.lower() else NotFoundError
        raise exc(message, **kw)
    if status in (402, 413):
        # 402: budget/quota denial from the policy engine. 413: over a size cap.
        raise QuotaExceededError(message, **kw)
    if status == 429:
        retry_after = _retry_after_seconds(headers)
        # A budget cap and a rate limit both surface as 429; disambiguate on text.
        if any(w in message.lower() for w in ("budget", "quota", "spend")):
            raise QuotaExceededError(message, **kw)
        raise RateLimitError(message, retry_after=retry_after, **kw)
    if status == 502:
        raise ProviderError(message, **kw)
    if status == 503:
        raise ServiceUnavailableError(message, **kw)
    if status >= 500:
        raise ServerError(message, **kw)
    raise AssistantsError(message, **kw)


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # HTTP-date form; not worth parsing for a hint


def _backoff(attempt: int, headers: Mapping[str, str] | None) -> float:
    if headers:
        ra = _retry_after_seconds(headers)
        if ra is not None:
            return min(ra, 30.0)
    return min(0.5 * (2**attempt), 8.0)  # 0.5, 1, 2, 4, … capped


class _BaseTransport:
    def __init__(self, config: Config) -> None:
        self._config = config

    def _should_retry(
        self, attempt: int, max_retries: int, status: int, headers: Mapping[str, str]
    ) -> bool:
        # The gateway marks a response that must not be retried (its deadline
        # 504: the provider call was cancelled, a retry is another one).
        if headers.get("x-should-retry", "").lower() == "false":
            return False
        return attempt < max_retries and status in RETRY_STATUSES


class SyncTransport(_BaseTransport):
    def __init__(self, config: Config, http_client: httpx.Client | None = None) -> None:
        super().__init__(config)
        self._client = http_client or httpx.Client(
            base_url=config.base_url, timeout=httpx.Timeout(config.timeout)
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> tuple[Any, httpx.Headers]:
        hdrs = build_headers(self._config, headers)
        req_timeout = httpx.Timeout(timeout) if timeout is not None else httpx.USE_CLIENT_DEFAULT
        attempt = 0
        while True:
            try:
                resp = self._client.request(
                    method, path, json=json, params=params, headers=hdrs, timeout=req_timeout
                )
            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    time.sleep(_backoff(attempt, None))
                    attempt += 1
                    continue
                raise TimeoutError(f"Request timed out: {e}") from e
            except httpx.HTTPError as e:
                raise ConnectionError(f"Failed to connect: {e}") from e

            if resp.status_code < 400:
                return _decode(resp), resp.headers
            if self._should_retry(attempt, max_retries, resp.status_code, resp.headers):
                time.sleep(_backoff(attempt, resp.headers))
                attempt += 1
                continue
            raise_for_status(resp.status_code, _decode(resp), resp.headers)

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Iterator[httpx.Response]:
        hdrs = build_headers(self._config, headers)
        hdrs["Accept"] = "text/event-stream"
        req_timeout = httpx.Timeout(timeout) if timeout is not None else httpx.USE_CLIENT_DEFAULT
        with self._client.stream(
            method, path, json=json, headers=hdrs, timeout=req_timeout
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise_for_status(resp.status_code, _decode(resp), resp.headers)
            yield resp

    def close(self) -> None:
        self._client.close()


class AsyncTransport(_BaseTransport):
    def __init__(self, config: Config, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(config)
        self._client = http_client or httpx.AsyncClient(
            base_url=config.base_url, timeout=httpx.Timeout(config.timeout)
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> tuple[Any, httpx.Headers]:
        import asyncio

        hdrs = build_headers(self._config, headers)
        req_timeout = httpx.Timeout(timeout) if timeout is not None else httpx.USE_CLIENT_DEFAULT
        attempt = 0
        while True:
            try:
                resp = await self._client.request(
                    method, path, json=json, params=params, headers=hdrs, timeout=req_timeout
                )
            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    await asyncio.sleep(_backoff(attempt, None))
                    attempt += 1
                    continue
                raise TimeoutError(f"Request timed out: {e}") from e
            except httpx.HTTPError as e:
                raise ConnectionError(f"Failed to connect: {e}") from e

            if resp.status_code < 400:
                return _decode(resp), resp.headers
            if self._should_retry(attempt, max_retries, resp.status_code, resp.headers):
                await asyncio.sleep(_backoff(attempt, resp.headers))
                attempt += 1
                continue
            raise_for_status(resp.status_code, _decode(resp), resp.headers)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        hdrs = build_headers(self._config, headers)
        hdrs["Accept"] = "text/event-stream"
        req_timeout = httpx.Timeout(timeout) if timeout is not None else httpx.USE_CLIENT_DEFAULT
        async with self._client.stream(
            method, path, json=json, headers=hdrs, timeout=req_timeout
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise_for_status(resp.status_code, _decode(resp), resp.headers)
            yield resp

    async def close(self) -> None:
        await self._client.aclose()


def _decode(resp: httpx.Response) -> Any:
    """Best-effort JSON decode; fall back to text for non-JSON error bodies."""
    try:
        return resp.json()
    except ValueError:
        return resp.text
