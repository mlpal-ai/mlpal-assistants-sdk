"""HTTP transport - pure communication, no business logic."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx

from mlpal_assistants.exceptions import (
    AssistantsError,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from mlpal_assistants.transport.streaming import ChatStream

if TYPE_CHECKING:
    from mlpal_assistants.config import Config


class HTTPTransport:
    """
    HTTP transport layer.

    Responsibilities:
    - Connection pooling
    - Auth headers
    - HTTP status → exception mapping

    NOT responsible for:
    - Response parsing (beyond JSON)
    - Business logic
    - Retries (service handles this)
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout),
        )

    async def post(
        self,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST request, returns parsed JSON."""
        return await self._request("POST", path, json=body, timeout=timeout)

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST multipart request for file uploads."""
        # Remove Content-Type header for multipart (httpx sets it with boundary)
        headers = {"Authorization": f"Bearer {self._config.api_key}"}

        try:
            response = await self._client.post(
                path,
                data=data,
                files=files,
                headers=headers,
                timeout=timeout or self._config.timeout,
            )
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}") from e

        self._raise_for_status(response)
        return response.json()

    async def get(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """GET request, returns parsed JSON."""
        return await self._request("GET", path, timeout=timeout)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute request and handle errors."""
        try:
            response = await self._client.request(
                method,
                path,
                timeout=timeout or self._config.timeout,
                **kwargs,
            )
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}") from e

        self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes to exceptions."""
        if response.is_success:
            return

        status = response.status_code
        message = ""
        response_data: dict[str, Any] | None = None

        try:
            response_data = response.json()
            detail = response_data.get("detail", "")
            if isinstance(detail, str):
                message = detail
            elif isinstance(detail, dict):
                message = detail.get("message", str(detail))
            else:
                message = str(detail)

            if not message:
                error_obj = response_data.get("error", {})
                if isinstance(error_obj, dict):
                    message = error_obj.get("message", "")
        except Exception:
            message = response.text[:500] if response.text else ""

        error_kwargs: dict[str, Any] = {
            "status_code": status,
            "response": response_data,
        }

        if status == 401:
            raise AuthenticationError(
                message or "Invalid API key", **error_kwargs
            )
        elif status == 403:
            raise PermissionDeniedError(
                message or "Permission denied", **error_kwargs
            )
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message or "Rate limit exceeded",
                retry_after=float(retry_after) if retry_after else None,
                **error_kwargs,
            )
        elif status == 400:
            raise BadRequestError(message or "Bad request", **error_kwargs)
        elif status == 502:
            raise ProviderError(message or "Provider error", **error_kwargs)
        elif status >= 500:
            raise ServerError(message or "Server error", **error_kwargs)
        else:
            raise AssistantsError(f"HTTP {status}: {message}", **error_kwargs)

    @asynccontextmanager
    async def stream_post(
        self,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[ChatStream]:
        """
        POST request with streaming response.

        Returns a ChatStream that can be iterated for deltas.

        Example:
            async with transport.stream_post("/v1/chat/completions", body=request) as stream:
                async for delta in stream:
                    print(delta.content, end="")
        """
        request_timeout = timeout or self._config.timeout

        try:
            async with self._client.stream(
                "POST",
                path,
                json=body,
                timeout=httpx.Timeout(request_timeout, read=None),  # No read timeout for streaming
            ) as response:
                # Check for errors before streaming
                if not response.is_success:
                    # Read response body for error message
                    await response.aread()
                    self._raise_for_status(response)

                yield ChatStream(response)

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}") from e

    async def close(self) -> None:
        """Close HTTP client and release connections."""
        await self._client.aclose()
