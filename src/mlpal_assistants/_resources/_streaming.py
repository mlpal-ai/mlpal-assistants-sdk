"""Streaming iterators over the native `/v1/messages` SSE response.

`MessageStream` / `AsyncMessageStream` own the httpx stream context and feed
its lines through `SSEDecoder`, yielding one `ServerSentEvent` per Anthropic
event. Both are context managers so the underlying connection is always closed,
and heartbeat comments (`: ping`) are transparently skipped by the decoder.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

from mlpal_assistants._sse import ServerSentEvent, SSEDecoder

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager, AbstractContextManager

    import httpx


class MessageStream:
    def __init__(self, stream_cm: AbstractContextManager[httpx.Response]) -> None:
        self._cm = stream_cm
        self._response: httpx.Response | None = None

    def __enter__(self) -> MessageStream:
        self._response = self._cm.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._cm.__exit__(*exc)

    def __iter__(self) -> Iterator[ServerSentEvent]:
        response = self._response or self._cm.__enter__()
        self._response = response
        decoder = SSEDecoder()
        for line in response.iter_lines():
            event = decoder.push(line)
            if event is not None:
                yield event

    def close(self) -> None:
        self._cm.__exit__(None, None, None)


class AsyncMessageStream:
    def __init__(self, stream_cm: AbstractAsyncContextManager[httpx.Response]) -> None:
        self._cm = stream_cm
        self._response: httpx.Response | None = None

    async def __aenter__(self) -> AsyncMessageStream:
        self._response = await self._cm.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._cm.__aexit__(*exc)

    async def __aiter__(self) -> AsyncIterator[ServerSentEvent]:
        response = self._response or await self._cm.__aenter__()
        self._response = response
        decoder = SSEDecoder()
        async for line in response.aiter_lines():
            event = decoder.push(line)
            if event is not None:
                yield event

    async def close(self) -> None:
        await self._cm.__aexit__(None, None, None)
