"""`client.messages` — native inference over `/v1/messages` (Anthropic wire).

The canonical inference surface. The request is the Anthropic Messages shape;
core params are explicit for ergonomics and type-checking, while `extra_body`
carries any newer/provider-specific field straight through (the gateway is a
faithful passthrough). Every configured model — OpenAI, Google, Bedrock, or a
custom adapter — is reachable here regardless of which provider actually runs;
the response always comes back in this Anthropic shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from mlpal_assistants._resources._streaming import AsyncMessageStream, MessageStream
from mlpal_assistants._wire import CostBreakdown, Message

if TYPE_CHECKING:
    import httpx

    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_PATH = "/v1/messages"


def _cost_from_headers(headers: httpx.Headers) -> CostBreakdown | None:
    """Parse the gateway's X-MLPal-* CU headers (absent on non-gateway or
    older deployments — then cost stays None)."""

    def _f(name: str) -> float | None:
        raw = headers.get(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    total = _f("x-mlpal-compute-units")
    provider = _f("x-mlpal-provider-cu")
    infra = _f("x-mlpal-infra-cu")
    if total is None and provider is None and infra is None:
        return None
    return CostBreakdown(
        compute_units=total, provider_compute_units=provider, infra_compute_units=infra
    )


def _build_body(
    *,
    model: str,
    messages: Iterable[Mapping[str, Any]],
    max_tokens: int,
    stream: bool,
    system: str | Iterable[Mapping[str, Any]] | None,
    tools: Iterable[Mapping[str, Any]] | None,
    tool_choice: Mapping[str, Any] | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    stop_sequences: Iterable[str] | None,
    thinking: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
    }
    optional = {
        "stream": stream or None,  # omit when False; the wire treats absence as non-stream
        "system": system,
        "tools": list(tools) if tools is not None else None,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop_sequences": list(stop_sequences) if stop_sequences is not None else None,
        "thinking": thinking,
        "metadata": metadata,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    if extra_body:
        body.update(extra_body)
    return body


class Messages:
    """Synchronous native-inference resource."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        max_tokens: int,
        system: str | Iterable[Mapping[str, Any]] | None = None,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Iterable[str] | None = None,
        thinking: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Message:
        body = _build_body(
            model=model, messages=messages, max_tokens=max_tokens, stream=False,
            system=system, tools=tools, tool_choice=tool_choice, temperature=temperature,
            top_p=top_p, top_k=top_k, stop_sequences=stop_sequences, thinking=thinking,
            metadata=metadata, extra_body=extra_body,
        )
        data, resp_headers = self._transport.request(
            "POST", _PATH, json=body, headers=extra_headers, timeout=timeout
        )
        msg = Message.model_validate(data)
        msg.cost = _cost_from_headers(resp_headers)
        return msg

    def stream(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        max_tokens: int,
        system: str | Iterable[Mapping[str, Any]] | None = None,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Iterable[str] | None = None,
        thinking: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> MessageStream:
        """Open an SSE stream. Use as a context manager and iterate events:

        ``with client.messages.stream(...) as s:``
        ``    for event in s: ...``
        """
        body = _build_body(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            system=system, tools=tools, tool_choice=tool_choice, temperature=temperature,
            top_p=top_p, top_k=top_k, stop_sequences=stop_sequences, thinking=thinking,
            metadata=metadata, extra_body=extra_body,
        )
        cm = self._transport.stream("POST", _PATH, json=body, headers=extra_headers, timeout=timeout)
        return MessageStream(cm)


class AsyncMessages:
    """Asynchronous native-inference resource."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        max_tokens: int,
        system: str | Iterable[Mapping[str, Any]] | None = None,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Iterable[str] | None = None,
        thinking: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Message:
        body = _build_body(
            model=model, messages=messages, max_tokens=max_tokens, stream=False,
            system=system, tools=tools, tool_choice=tool_choice, temperature=temperature,
            top_p=top_p, top_k=top_k, stop_sequences=stop_sequences, thinking=thinking,
            metadata=metadata, extra_body=extra_body,
        )
        data, resp_headers = await self._transport.request(
            "POST", _PATH, json=body, headers=extra_headers, timeout=timeout
        )
        msg = Message.model_validate(data)
        msg.cost = _cost_from_headers(resp_headers)
        return msg

    def stream(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        max_tokens: int,
        system: str | Iterable[Mapping[str, Any]] | None = None,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Iterable[str] | None = None,
        thinking: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncMessageStream:
        """Open an SSE stream. Use as an async context manager:

        ``async with client.messages.stream(...) as s:``
        ``    async for event in s: ...``
        """
        body = _build_body(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            system=system, tools=tools, tool_choice=tool_choice, temperature=temperature,
            top_p=top_p, top_k=top_k, stop_sequences=stop_sequences, thinking=thinking,
            metadata=metadata, extra_body=extra_body,
        )
        cm = self._transport.stream("POST", _PATH, json=body, headers=extra_headers, timeout=timeout)
        return AsyncMessageStream(cm)
