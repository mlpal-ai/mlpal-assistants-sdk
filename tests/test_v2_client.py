"""Tests for the v2 MLPal client (native inference + management surface).

Uses respx to mock the gateway at the httpx-transport layer, so the real
request-building, retry, SSE-decoding, and error-mapping code all execute.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from mlpal_assistants import AsyncMLPal, MLPal
from mlpal_assistants.exceptions import (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from mlpal_assistants.wire import Message

BASE = "http://gw.test"

_MESSAGE_BODY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "Hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 4},
}


def _client() -> MLPal:
    return MLPal(api_key="mlpal_sk_test", base_url=BASE)


# ── native inference (sync) ──────────────────────────────────────────────────


@respx.mock
def test_messages_create_sync():
    route = respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(200, json=_MESSAGE_BODY)
    )
    with _client() as client:
        msg = client.messages.create(
            model="mlpal",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert isinstance(msg, Message)
    assert msg.text == "Hello there"
    assert msg.usage.input_tokens == 10
    assert msg.usage.cache_read_input_tokens == 4

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "mlpal",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1024,
    }
    # non-stream requests must not carry a stream flag
    assert "stream" not in sent
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer mlpal_sk_test"
    assert req.headers["anthropic-version"]  # native wire version header present


@respx.mock
def test_messages_create_forwards_optional_and_extra_body():
    route = respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(200, json=_MESSAGE_BODY)
    )
    with _client() as client:
        client.messages.create(
            model="mlpal",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
            system="Be terse.",
            temperature=0.2,
            stop_sequences=["STOP"],
            thinking={"type": "enabled", "budget_tokens": 128},
            extra_body={"top_logprobs": 3},
        )
    sent = json.loads(route.calls.last.request.content)
    assert sent["system"] == "Be terse."
    assert sent["temperature"] == 0.2
    assert sent["stop_sequences"] == ["STOP"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 128}
    assert sent["top_logprobs"] == 3  # extra_body passthrough


# ── native inference (async) ─────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_messages_create_async():
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(200, json=_MESSAGE_BODY))
    async with AsyncMLPal(api_key="mlpal_sk_test", base_url=BASE) as client:
        msg = await client.messages.create(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.text == "Hello there"


# ── streaming (Anthropic SSE, with heartbeats) ───────────────────────────────

_SSE = (
    b"event: message_start\n"
    b'data: {"type": "message_start", "message": {"id": "msg_01"}}\n\n'
    b": ping\n\n"  # heartbeat — must be skipped
    b"event: content_block_delta\n"
    b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n'
    b"event: message_stop\n"
    b'data: {"type": "message_stop"}\n\n'
)


@respx.mock
def test_messages_stream_sync():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_SSE
        )
    )
    with (
        _client() as client,
        client.messages.stream(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        ) as stream,
    ):
        types = [event.type for event in stream]

    # heartbeat dropped; three real events survive, in order
    assert types == ["message_start", "content_block_delta", "message_stop"]


@pytest.mark.asyncio
@respx.mock
async def test_messages_stream_async():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_SSE
        )
    )
    async with AsyncMLPal(api_key="mlpal_sk_test", base_url=BASE) as client:
        collected = []
        async with client.messages.stream(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        ) as stream:
            async for event in stream:
                collected.append(event.type)
    assert collected == ["message_start", "content_block_delta", "message_stop"]


# ── management surface (/admin/v1/keys) ──────────────────────────────────────


@respx.mock
def test_admin_keys_create():
    created = {
        "id": 7,
        "name": "team-web",
        "key_prefix": "mlpal_sk_abcd",
        "permissions": ["messages"],
        "is_active": True,
        "created_at": "2026-08-01T00:00:00Z",
        "model_policy": {"allow": ["claude-*"], "deny": []},
        "budgets": [{"unit": "usd", "amount": 100.0, "window": "month"}],
        "secret": "mlpal_sk_abcd1234deadbeef",
    }
    route = respx.post(f"{BASE}/admin/v1/keys").mock(return_value=httpx.Response(201, json=created))
    with _client() as client:
        key = client.admin.keys.create(
            name="team-web",
            permissions=["messages"],
            model_policy={"allow": ["claude-*"], "deny": []},
            budgets=[{"unit": "usd", "amount": 100.0, "window": "month"}],
        )
    assert key.id == 7
    assert key.secret == "mlpal_sk_abcd1234deadbeef"
    assert key.model_policy["allow"] == ["claude-*"]

    sent = json.loads(route.calls.last.request.content)
    assert sent["name"] == "team-web"
    assert sent["budgets"][0]["unit"] == "usd"


@respx.mock
def test_admin_keys_list_and_delete():
    respx.get(f"{BASE}/admin/v1/keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"id": 1, "name": "a", "key_prefix": "mlpal_sk_a", "permissions": []},
                    {"id": 2, "name": "b", "key_prefix": "mlpal_sk_b", "permissions": []},
                ],
                "total": 2,
            },
        )
    )
    delete_route = respx.delete(f"{BASE}/admin/v1/keys/2").mock(return_value=httpx.Response(204))
    with _client() as client:
        listing = client.admin.keys.list()
        assert listing.total == 2
        assert [k.id for k in listing.items] == [1, 2]
        client.admin.keys.delete(2)
    assert delete_route.called


# ── models (capability advertisement) ────────────────────────────────────────


@respx.mock
def test_models_list():
    respx.get(f"{BASE}/v1/messages/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "claude-opus-5",
                        "display_name": "Claude Opus 5",
                        "provider": "anthropic",
                        "max_output_tokens": 64000,
                        "capabilities": {"tools": True, "reasoning": True},
                    }
                ]
            },
        )
    )
    with _client() as client:
        models = client.models.list()
    assert len(models) == 1
    assert models[0].id == "claude-opus-5"
    assert models[0].capabilities["reasoning"] is True


# ── error mapping ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "body", "exc"),
    [
        (400, {"error": {"message": "bad shape"}}, BadRequestError),
        (401, {"error": {"message": "invalid key"}}, AuthenticationError),
        (403, {"detail": "no messages scope"}, PermissionDeniedError),
        (402, {"detail": "budget exceeded"}, QuotaExceededError),
        (502, {"error": {"message": "upstream 500"}}, ProviderError),
    ],
)
@respx.mock
def test_error_mapping(status, body, exc):
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(status, json=body))
    with _client() as client, pytest.raises(exc) as ei:
        client.messages.create(
            model="mlpal", max_tokens=8, messages=[{"role": "user", "content": "Hi"}]
        )
    assert ei.value.status_code == status
    assert ei.value.message  # human-readable message extracted from the envelope


@respx.mock
def test_429_budget_text_maps_to_quota_not_ratelimit():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(429, json={"detail": "monthly budget exceeded for this key"})
    )
    with _client() as client, pytest.raises(QuotaExceededError):
        client.messages.create(
            model="mlpal", max_tokens=8, messages=[{"role": "user", "content": "Hi"}]
        )


@respx.mock
def test_429_rate_limit_sets_retry_after():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}
        )
    )
    with _client() as client, pytest.raises(RateLimitError) as ei:
        client.messages.create(
            model="mlpal", max_tokens=8, messages=[{"role": "user", "content": "Hi"}]
        )
    assert ei.value.retry_after == 2.0


@respx.mock
def test_retries_then_succeeds_on_502():
    route = respx.post(f"{BASE}/v1/messages").mock(
        side_effect=[
            httpx.Response(502, json={"error": {"message": "transient"}}),
            httpx.Response(200, json=_MESSAGE_BODY),
        ]
    )
    with _client() as client:
        msg = client.messages.create(
            model="mlpal", max_tokens=8, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.text == "Hello there"
    assert route.call_count == 2


# ── CU cost header (X-MLPal-Compute-Units) ───────────────────────────────────


@respx.mock
def test_messages_compute_units_from_header():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200, json=_MESSAGE_BODY, headers={"X-MLPal-Compute-Units": "0.42"}
        )
    )
    with _client() as client:
        msg = client.messages.create(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.compute_units == pytest.approx(0.42)


@respx.mock
def test_messages_compute_units_absent_without_header():
    """Non-gateway upstreams send no X-MLPal header — compute_units stays None,
    never a fabricated 0."""
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(200, json=_MESSAGE_BODY))
    with _client() as client:
        msg = client.messages.create(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.compute_units is None


@respx.mock
def test_messages_compute_units_malformed_header_is_dropped():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200, json=_MESSAGE_BODY, headers={"X-MLPal-Compute-Units": "not-a-number"}
        )
    )
    with _client() as client:
        msg = client.messages.create(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.compute_units is None


@pytest.mark.asyncio
@respx.mock
async def test_messages_compute_units_from_header_async():
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200, json=_MESSAGE_BODY, headers={"X-MLPal-Compute-Units": "0.007"}
        )
    )
    async with AsyncMLPal(api_key="mlpal_sk_test", base_url=BASE) as client:
        msg = await client.messages.create(
            model="mlpal", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
    assert msg.compute_units == pytest.approx(0.007)


# --- 0.4.0: the OpenAI shape with the mlpal extra -----------------------------


def test_chat_response_parses_openai_shape_with_mlpal_extra():
    from mlpal_assistants.models.responses import ChatResponse

    data = {
        "id": "chatcmpl-trace-1",
        "object": "chat.completion",
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        "mlpal": {
            "cost": {
                "model_name": "gpt-6-luna",
                "provider": "openai",
                "tokens": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
                "latency_ms": 7,
                "compute_units": 0.000123,
            },
            "routing": {
                "requested_model": "mlpal",
                "resolved_model": "gpt-6-luna",
                "resolved_provider": "openai",
            },
            "data": {"a": 1},
            "trace_id": "trace-1",
            "metadata": {"max_output_tokens": {"applied": 16}},
        },
    }
    r = ChatResponse.from_api(data)
    assert r.content is None and r.finish_reason == "tool_calls"
    assert r.tool_calls and r.tool_calls[0].id == "call_1"
    assert r.cost.compute_units == 0.000123 and r.cost.tokens.input_tokens == 9
    assert r.routing.requested_model == "mlpal" and r.routing.resolved_model == "gpt-6-luna"
    assert r.metadata["trace_id"] == "trace-1" and r.metadata["max_output_tokens"]["applied"] == 16


def test_chat_response_openai_shape_without_extra_uses_usage():
    from mlpal_assistants.models.responses import ChatResponse

    data = {
        "model": "gpt-6-luna",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    r = ChatResponse.from_api(data)
    assert (
        r.content == "hi" and r.cost.tokens.total_tokens == 4 and r.cost.model_name == "gpt-6-luna"
    )


def test_flat_shape_still_parses():
    from mlpal_assistants.models.responses import ChatResponse

    r = ChatResponse.from_api(
        {"content": "ok", "finish_reason": "stop", "cost": {"model_name": "m", "provider": "p"}}
    )
    assert r.content == "ok" and r.routing.resolved_model == "m"


def test_every_request_declares_the_openai_wire():
    from mlpal_assistants._transport import build_headers
    from mlpal_assistants.config import Config

    headers = build_headers(Config(api_key="mlpal_sk_test"), None)
    assert headers["X-MLPal-Wire"] == "openai"
    assert headers["User-Agent"].startswith("mlpal-assistants-python/0.4")


def test_chat_transport_declares_the_openai_wire_too():
    """The chat facade (Assistant.chat / stream_chat) posts /v1/chat/completions
    through its own transport; that one must say which shape it speaks."""
    from mlpal_assistants.config import Config
    from mlpal_assistants.transport import HTTPTransport

    client = HTTPTransport(Config(api_key="mlpal_sk_test"))._client
    assert client.headers["x-mlpal-wire"] == "openai"
    assert client.headers["user-agent"].startswith("mlpal-assistants-python/0.4")
    assert client.headers["authorization"] == "Bearer mlpal_sk_test"
