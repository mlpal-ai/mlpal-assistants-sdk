"""Tests for the v2 client's catalog, feedback, and account-usage resources."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from mlpal_assistants import AsyncMLPal, MLPal
from mlpal_assistants.wire import Catalog, DailyUsage, FeedbackResult, UsageSummary

BASE = "http://gw.test"


def _client() -> MLPal:
    return MLPal(api_key="mlpal_sk_test", base_url=BASE)


# ── catalog ──────────────────────────────────────────────────────────────────

_CATALOG = {
    "schema": 1,
    "profile": "coding",
    "updated": "2026-08-01",
    "quality_dimensions": ["coding", "reasoning"],
    "flagships": {"anthropic": "claude-opus-5"},
    "benchmark_rankings": {"coding": []},
    "routing_ladder": ["cheap", "mid", "frontier"],
    "models": {"claude-opus-5": {"provider": "anthropic"}},
    "tiers": {
        "frontier": {"model": "claude-opus-5", "rel_cost": 100, "served_alternate": False},
        "cheap": {"model": "gpt-5-nano", "rel_cost": 4},
    },
}


@respx.mock
def test_catalog_retrieve_passes_profile():
    route = respx.get(f"{BASE}/v2/catalog").mock(return_value=httpx.Response(200, json=_CATALOG))
    with _client() as client:
        catalog = client.catalog.retrieve(profile="coding")

    assert isinstance(catalog, Catalog)
    assert catalog.profile == "coding"
    assert catalog.routing_ladder == ["cheap", "mid", "frontier"]
    assert catalog.tiers["frontier"]["model"] == "claude-opus-5"
    assert route.calls.last.request.url.params["profile"] == "coding"


@pytest.mark.asyncio
@respx.mock
async def test_catalog_retrieve_async():
    respx.get(f"{BASE}/v2/catalog").mock(return_value=httpx.Response(200, json=_CATALOG))
    async with AsyncMLPal(api_key="mlpal_sk_test", base_url=BASE) as client:
        catalog = await client.catalog.retrieve(profile="coding")
    assert catalog.tiers["cheap"]["rel_cost"] == 4


# ── feedback ─────────────────────────────────────────────────────────────────


@respx.mock
def test_feedback_create():
    route = respx.post(f"{BASE}/v2/feedback").mock(
        return_value=httpx.Response(200, json={"recorded": True})
    )
    with _client() as client:
        result = client.feedback.create(
            model="gpt-5-nano",
            task_type="coding",
            outcome="escalated",
            escalated_to="claude-opus-5",
            trace_id="trace-123",
        )
    assert isinstance(result, FeedbackResult)
    assert result.recorded is True

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "gpt-5-nano",
        "task_type": "coding",
        "outcome": "escalated",
        "escalated_to": "claude-opus-5",
        "trace_id": "trace-123",
    }


@respx.mock
def test_feedback_omits_optional_fields():
    route = respx.post(f"{BASE}/v2/feedback").mock(
        return_value=httpx.Response(200, json={"recorded": True})
    )
    with _client() as client:
        client.feedback.create(model="claude-opus-5", task_type="reasoning", outcome="accepted")
    sent = json.loads(route.calls.last.request.content)
    assert "escalated_to" not in sent
    assert "trace_id" not in sent


# ── account usage ────────────────────────────────────────────────────────────


@respx.mock
def test_usage_summary():
    respx.get(f"{BASE}/v1/usage/summary").mock(
        return_value=httpx.Response(
            200,
            json={
                "period_start": "2026-07-01T00:00:00Z",
                "period_end": "2026-08-01T00:00:00Z",
                "total_requests": 42,
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_compute_units": 3.5,
                "quota_limit": 100.0,
                "quota_remaining": 96.5,
                "by_model": {
                    "claude-opus-5": {
                        "model_tag": "claude-opus-5",
                        "requests": 42,
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "compute_units": 3.5,
                    }
                },
            },
        )
    )
    with _client() as client:
        summary = client.usage.summary()
    assert isinstance(summary, UsageSummary)
    assert summary.total_requests == 42
    assert summary.quota_remaining == 96.5
    assert summary.by_model["claude-opus-5"].compute_units == 3.5


@respx.mock
def test_usage_daily_passes_days():
    route = respx.get(f"{BASE}/v1/usage/daily").mock(
        return_value=httpx.Response(
            200,
            json={
                "period_start": "2026-07-25T00:00:00Z",
                "period_end": "2026-08-01T00:00:00Z",
                "days": 7,
                "total_compute_units": 1.25,
                "daily": [
                    {"date": "2026-07-25", "requests": 3, "input_tokens": 10,
                     "output_tokens": 5, "compute_units": 0.5}
                ],
            },
        )
    )
    with _client() as client:
        daily = client.usage.daily(days=7)
    assert isinstance(daily, DailyUsage)
    assert daily.days == 7
    assert daily.daily[0].date == "2026-07-25"
    assert route.calls.last.request.url.params["days"] == "7"
