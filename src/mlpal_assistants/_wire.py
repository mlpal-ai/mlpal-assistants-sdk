"""Typed models for the v2 native wire and the management surface.

The native inference wire (`/v2/messages`) is the Anthropic Messages shape,
passed through faithfully — so these models stay permissive (`extra="allow"`)
and any new provider/Anthropic field survives on the parsed object rather than
being dropped. Management models mirror the gateway's `/admin/v1/keys` schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Wire(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ── native inference (Anthropic Messages wire) ───────────────────────────────


class Usage(_Wire):
    """Provider-reported token counts. Fields are provider-faithful; absent
    fields stay None rather than being guessed (no estimated metrics)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class ContentBlock(_Wire):
    """A single content block. Kept permissive because block shapes vary by
    type (`text`, `tool_use`, `thinking`, …) and by provider."""

    type: str
    text: str | None = None


class Message(_Wire):
    """A non-streaming `/v2/messages` response (Anthropic Messages shape)."""

    id: str | None = None
    type: str | None = None
    role: str | None = None
    model: str | None = None
    content: list[ContentBlock] = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage | None = None

    @property
    def text(self) -> str:
        """Concatenation of the text blocks — the common ergonomic accessor.
        Non-text blocks (tool_use, thinking) are skipped; inspect `.content`
        for those."""
        return "".join(b.text for b in self.content if b.type == "text" and b.text)


# ── management (/admin/v1/keys) ──────────────────────────────────────────────


class ManagedKey(_Wire):
    """An API key as returned by the management surface (no secret)."""

    id: int
    name: str
    description: str | None = None
    key_prefix: str
    permissions: list[str] = Field(default_factory=list)
    rate_limit_tier: str | None = None
    is_active: bool = True
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    model_policy: dict[str, Any] | None = None
    budgets: list[dict[str, Any]] | None = None


class ManagedKeyWithSecret(ManagedKey):
    """A freshly-created key — carries the one-time plaintext `secret`."""

    secret: str


class ManagedKeyList(_Wire):
    items: list[ManagedKey] = Field(default_factory=list)
    total: int = 0


class KeyUsageSummary(_Wire):
    api_key_id: int
    period_start: datetime | None = None
    period_end: datetime | None = None
    total_requests: int = 0
    success_requests: int = 0
    error_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_compute_units: float = 0.0
    last_used_at: datetime | None = None


# ── catalog / capability advertisement (/v2/messages/models) ─────────────────


class ModelInfo(_Wire):
    id: str
    display_name: str | None = None
    provider: str | None = None
    max_output_tokens: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


# ── curated catalog (/v2/catalog) ────────────────────────────────────────────


class Catalog(_Wire):
    """The curated model view for a profile. Kept permissive because the
    curation shape evolves; the stable, routable pieces are surfaced as typed
    fields. `tiers` maps a tier name → its served model + attributes, and
    `routing_ladder` orders tiers cheapest → most capable — map a subtask's
    complexity estimate onto it to pick a model with no client-side "which model
    is current" logic."""

    profile: str
    updated: str | None = None
    quality_dimensions: list[str] = Field(default_factory=list)
    flagships: dict[str, Any] = Field(default_factory=dict)
    benchmark_rankings: dict[str, Any] = Field(default_factory=dict)
    routing_ladder: list[str] = Field(default_factory=list)
    models: dict[str, Any] = Field(default_factory=dict)
    tiers: dict[str, Any] = Field(default_factory=dict)


# ── feedback (/v2/feedback) ──────────────────────────────────────────────────


class FeedbackResult(_Wire):
    recorded: bool = False


# ── account usage (/v1/usage) ────────────────────────────────────────────────


class ModelUsage(_Wire):
    model_tag: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compute_units: float = 0.0


class UsageSummary(_Wire):
    period_start: datetime | None = None
    period_end: datetime | None = None
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_compute_units: float = 0.0
    quota_limit: float | None = None
    quota_remaining: float | None = None
    by_model: dict[str, ModelUsage] = Field(default_factory=dict)


class DailyUsageBucket(_Wire):
    date: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compute_units: float = 0.0


class DailyUsage(_Wire):
    period_start: datetime | None = None
    period_end: datetime | None = None
    days: int = 0
    total_compute_units: float = 0.0
    daily: list[DailyUsageBucket] = Field(default_factory=list)
