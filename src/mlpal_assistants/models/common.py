"""Common types shared across the SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> TokenUsage:
        """Create from API response."""
        return cls(
            input_tokens=data.get("input_tokens", 0) or data.get("prompt_tokens", 0),
            output_tokens=data.get("output_tokens", 0) or data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("cached_tokens"),
        )


@dataclass(frozen=True, slots=True)
class CostInfo:
    """Cost and usage information."""

    model_name: str
    provider: str
    tokens: TokenUsage
    latency_ms: int
    compute_units: float

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> CostInfo:
        """Create from API response."""
        tokens_data = data.get("tokens", data.get("usage", {}))

        # Handle case where tokens is just an integer (embeddings API)
        if isinstance(tokens_data, int):
            tokens_data = {"total_tokens": tokens_data}

        return cls(
            model_name=data.get("model_name", data.get("model", "")),
            provider=data.get("provider", ""),
            tokens=TokenUsage.from_api(tokens_data),
            latency_ms=data.get("latency_ms", 0),
            compute_units=data.get("compute_units", 0.0),
        )


@dataclass(frozen=True, slots=True)
class RoutingInfo:
    """Routing metadata for meta-model resolution."""

    requested_model: str
    resolved_model: str
    provider: str
    strategy: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RoutingInfo:
        """Create from API response."""
        return cls(
            requested_model=data.get("requested_model", ""),
            resolved_model=data.get("resolved_model", ""),
            provider=data.get("resolved_provider", data.get("provider", "")),
            strategy=data.get("strategy"),
        )
