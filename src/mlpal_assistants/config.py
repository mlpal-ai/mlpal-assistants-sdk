"""Configuration for mlpal-assistants SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://models.mlpal.ai"
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable SDK configuration."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> Config:
        """
        Create configuration from environment variables and overrides.

        Environment variables:
            MLPAL_API_KEY: API key (required if not provided)
            MLPAL_BASE_URL: Base URL for API (optional)
            MLPAL_TIMEOUT: Request timeout in seconds (optional)
        """
        resolved_api_key = api_key or os.environ.get("MLPAL_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "API key required. Pass api_key= or set MLPAL_API_KEY environment variable."
            )

        resolved_base_url = (
            base_url or os.environ.get("MLPAL_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")

        if timeout is not None:
            resolved_timeout = timeout
        elif env_timeout := os.environ.get("MLPAL_TIMEOUT"):
            resolved_timeout = float(env_timeout)
        else:
            resolved_timeout = DEFAULT_TIMEOUT

        return cls(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
        )
