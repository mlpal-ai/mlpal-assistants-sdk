"""Exception hierarchy for mlpal-assistants SDK."""

from __future__ import annotations

from typing import Any


class AssistantsError(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(status={self.status_code})")
        if self.request_id:
            parts.append(f"[request_id={self.request_id}]")
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Authentication & Authorization
# ─────────────────────────────────────────────────────────────────


class AuthenticationError(AssistantsError):
    """Invalid or missing API key (401)."""


class PermissionDeniedError(AssistantsError):
    """API key lacks required permissions (403)."""


# ─────────────────────────────────────────────────────────────────
# Rate Limiting & Quotas
# ─────────────────────────────────────────────────────────────────


class RateLimitError(AssistantsError):
    """Rate limit exceeded (429)."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(AssistantsError):
    """Spending quota exceeded (429)."""


# ─────────────────────────────────────────────────────────────────
# Request Errors
# ─────────────────────────────────────────────────────────────────


class BadRequestError(AssistantsError):
    """Invalid request parameters (400)."""


class ValidationError(BadRequestError):
    """Request validation failed."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.errors = errors or []


# ─────────────────────────────────────────────────────────────────
# Resource Errors
# ─────────────────────────────────────────────────────────────────


class NotFoundError(AssistantsError):
    """Resource not found (404)."""


class ModelNotAvailableError(NotFoundError):
    """Model is unavailable or doesn't exist."""


class UnsupportedCapabilityError(BadRequestError):
    """Model doesn't support requested capability."""

    def __init__(
        self,
        message: str,
        *,
        suggested_models: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.suggested_models = suggested_models


# ─────────────────────────────────────────────────────────────────
# Server Errors
# ─────────────────────────────────────────────────────────────────


class ServerError(AssistantsError):
    """Server-side error (5xx)."""


class ServiceUnavailableError(ServerError):
    """Service temporarily unavailable (503)."""


class ProviderError(ServerError):
    """Upstream provider error (502)."""


# ─────────────────────────────────────────────────────────────────
# Network Errors
# ─────────────────────────────────────────────────────────────────


class TimeoutError(AssistantsError):
    """Request timed out."""


class ConnectionError(AssistantsError):
    """Failed to connect."""


# ─────────────────────────────────────────────────────────────────
# Tool & MCP Errors
# ─────────────────────────────────────────────────────────────────


class ToolExecutionError(AssistantsError):
    """Tool execution failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        original_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.original_error = original_error


class MaxTurnsExceededError(AssistantsError):
    """Agentic run exceeded max turns."""

    def __init__(
        self,
        message: str,
        *,
        turns_completed: int,
        max_turns: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.turns_completed = turns_completed
        self.max_turns = max_turns


class MCPError(AssistantsError):
    """MCP-related error."""


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""

    def __init__(
        self,
        message: str,
        *,
        server: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.server = server


class MCPToolNotFoundError(MCPError):
    """MCP tool not found."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
