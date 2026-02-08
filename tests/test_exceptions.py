"""Tests for exception classes."""

import pytest

from mlpal_assistants.exceptions import (
    AssistantsError,
    AuthenticationError,
    MaxTurnsExceededError,
    RateLimitError,
    ToolExecutionError,
    ValidationError,
)


class TestAssistantsError:
    """Tests for base AssistantsError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = AssistantsError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.status_code is None
        assert error.request_id is None

    def test_error_with_details(self):
        """Test error with all details."""
        error = AssistantsError(
            "Failed",
            status_code=500,
            response={"error": "internal"},
            request_id="req-123",
        )
        assert error.status_code == 500
        assert error.response == {"error": "internal"}
        assert error.request_id == "req-123"

    def test_str_representation(self):
        """Test string representation."""
        error = AssistantsError("Error", status_code=400, request_id="req-123")
        s = str(error)
        assert "Error" in s
        assert "400" in s
        assert "req-123" in s


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_with_retry_after(self):
        """Test rate limit error with retry-after."""
        error = RateLimitError("Rate limited", retry_after=30.0, status_code=429)
        assert error.retry_after == 30.0
        assert error.status_code == 429


class TestValidationError:
    """Tests for ValidationError."""

    def test_with_errors_list(self):
        """Test validation error with error list."""
        errors = [{"field": "name", "message": "required"}]
        error = ValidationError("Validation failed", errors=errors)
        assert error.errors == errors


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_with_original_error(self):
        """Test tool execution error with original exception."""
        original = ValueError("bad value")
        error = ToolExecutionError(
            "Tool failed",
            tool_name="my_tool",
            original_error=original,
        )
        assert error.tool_name == "my_tool"
        assert error.original_error is original


class TestMaxTurnsExceededError:
    """Tests for MaxTurnsExceededError."""

    def test_with_turn_info(self):
        """Test max turns error with turn counts."""
        error = MaxTurnsExceededError(
            "Too many turns",
            turns_completed=10,
            max_turns=10,
        )
        assert error.turns_completed == 10
        assert error.max_turns == 10
