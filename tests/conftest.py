"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def api_key():
    """Test API key."""
    return "test_api_key_12345"


@pytest.fixture
def base_url():
    """Test base URL."""
    return "https://test.mlpal.ai"
