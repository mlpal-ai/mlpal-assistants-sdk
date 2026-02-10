"""Tests for configuration."""

import os

import pytest

from mlpal_assistants.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_from_env_with_explicit_key(self):
        """Test creating config with explicit API key."""
        config = Config.from_env(api_key="test_key")
        assert config.api_key == "test_key"
        assert config.base_url == "https://models.mlpal.ai"

    def test_from_env_missing_key_raises(self):
        """Test that missing API key raises ValueError."""
        old_key = os.environ.pop("MLPAL_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                Config.from_env()
        finally:
            if old_key:
                os.environ["MLPAL_API_KEY"] = old_key

    def test_from_env_with_env_var(self, monkeypatch):
        """Test creating config from environment variable."""
        monkeypatch.setenv("MLPAL_API_KEY", "env_key")
        config = Config.from_env()
        assert config.api_key == "env_key"

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base URL."""
        config = Config.from_env(
            api_key="key", base_url="https://example.com/"
        )
        assert config.base_url == "https://example.com"

    def test_timeout_override(self):
        """Test timeout override."""
        config = Config.from_env(api_key="key", timeout=60.0)
        assert config.timeout == 60.0

    def test_default_timeout(self):
        """Test default timeout."""
        config = Config.from_env(api_key="key")
        assert config.timeout == 120.0

    def test_config_is_immutable(self):
        """Test that config is immutable (frozen dataclass)."""
        config = Config.from_env(api_key="key")
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore

    def test_timeout_from_env_var(self, monkeypatch):
        """Test timeout from environment variable."""
        monkeypatch.setenv("MLPAL_API_KEY", "key")
        monkeypatch.setenv("MLPAL_TIMEOUT", "30")
        config = Config.from_env()
        assert config.timeout == 30.0
