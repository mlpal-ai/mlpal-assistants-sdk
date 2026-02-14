"""MCP configuration loader for mlpal.json files.

Supports automatic discovery and parsing of mlpal.json config files,
including environment variable interpolation.

Example mlpal.json:
    {
      "mcpServers": {
        "perplexity": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "server-perplexity-ask"],
          "env": {
            "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
          }
        },
        "api-server": {
          "transport": "http",
          "url": "http://localhost:8080/mcp",
          "headers": {
            "Authorization": "Bearer ${API_TOKEN}"
          }
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlpal_assistants.exceptions import MCPError


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: str  # "stdio" or "http"
    command: str | None = None  # For stdio transport
    args: list[str] = field(default_factory=list)
    url: str | None = None  # For http transport
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class MCPConfig:
    """Parsed mlpal.json configuration."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> MCPConfig:
        """Load config from a specific file path."""
        path = Path(path)
        if not path.exists():
            raise MCPError(f"Config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls._parse_config(data, path)

    @classmethod
    def discover(cls, start_dir: str | Path | None = None) -> MCPConfig | None:
        """
        Discover mlpal.json by searching current directory and parents.

        Args:
            start_dir: Directory to start searching from (default: cwd)

        Returns:
            MCPConfig if found, None otherwise
        """
        start = Path(start_dir) if start_dir else Path.cwd()

        # Search current directory and parents
        for directory in [start, *start.parents]:
            config_path = directory / "mlpal.json"
            if config_path.exists():
                return cls.from_file(config_path)

        return None

    @classmethod
    def _parse_config(cls, data: dict[str, Any], config_path: Path) -> MCPConfig:
        """Parse config data into MCPConfig."""
        servers: dict[str, MCPServerConfig] = {}

        mcp_servers = data.get("mcpServers", {})
        for name, server_data in mcp_servers.items():
            servers[name] = cls._parse_server_config(name, server_data)

        return cls(servers=servers, config_path=config_path)

    @classmethod
    def _parse_server_config(
        cls, name: str, data: dict[str, Any]
    ) -> MCPServerConfig:
        """Parse a single server configuration."""
        transport = data.get("transport", "stdio")

        # Interpolate environment variables
        env = cls._interpolate_dict(data.get("env", {}))
        headers = cls._interpolate_dict(data.get("headers", {}))
        url = cls._interpolate_string(data.get("url")) if data.get("url") else None

        return MCPServerConfig(
            name=name,
            transport=transport,
            command=data.get("command"),
            args=data.get("args", []),
            url=url,
            env=env,
            headers=headers,
            cwd=data.get("cwd"),
        )

    @classmethod
    def _interpolate_string(cls, value: str | None) -> str | None:
        """Interpolate environment variables in a string.

        Supports ${VAR_NAME} syntax.
        """
        if value is None:
            return None

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return re.sub(r"\$\{([^}]+)\}", replace_var, value)

    @classmethod
    def _interpolate_dict(cls, data: dict[str, str]) -> dict[str, str]:
        """Interpolate environment variables in all dict values."""
        return {
            key: cls._interpolate_string(value) or ""
            for key, value in data.items()
        }


async def connect_from_config(
    bridge: Any,  # MCPBridge - avoiding circular import
    config: MCPConfig | None = None,
    server_names: list[str] | None = None,
) -> None:
    """
    Connect MCPBridge to servers defined in config.

    Args:
        bridge: MCPBridge instance to connect servers to
        config: MCPConfig to use (default: auto-discover)
        server_names: Specific servers to connect (default: all)

    Example:
        from mlpal_assistants.tools import MCPBridge
        from mlpal_assistants.tools.mcp_config import MCPConfig, connect_from_config

        async with MCPBridge() as bridge:
            config = MCPConfig.discover()
            if config:
                await connect_from_config(bridge, config)

            tools = bridge.get_tool_registry()
    """
    if config is None:
        config = MCPConfig.discover()
        if config is None:
            return  # No config found, nothing to connect

    servers_to_connect = (
        {name: config.servers[name] for name in server_names if name in config.servers}
        if server_names
        else config.servers
    )

    for name, server_config in servers_to_connect.items():
        if server_config.transport == "stdio":
            if not server_config.command:
                raise MCPError(f"Server '{name}' has transport 'stdio' but no command")

            await bridge.connect_stdio(
                name=name,
                command=server_config.command,
                args=server_config.args,
                env=server_config.env or None,
                cwd=server_config.cwd,
            )
        elif server_config.transport == "http":
            if not server_config.url:
                raise MCPError(f"Server '{name}' has transport 'http' but no url")

            await bridge.connect_http(
                name=name,
                url=server_config.url,
                headers=server_config.headers or None,
            )
        else:
            raise MCPError(
                f"Server '{name}' has unknown transport: {server_config.transport}"
            )
