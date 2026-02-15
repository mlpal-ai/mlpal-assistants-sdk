"""MCP bridge - integrates MCP servers with the tool system.

This module provides integration between the mlpal-assistants SDK and
MCP (Model Context Protocol) servers. It allows loading tools from MCP
servers and using them in chat completions.

Requires the 'mcp' optional dependency:
    pip install mlpal-assistants[mcp]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlpal_assistants.exceptions import MCPConnectionError, MCPError, MCPToolNotFoundError
from mlpal_assistants.tools.registry import ToolDefinition, ToolRegistry

if TYPE_CHECKING:
    pass


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
    timeout: float = 30.0


@dataclass
class MCPTool:
    """A tool loaded from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name (server_tool).

        Uses underscore separator to comply with OpenAI's tool name pattern:
        ^[a-zA-Z0-9_-]+$
        """
        # Sanitize both parts to ensure valid tool names
        safe_server = self.server_name.replace(".", "_").replace("-", "_")
        safe_name = self.name.replace(".", "_").replace("-", "_")
        return f"{safe_server}_{safe_name}"


class MCPBridge:
    """
    Bridge between MCP servers and the mlpal-assistants tool system.

    Provides:
    - Connection management to MCP servers
    - Tool discovery and loading
    - Tool execution via MCP protocol
    - Conversion to ToolRegistry for use with ChatService

    Example:
        # Create bridge and connect to MCP server
        bridge = MCPBridge()
        await bridge.connect_stdio(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )

        # Load tools and use with assistant
        tools = await bridge.get_tool_registry()

        response = await assistant.chat(
            messages=[{"role": "user", "content": "List files in /tmp"}],
            tools=tools,
        )

        await bridge.close()
    """

    def __init__(self) -> None:
        self._servers: dict[str, Any] = {}  # name -> session
        self._tools: dict[str, MCPTool] = {}  # qualified_name -> tool
        self._connections: dict[str, Any] = {}  # For cleanup

    async def connect_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """
        Connect to an MCP server via stdio transport.

        Args:
            name: Unique name for this server connection
            command: Command to run (e.g., "npx", "python")
            args: Command arguments
            env: Environment variables
            cwd: Working directory
        """
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as err:
            raise MCPError(
                "MCP package not installed. Install with: pip install mlpal-assistants[mcp]"
            ) from err

        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
        )

        exit_stack = AsyncExitStack()

        try:
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(params)
            )
            read_stream, write_stream = stdio_transport

            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await session.initialize()

            self._servers[name] = session
            self._connections[name] = exit_stack

            # Load tools from this server
            await self._load_tools_from_server(name, session)

        except Exception as e:
            await exit_stack.aclose()
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{name}': {e}",
                server=name,
            ) from e

    async def connect_http(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Connect to an MCP server via HTTP transport.

        Args:
            name: Unique name for this server connection
            url: HTTP URL of the MCP server
            headers: Optional HTTP headers (for auth, etc.)
        """
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as err:
            raise MCPError(
                "MCP package not installed. Install with: pip install mlpal-assistants[mcp]"
            ) from err

        exit_stack = AsyncExitStack()

        try:
            http_transport = await exit_stack.enter_async_context(
                streamablehttp_client(url, headers=headers or {})
            )
            read_stream, write_stream, _ = http_transport

            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await session.initialize()

            self._servers[name] = session
            self._connections[name] = exit_stack

            # Load tools from this server
            await self._load_tools_from_server(name, session)

        except Exception as e:
            await exit_stack.aclose()
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{name}': {e}",
                server=name,
            ) from e

    async def _load_tools_from_server(
        self,
        server_name: str,
        session: Any,
    ) -> None:
        """Load tools from an MCP server."""
        response = await session.list_tools()

        for tool in response.tools:
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                server_name=server_name,
            )
            self._tools[mcp_tool.qualified_name] = mcp_tool

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Call a tool on an MCP server.

        Args:
            name: Tool name (qualified like "server.tool" or unqualified)
            arguments: Tool arguments

        Returns:
            Tool result as string

        Raises:
            MCPToolNotFoundError: If tool is not found
        """
        # Resolve tool name
        tool = self._resolve_tool(name)
        if tool is None:
            raise MCPToolNotFoundError(
                f"Tool not found: {name}",
                tool_name=name,
            )

        # Get session for this server
        session = self._servers.get(tool.server_name)
        if session is None:
            raise MCPConnectionError(
                f"Server not connected: {tool.server_name}",
                server=tool.server_name,
            )

        # Call the tool
        result = await session.call_tool(tool.name, arguments)

        # Extract content from result
        if hasattr(result, 'content') and result.content:
            contents = []
            for item in result.content:
                if hasattr(item, 'text'):
                    contents.append(item.text)
            return "\n".join(contents)

        return str(result)

    def _resolve_tool(self, name: str) -> MCPTool | None:
        """Resolve a tool name to an MCPTool."""
        # Try qualified name first
        if name in self._tools:
            return self._tools[name]

        # Try unqualified name (find first match)
        for _qualified_name, tool in self._tools.items():
            if tool.name == name:
                return tool

        return None

    def get_tool_registry(self) -> ToolRegistry:
        """
        Get a ToolRegistry containing all MCP tools.

        The registry wraps MCP tool calls so they can be used
        with the standard tool execution system.

        Returns:
            ToolRegistry with MCP tools
        """
        registry = ToolRegistry()

        for qualified_name, tool in self._tools.items():
            # Create a wrapper function for this tool
            wrapper = self._create_tool_wrapper(tool)

            # Register with the tool's schema
            registry._tools[qualified_name] = ToolDefinition(
                name=qualified_name,
                func=wrapper,
                description=tool.description,
                parameters=tool.input_schema,
            )

        return registry

    def _create_tool_wrapper(self, tool: MCPTool) -> Any:
        """Create a wrapper function for an MCP tool."""
        bridge = self

        async def wrapper(**kwargs: Any) -> str:
            return await bridge.call_tool(tool.qualified_name, kwargs)

        # Set metadata for schema generation
        wrapper.__name__ = tool.name
        wrapper.__doc__ = tool.description

        return wrapper

    @property
    def connected_servers(self) -> list[str]:
        """List of connected server names."""
        return list(self._servers.keys())

    @property
    def available_tools(self) -> list[str]:
        """List of available tool names (qualified)."""
        return list(self._tools.keys())

    async def refresh_tools(self, server_name: str | None = None) -> None:
        """
        Refresh tool list from server(s).

        Args:
            server_name: Specific server to refresh, or None for all
        """
        if server_name:
            if server_name not in self._servers:
                raise MCPConnectionError(
                    f"Server not connected: {server_name}",
                    server=server_name,
                )
            # Remove old tools from this server
            self._tools = {
                k: v for k, v in self._tools.items()
                if v.server_name != server_name
            }
            await self._load_tools_from_server(
                server_name, self._servers[server_name]
            )
        else:
            # Refresh all servers
            self._tools.clear()
            for name, session in self._servers.items():
                await self._load_tools_from_server(name, session)

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from a specific server."""
        if server_name in self._connections:
            await self._connections[server_name].aclose()
            del self._connections[server_name]

        if server_name in self._servers:
            del self._servers[server_name]

        # Remove tools from this server
        self._tools = {
            k: v for k, v in self._tools.items()
            if v.server_name != server_name
        }

    async def close(self) -> None:
        """Close all server connections."""
        for exit_stack in self._connections.values():
            try:
                await exit_stack.aclose()
            except Exception:
                pass

        self._servers.clear()
        self._connections.clear()
        self._tools.clear()

    async def connect_from_config(
        self,
        config_path: str | None = None,
        server_names: list[str] | None = None,
    ) -> None:
        """
        Connect to MCP servers defined in mlpal.json config.

        Auto-discovers mlpal.json in current directory and parents,
        or loads from explicit path.

        Args:
            config_path: Explicit config file path (default: auto-discover)
            server_names: Specific servers to connect (default: all)

        Example:
            async with MCPBridge() as bridge:
                # Auto-discover and connect all servers
                await bridge.connect_from_config()

                # Or connect specific servers only
                await bridge.connect_from_config(
                    server_names=["perplexity", "filesystem"]
                )

                tools = bridge.get_tool_registry()
        """
        from mlpal_assistants.tools.mcp_config import MCPConfig, connect_from_config

        config: MCPConfig | None
        if config_path:
            config = MCPConfig.from_file(config_path)
        else:
            config = MCPConfig.discover()

        if config:
            await connect_from_config(self, config, server_names)

    async def __aenter__(self) -> MCPBridge:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()
