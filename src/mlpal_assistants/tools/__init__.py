"""Tool execution module for mlpal-assistants SDK.

This module provides:
- ToolRegistry: Register and manage callable tools
- ToolExecutor: Execute tool calls with validation
- AgentLoop: Multi-turn orchestration for agentic workflows
- MCPBridge: Integration with MCP servers (optional)
"""

from mlpal_assistants.tools.registry import ToolDefinition, ToolRegistry
from mlpal_assistants.tools.executor import ToolExecutor
from mlpal_assistants.tools.result import ToolResult
from mlpal_assistants.tools.loop import AgentLoop

# Lazy import for MCP bridge (requires optional dependency)
def __getattr__(name: str):
    if name == "MCPBridge":
        from mlpal_assistants.tools.mcp_bridge import MCPBridge
        return MCPBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
    "AgentLoop",
    "MCPBridge",
]
