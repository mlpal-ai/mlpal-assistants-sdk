"""Tool registration and management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mlpal_assistants.tools.schema import generate_tool_schema


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """
    A registered tool definition.

    Attributes:
        name: Tool name (used in API requests)
        func: The callable to execute
        description: Human-readable description
        parameters: JSON Schema for input parameters
    """

    name: str
    func: Callable[..., Any]
    description: str
    parameters: dict[str, Any]

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API request format (OpenAI-compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Registry for managing callable tools.

    Provides tool registration, discovery, and API format conversion.
    Supports both explicit registration and decorator syntax.

    Example:
        # Explicit registration
        tools = ToolRegistry()
        tools.register("get_weather", get_weather_func)

        # Decorator syntax
        @tools.tool
        def get_time() -> str:
            '''Get current time.'''
            return datetime.now().isoformat()

        # From dict
        tools = ToolRegistry.from_dict({
            "search": search_func,
            "calculate": calculate_func,
        })

        # Use in chat
        response = await assistant.chat(
            messages=[...],
            tools=tools,
        )
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str | None = None,
        func: Callable[..., Any] | None = None,
        *,
        description: str | None = None,
    ) -> ToolRegistry | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Register a tool function.

        Can be used directly or as a decorator:
            tools.register("name", func)
            tools.register("name", func, description="...")

            @tools.register("name")
            def func(): ...

        Args:
            name: Tool name (defaults to function name)
            func: Function to register
            description: Override auto-generated description

        Returns:
            Self for chaining, or decorator if func is None
        """
        if func is None:
            # Decorator usage
            def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
                self._register_func(name or f.__name__, f, description)
                return f
            return decorator

        self._register_func(name or func.__name__, func, description)
        return self

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a tool.

        Example:
            @tools.tool
            def search(query: str) -> str:
                '''Search for information.'''
                return results

            @tools.tool(name="custom_name")
            def func(): ...

        Args:
            func: Function to register
            name: Override function name
            description: Override auto-generated description

        Returns:
            The original function (unchanged)
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            self._register_func(name or f.__name__, f, description)
            return f

        if func is None:
            return decorator
        return decorator(func)

    def _register_func(
        self,
        name: str,
        func: Callable[..., Any],
        description: str | None,
    ) -> None:
        """Internal registration logic."""
        schema = generate_tool_schema(func)

        self._tools[name] = ToolDefinition(
            name=name,
            func=func,
            description=description or schema.get("description", ""),
            parameters=schema["parameters"],
        )

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools)

    @property
    def names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    def to_api_format(self) -> list[dict[str, Any]]:
        """Convert all tools to API request format."""
        return [tool.to_api_format() for tool in self._tools.values()]

    @classmethod
    def from_dict(
        cls,
        tools: dict[str, Callable[..., Any]],
    ) -> ToolRegistry:
        """
        Create registry from dict of name -> function.

        Example:
            tools = ToolRegistry.from_dict({
                "search": search_func,
                "calculate": lambda x: x * 2,
            })
        """
        registry = cls()
        for name, func in tools.items():
            registry.register(name, func)
        return registry

    def merge(self, other: ToolRegistry) -> ToolRegistry:
        """
        Merge another registry into this one.

        Returns self for chaining.
        """
        self._tools.update(other._tools)
        return self
