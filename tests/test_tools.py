"""Tests for tool execution module."""

import asyncio

import pytest

from mlpal_assistants.tools import ToolRegistry, ToolResult
from mlpal_assistants.tools.executor import ToolExecutor
from mlpal_assistants.tools.schema import generate_tool_schema
from mlpal_assistants.types import FunctionCall, ToolCall


class TestToolSchema:
    """Tests for JSON schema generation."""

    def test_basic_function(self):
        """Test schema generation for basic function."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = generate_tool_schema(add)

        assert schema["name"] == "add"
        assert "Add two numbers" in schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["properties"]["a"]["type"] == "integer"
        assert schema["parameters"]["properties"]["b"]["type"] == "integer"
        assert set(schema["parameters"]["required"]) == {"a", "b"}

    def test_optional_parameter(self):
        """Test schema generation with optional parameter."""
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        schema = generate_tool_schema(greet)

        assert "name" in schema["parameters"]["required"]
        assert "greeting" not in schema["parameters"]["required"]
        assert schema["parameters"]["properties"]["greeting"]["default"] == "Hello"

    def test_list_parameter(self):
        """Test schema generation with list parameter."""
        def sum_numbers(numbers: list[int]) -> int:
            """Sum a list of numbers."""
            return sum(numbers)

        schema = generate_tool_schema(sum_numbers)

        assert schema["parameters"]["properties"]["numbers"]["type"] == "array"
        assert schema["parameters"]["properties"]["numbers"]["items"]["type"] == "integer"

    def test_docstring_param_descriptions(self):
        """Test parameter descriptions from docstring."""
        def search(query: str, limit: int = 10) -> list[str]:
            """Search for items.

            Args:
                query: The search query string
                limit: Maximum number of results
            """
            return []

        schema = generate_tool_schema(search)

        assert "search query" in schema["parameters"]["properties"]["query"]["description"]
        assert "Maximum" in schema["parameters"]["properties"]["limit"]["description"]


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_function(self):
        """Test registering a function."""
        registry = ToolRegistry()

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        registry.register("get_weather", get_weather)

        assert "get_weather" in registry
        assert registry.get("get_weather") is not None

    def test_register_with_decorator(self):
        """Test registering with decorator."""
        registry = ToolRegistry()

        @registry.tool
        def calculate(x: int, y: int) -> int:
            """Calculate something."""
            return x + y

        assert "calculate" in registry

    def test_register_with_custom_name(self):
        """Test registering with custom name."""
        registry = ToolRegistry()

        @registry.tool(name="custom_name")
        def my_func(x: int) -> int:
            """Do something."""
            return x

        assert "custom_name" in registry
        assert "my_func" not in registry

    def test_from_dict(self):
        """Test creating registry from dict."""
        def func1(x: int) -> int:
            """First function."""
            return x

        def func2(y: str) -> str:
            """Second function."""
            return y

        registry = ToolRegistry.from_dict({
            "func1": func1,
            "func2": func2,
        })

        assert "func1" in registry
        assert "func2" in registry

    def test_to_api_format(self):
        """Test converting to API format."""
        registry = ToolRegistry()

        def search(query: str) -> str:
            """Search for items."""
            return "results"

        registry.register("search", search)
        api_format = registry.to_api_format()

        assert len(api_format) == 1
        assert api_format[0]["type"] == "function"
        assert api_format[0]["function"]["name"] == "search"
        assert api_format[0]["function"]["parameters"]["type"] == "object"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        registry = ToolRegistry()

        def sync_func(x: int) -> int:
            """Sync function."""
            return x * 2

        async def async_func(y: int) -> int:
            """Async function."""
            await asyncio.sleep(0.01)
            return y * 3

        registry.register("sync_func", sync_func)
        registry.register("async_func", async_func)
        return registry

    @pytest.fixture
    def executor(self):
        """Create a ToolExecutor."""
        return ToolExecutor()

    async def test_execute_sync_function(self, executor, registry):
        """Test executing a sync function."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="sync_func",
                arguments='{"x": 5}'
            )
        )

        result = await executor.execute_one(tool_call, registry)

        assert result.success
        assert result.result == "10"
        assert result.tool_call_id == "call_1"

    async def test_execute_async_function(self, executor, registry):
        """Test executing an async function."""
        tool_call = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(
                name="async_func",
                arguments='{"y": 5}'
            )
        )

        result = await executor.execute_one(tool_call, registry)

        assert result.success
        assert result.result == "15"

    async def test_execute_unknown_tool(self, executor, registry):
        """Test executing unknown tool returns error."""
        tool_call = ToolCall(
            id="call_3",
            type="function",
            function=FunctionCall(
                name="unknown_func",
                arguments='{}'
            )
        )

        result = await executor.execute_one(tool_call, registry)

        assert not result.success
        assert "Unknown tool" in result.error

    async def test_execute_with_invalid_args(self, executor, registry):
        """Test executing with invalid JSON args."""
        tool_call = ToolCall(
            id="call_4",
            type="function",
            function=FunctionCall(
                name="sync_func",
                arguments='invalid json'
            )
        )

        result = await executor.execute_one(tool_call, registry)

        assert not result.success
        assert result.error is not None

    async def test_execute_all_parallel(self, executor, registry):
        """Test parallel execution of multiple tools."""
        calls = [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=FunctionCall(
                    name="sync_func",
                    arguments=f'{{"x": {i}}}'
                )
            )
            for i in range(3)
        ]

        results = await executor.execute_all(calls, registry, parallel=True)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert [r.result for r in results] == ["0", "2", "4"]


class TestToolResult:
    """Tests for ToolResult."""

    def test_successful_result(self):
        """Test creating successful result."""
        result = ToolResult(
            tool_call_id="call_1",
            name="func",
            result="success",
            success=True,
        )

        assert result.success
        assert result.result == "success"
        assert result.error is None

    def test_error_result(self):
        """Test creating error result."""
        result = ToolResult(
            tool_call_id="call_1",
            name="func",
            result="",
            success=False,
            error="Something went wrong",
        )

        assert not result.success
        assert result.error == "Something went wrong"

    def test_to_message(self):
        """Test converting to message format."""
        result = ToolResult(
            tool_call_id="call_1",
            name="get_weather",
            result="Sunny",
            success=True,
        )

        message = result.to_message()

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_1"
        assert message["content"] == "Sunny"
