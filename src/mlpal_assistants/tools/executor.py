"""Tool execution with validation and error handling."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from mlpal_assistants.tools.result import ToolResult

if TYPE_CHECKING:
    from mlpal_assistants.models.tools import ToolCall
    from mlpal_assistants.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Executes tool calls against a registry.

    Handles:
    - Argument parsing and validation
    - Sync and async function execution
    - Error handling and result formatting
    - Execution timing

    Example:
        executor = ToolExecutor()
        results = await executor.execute_all(tool_calls, registry)
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        validate_args: bool = True,
    ) -> None:
        """
        Initialize executor.

        Args:
            timeout: Default timeout for tool execution (seconds)
            validate_args: Whether to validate arguments against schema
        """
        self._timeout = timeout
        self._validate_args = validate_args

    async def execute_all(
        self,
        tool_calls: list[ToolCall] | tuple[ToolCall, ...],
        registry: ToolRegistry,
        *,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: Tool calls from model response
            registry: Registry containing tool definitions
            parallel: Execute tools in parallel (default True)

        Returns:
            List of ToolResult in same order as input
        """
        if parallel:
            tasks = [self.execute_one(call, registry) for call in tool_calls]
            return await asyncio.gather(*tasks)

        return [await self.execute_one(call, registry) for call in tool_calls]

    async def execute_one(
        self,
        tool_call: ToolCall,
        registry: ToolRegistry,
    ) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call from model response
            registry: Registry containing tool definition

        Returns:
            ToolResult with success or error info
        """
        start_time = time.perf_counter()
        tool_name = tool_call.function.name

        # Look up tool
        tool = registry.get(tool_name)
        if tool is None:
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                name=tool_name,
                error=f"Unknown tool: {tool_name}",
            )

        # Parse arguments
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                name=tool_name,
                error=f"Invalid JSON arguments: {e}",
            )

        # Validate arguments against schema
        if self._validate_args:
            validation_error = self._validate_arguments(args, tool.parameters)
            if validation_error:
                return ToolResult.error_result(
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    error=validation_error,
                )

        # Execute tool
        try:
            result = await self._execute_with_timeout(tool.func, args)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Ensure string result
            if not isinstance(result, str):
                result = json.dumps(result, default=str)

            return ToolResult.success_result(
                tool_call_id=tool_call.id,
                name=tool_name,
                result=result,
                latency_ms=latency_ms,
            )

        except TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                name=tool_name,
                error=f"Tool execution timed out after {self._timeout}s",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                name=tool_name,
                error=f"Execution error: {type(e).__name__}: {e}",
                latency_ms=latency_ms,
            )

    async def _execute_with_timeout(
        self,
        func: Any,
        args: dict[str, Any],
    ) -> Any:
        """Execute function with timeout, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(**args), timeout=self._timeout)

        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(**args)),
            timeout=self._timeout,
        )

    def _validate_arguments(
        self,
        args: dict[str, Any],
        schema: dict[str, Any],
    ) -> str | None:
        """
        Validate arguments against JSON Schema.

        Returns error message if invalid, None if valid.
        """
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required arguments
        for req_arg in required:
            if req_arg not in args:
                return f"Missing required argument: {req_arg}"

        # Check argument types
        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                continue

            prop_schema = properties[arg_name]
            type_error = self._validate_type(arg_value, prop_schema)
            if type_error:
                return f"Invalid type for '{arg_name}': {type_error}"

        return None

    def _validate_type(
        self,
        value: Any,
        schema: dict[str, Any],
    ) -> str | None:
        """Validate a value against a schema type."""
        expected_type = schema.get("type")

        if expected_type is None:
            return None

        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }

        check = type_checks.get(expected_type)
        if check and not check(value):
            return f"expected {expected_type}, got {type(value).__name__}"

        return None
