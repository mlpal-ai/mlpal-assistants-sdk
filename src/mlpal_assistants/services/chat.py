"""Chat service - handles chat completions and structured output."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from mlpal_assistants.models.messages import normalize_messages
from mlpal_assistants.models.requests import ChatRequest
from mlpal_assistants.models.responses import ChatResponse
from mlpal_assistants.services._base import BaseService
from mlpal_assistants.transport.streaming import ChatStream

if TYPE_CHECKING:
    from mlpal_assistants.models.messages import Message
    from mlpal_assistants.tools.registry import ToolRegistry


class ChatService(BaseService):
    """
    Chat completion service.

    Handles:
    - Simple chat completions
    - Structured output (Pydantic models)
    - Tool definitions (for model to request)
    - Automatic tool execution (with ToolRegistry)
    """

    async def complete(
        self,
        model: str,
        messages: list[Message | dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | ToolRegistry | dict[str, Callable[..., Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        max_turns: int = 10,
    ) -> ChatResponse:
        """
        Create a chat completion.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet")
            messages: Conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            response_format: Pydantic model for structured output, or schema dict
            tools: Tool definitions - can be:
                - list[dict]: Raw OpenAI-format tool definitions (no auto-execution)
                - ToolRegistry: Registry with auto-execution
                - dict[str, Callable]: Dict of name->function (auto-converted to registry)
            tool_choice: How to select tools ("auto", "none", or specific)
            parallel_tool_calls: Allow parallel tool calls
            max_turns: Max turns for agentic tool execution (only with ToolRegistry)

        Returns:
            ChatResponse with content, cost, routing, and optional structured data
        """
        # Convert tools to appropriate format
        tool_registry: ToolRegistry | None = None
        tool_definitions: list[dict[str, Any]] | None = None

        if tools is not None:
            tool_registry, tool_definitions = self._prepare_tools(tools)

        # If we have a registry, use the agent loop for automatic execution
        if tool_registry is not None:
            # Extract Pydantic model for tool loop (doesn't support raw dict schemas)
            pydantic_response_format: type[BaseModel] | None = None
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_response_format = response_format

            return await self._complete_with_tools(
                model=model,
                messages=messages,
                tool_registry=tool_registry,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                response_format=pydantic_response_format,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )

        # Single completion (no auto-execution)
        return await self._complete_single(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            tools=tool_definitions,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    async def _complete_single(
        self,
        model: str,
        messages: list[Message | dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatResponse:
        """Single chat completion without automatic tool execution."""
        body: dict[str, Any] = {
            "model": model,
            "messages": normalize_messages(messages),
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls

        # Handle structured output
        pydantic_model: type[BaseModel] | None = None
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_model = response_format
                body["response_format"] = self._build_json_schema(response_format)
            else:
                body["response_format"] = response_format

        # Validate request
        ChatRequest.model_validate(body)

        # Make request
        data = await self._transport.post("/v1/chat/completions", body=body)

        response = ChatResponse.from_api(data)
        # Parse structured output if requested
        parsed_data = None
        if pydantic_model is not None and response.content:
            try:
                parsed_data = pydantic_model.model_validate_json(response.content)
            except Exception:
                pass
        if parsed_data is not None:
            response = ChatResponse.from_api(data, parsed_data=parsed_data)
        return response

    async def _complete_with_tools(
        self,
        model: str,
        messages: list[Message | dict[str, Any]],
        tool_registry: ToolRegistry,
        max_turns: int,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatResponse:
        """Chat completion with automatic tool execution."""
        from mlpal_assistants.tools.loop import AgentLoop

        loop = AgentLoop(
            transport=self._transport,
            tools=tool_registry,
            max_turns=max_turns,
            parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else True,
        )

        return await loop.run(
            messages=normalize_messages(messages),
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tool_choice=tool_choice,
        )

    def _prepare_tools(
        self,
        tools: list[dict[str, Any]] | ToolRegistry | dict[str, Callable[..., Any]],
    ) -> tuple[ToolRegistry | None, list[dict[str, Any]] | None]:
        """
        Prepare tools for use.

        Returns:
            (registry, definitions) - registry is set for auto-execution,
            definitions for pass-through to API
        """
        from mlpal_assistants.tools.registry import ToolRegistry

        # Already a registry - use for auto-execution
        if isinstance(tools, ToolRegistry):
            return tools, None

        # Dict of callables - convert to registry
        if isinstance(tools, dict):
            # Check if it's a dict of callables
            if tools and callable(next(iter(tools.values()))):
                registry = ToolRegistry.from_dict(tools)
                return registry, None

        # List of dicts - pass through as definitions (no auto-execution)
        if isinstance(tools, list):
            return None, tools

        return None, None

    def _build_json_schema(self, model: type[BaseModel]) -> dict[str, Any]:
        """Build JSON schema response format from Pydantic model."""
        schema = model.model_json_schema()
        strict_schema = self._make_strict_schema(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": strict_schema,
                "strict": True,
            },
        }

    def _make_strict_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Make schema OpenAI-strict compliant.

        OpenAI requires:
        - additionalProperties: false on all objects
        - All properties must be in the required array
        - $ref must not have sibling properties (like description)
        """
        result = schema.copy()

        # Handle $ref - OpenAI requires $ref to be standalone without siblings
        if "$ref" in result:
            # Keep only $ref, strip all sibling properties (description, title, etc.)
            return {"$ref": result["$ref"]}

        if result.get("type") == "object":
            result["additionalProperties"] = False

            if "properties" in result:
                result["required"] = list(result["properties"].keys())
                result["properties"] = {
                    k: self._make_strict_schema(v)
                    for k, v in result["properties"].items()
                }

        if "$defs" in result:
            result["$defs"] = {
                k: self._make_strict_schema(v) for k, v in result["$defs"].items()
            }

        if "items" in result:
            result["items"] = self._make_strict_schema(result["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in result:
                result[key] = [self._make_strict_schema(s) for s in result[key]]

        return result

    @asynccontextmanager
    async def stream(
        self,
        model: str,
        messages: list[Message | dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AsyncIterator[ChatStream]:
        """
        Stream a chat completion.

        Yields deltas as they arrive from the server.

        Note: Streaming does not support automatic tool execution.
        Use `complete()` with ToolRegistry for agentic workflows.

        Args:
            model: Model identifier
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            tools: Raw tool definitions (no auto-execution in streaming)
            tool_choice: Tool choice mode
            parallel_tool_calls: Allow parallel tool calls

        Yields:
            ChatStream that yields ChatStreamDelta objects

        Example:
            async with service.stream("gpt-4o", messages) as stream:
                async for delta in stream:
                    if delta.content:
                        print(delta.content, end="", flush=True)
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": normalize_messages(messages),
            "stream": True,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls

        # Validate request (stream=True is extra, so skip validation for it)
        validate_body = {k: v for k, v in body.items() if k != "stream"}
        ChatRequest.model_validate(validate_body)

        async with self._transport.stream_post("/v1/chat/completions", body=body) as stream:
            yield stream
