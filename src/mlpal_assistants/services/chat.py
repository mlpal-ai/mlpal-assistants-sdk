"""Chat service - handles chat completions and structured output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from mlpal_assistants.models.messages import normalize_messages
from mlpal_assistants.models.requests import ChatRequest
from mlpal_assistants.models.responses import ChatResponse
from mlpal_assistants.services._base import BaseService

if TYPE_CHECKING:
    from mlpal_assistants.models.messages import Message


class ChatService(BaseService):
    """
    Chat completion service.

    Single responsibility: Transform user parameters into API requests
    and API responses into domain objects.
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
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
            tools: Tool definitions for function calling
            tool_choice: How to select tools ("auto", "none", or specific)
            parallel_tool_calls: Allow parallel tool calls

        Returns:
            ChatResponse with content, cost, routing, and optional structured data
        """
        # Build request body
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

        # Parse structured output if requested
        parsed_data = None
        if pydantic_model is not None and data.get("content"):
            try:
                parsed_data = pydantic_model.model_validate_json(data["content"])
            except Exception:
                pass

        return ChatResponse.from_api(data, parsed_data=parsed_data)

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
        Make schema strict by adding additionalProperties: false recursively.

        Required for OpenAI structured output compliance.
        """
        result = schema.copy()

        if result.get("type") == "object":
            result["additionalProperties"] = False

            # Process properties
            if "properties" in result:
                result["properties"] = {
                    k: self._make_strict_schema(v)
                    for k, v in result["properties"].items()
                }

        # Process nested definitions ($defs)
        if "$defs" in result:
            result["$defs"] = {
                k: self._make_strict_schema(v) for k, v in result["$defs"].items()
            }

        # Process array items
        if "items" in result:
            result["items"] = self._make_strict_schema(result["items"])

        # Process anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in result:
                result[key] = [self._make_strict_schema(s) for s in result[key]]

        return result
