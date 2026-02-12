"""Agent loop for multi-turn tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from mlpal_assistants.exceptions import MaxTurnsExceededError
from mlpal_assistants.models.messages import normalize_messages
from mlpal_assistants.models.responses import ChatResponse, Turn
from mlpal_assistants.models.tools import ToolCall
from mlpal_assistants.tools.executor import ToolExecutor
from mlpal_assistants.tools.result import ToolResult

if TYPE_CHECKING:
    from mlpal_assistants.tools.registry import ToolRegistry
    from mlpal_assistants.transport import HTTPTransport


@dataclass
class AgentLoopConfig:
    """Configuration for agent loop."""

    max_turns: int = 10
    parallel_tool_calls: bool = True
    tool_timeout: float = 30.0
    validate_args: bool = True


class AgentLoop:
    """
    Orchestrates multi-turn chat with automatic tool execution.

    Continues until:
    - Model returns without tool_calls (finish_reason != "tool_calls")
    - max_turns is reached

    Example:
        loop = AgentLoop(transport, tools, max_turns=5)
        response = await loop.run(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="gpt-4o",
        )
        print(response.content)
        print(f"Took {len(response.turns)} turns")
    """

    def __init__(
        self,
        transport: HTTPTransport,
        tools: ToolRegistry,
        *,
        max_turns: int = 10,
        parallel_tool_calls: bool = True,
        tool_timeout: float = 30.0,
    ) -> None:
        """
        Initialize agent loop.

        Args:
            transport: HTTP transport for API calls
            tools: Registry of available tools
            max_turns: Maximum number of turns before stopping
            parallel_tool_calls: Execute multiple tools in parallel
            tool_timeout: Timeout for individual tool execution
        """
        self._transport = transport
        self._tools = tools
        self._max_turns = max_turns
        self._parallel = parallel_tool_calls
        self._executor = ToolExecutor(timeout=tool_timeout)

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        response_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Run the agent loop until completion.

        Args:
            messages: Initial conversation messages
            model: Model to use
            response_format: Pydantic model for structured output (final response)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            tool_choice: Tool selection mode

        Returns:
            Final ChatResponse with turn history

        Raises:
            MaxTurnsExceededError: If max_turns is reached
        """
        current_messages = list(messages)
        turns: list[Turn] = []

        for turn_num in range(self._max_turns):
            # Make API call
            response = await self._call_api(
                current_messages,
                model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                tool_choice=tool_choice,
            )

            # Check if done (no more tool calls)
            if response.finish_reason != "tool_calls" or not response.tool_calls:
                return self._finalize_response(response, turns, response_format)

            # Execute tools
            results = await self._executor.execute_all(
                response.tool_calls,
                self._tools,
                parallel=self._parallel,
            )

            # Build assistant message with tool calls
            assistant_message = self._build_assistant_message(response)
            current_messages.append(assistant_message)

            # Add tool result messages
            for result in results:
                current_messages.append(result.to_message())

            # Record turn
            turns.append(Turn(
                messages=(assistant_message,),
                tool_results=tuple(
                    {
                        "tool_call_id": r.tool_call_id,
                        "name": r.name,
                        "result": r.result,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in results
                ),
            ))

        # Max turns exceeded
        raise MaxTurnsExceededError(
            f"Agent loop exceeded {self._max_turns} turns",
            turns_completed=self._max_turns,
            max_turns=self._max_turns,
        )

    async def _call_api(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        """Make a chat completion API call."""
        # Cast to satisfy type checker - dict[str, Any] is compatible with Message | dict
        normalized = normalize_messages(messages)  # type: ignore[arg-type]
        body: dict[str, Any] = {
            "model": model,
            "messages": normalized,
            "tools": self._tools.to_api_format(),
        }

        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                body[key] = value

        data = await self._transport.post("/v1/chat/completions", body=body)
        return ChatResponse.from_api(data)

    def _build_assistant_message(self, response: ChatResponse) -> dict[str, Any]:
        """Build assistant message from response."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
        }

        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]

        return message

    def _finalize_response(
        self,
        response: ChatResponse,
        turns: list[Turn],
        response_format: type[BaseModel] | None,
    ) -> ChatResponse:
        """Finalize response with structured output parsing."""
        parsed_data = None

        if response_format is not None and response.content:
            try:
                parsed_data = response_format.model_validate_json(response.content)
            except Exception:
                pass

        # Return new response with turns and parsed data
        return ChatResponse(
            content=response.content,
            cost=response.cost,
            routing=response.routing,
            finish_reason=response.finish_reason,
            data=parsed_data or response.data,
            tool_calls=response.tool_calls,
            turns=turns if turns else None,
            metadata=response.metadata,
        )
