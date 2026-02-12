"""Response types for the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mlpal_assistants.types.common import CostInfo, RoutingInfo

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(slots=True)
class FunctionCall:
    """Function call details."""

    name: str
    arguments: str

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> FunctionCall:
        """Create from API response."""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", "{}"),
        )


@dataclass(slots=True)
class ToolCall:
    """A tool call made by the model."""

    id: str
    type: Literal["function"]
    function: FunctionCall

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ToolCall:
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=FunctionCall.from_api(data.get("function", {})),
        )


@dataclass(slots=True)
class Turn:
    """A single turn in an agentic conversation."""

    messages: list[dict[str, Any]]
    response: ChatResponse
    tool_results: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class ChatResponse:
    """
    Response from chat().

    Mirrors the service response structure.
    """

    content: str | None
    cost: CostInfo
    routing: RoutingInfo
    role: Literal["assistant"] = "assistant"
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] = "stop"
    data: BaseModel | None = None
    tool_calls: list[ToolCall] | None = None
    turns: list[Turn] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        *,
        parsed_data: BaseModel | None = None,
        turns: list[Turn] | None = None,
    ) -> ChatResponse:
        """Create from API response.

        MLPal API returns a flat structure:
        {
            "content": "...",
            "role": "assistant",
            "finish_reason": "stop",
            "tool_calls": [...] | null,
            "data": {...} | null,
            "cost": {...},
            "routing": {...},
            "metadata": {...}
        }
        """
        tool_calls = None
        if raw_tool_calls := data.get("tool_calls"):
            tool_calls = [ToolCall.from_api(tc) for tc in raw_tool_calls]

        cost_data = data.get("cost") or {}
        routing_data = data.get("routing") or {}

        # If routing is empty, construct from cost data
        if not routing_data:
            routing_data = {
                "requested_model": "",
                "resolved_model": cost_data.get("model_name", ""),
                "resolved_provider": cost_data.get("provider", ""),
            }

        return cls(
            content=data.get("content"),
            role=data.get("role", "assistant"),
            finish_reason=data.get("finish_reason", "stop"),
            data=parsed_data or data.get("data"),
            tool_calls=tool_calls,
            turns=turns,
            cost=CostInfo.from_api(cost_data),
            routing=RoutingInfo.from_api(routing_data),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class Embedding:
    """A single embedding result."""

    index: int
    embedding: list[float]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Embedding:
        """Create from API response."""
        return cls(
            index=data.get("index", 0),
            embedding=data.get("embedding", []),
        )


@dataclass(slots=True)
class EmbeddingResponse:
    """Response from embed()."""

    data: list[Embedding]
    model: str
    cost: CostInfo

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> EmbeddingResponse:
        """Create from API response."""
        return cls(
            data=[Embedding.from_api(e) for e in data.get("data", [])],
            model=data.get("model", ""),
            cost=CostInfo.from_api(data.get("cost", data.get("usage", {}))),
        )


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """A generated image."""

    url: str
    revised_prompt: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> GeneratedImage:
        """Create from API response."""
        return cls(
            url=data.get("url", ""),
            revised_prompt=data.get("revised_prompt"),
        )


@dataclass(slots=True)
class ImageResponse:
    """Response from generate.image()."""

    images: list[GeneratedImage]
    cost: CostInfo

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ImageResponse:
        """Create from API response."""
        return cls(
            images=[GeneratedImage.from_api(img) for img in data.get("data", [])],
            cost=CostInfo.from_api(data.get("cost", data.get("usage", {}))),
        )


@dataclass(slots=True)
class GeneratedAudio:
    """Generated audio data."""

    url: str
    content: bytes
    format: str
    size_bytes: int

    @classmethod
    def from_api(cls, data: dict[str, Any], content: bytes) -> GeneratedAudio:
        """Create from API response."""
        return cls(
            url=data.get("url", ""),
            content=content,
            format=data.get("format", "mp3"),
            size_bytes=len(content),
        )


@dataclass(slots=True)
class AudioResponse:
    """Response from generate.speech()."""

    audio: GeneratedAudio
    cost: CostInfo

    def save(self, path: str) -> None:
        """Save audio to file."""
        with open(path, "wb") as f:
            f.write(self.audio.content)

    @classmethod
    def from_api(cls, data: dict[str, Any], content: bytes) -> AudioResponse:
        """Create from API response."""
        return cls(
            audio=GeneratedAudio.from_api(data, content),
            cost=CostInfo.from_api(data.get("cost", data.get("usage", {}))),
        )


@dataclass(slots=True)
class TranscriptionResponse:
    """Response from transcribe()."""

    text: str
    language: str | None
    cost: CostInfo

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> TranscriptionResponse:
        """Create from API response."""
        return cls(
            text=data.get("text", ""),
            language=data.get("language"),
            cost=CostInfo.from_api(data.get("cost", data.get("usage", {}))),
        )
