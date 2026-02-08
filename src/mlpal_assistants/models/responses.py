"""Response models as frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlpal_assistants.models.common import CostInfo, RoutingInfo
from mlpal_assistants.models.tools import ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class Turn:
    """A single turn in an agentic conversation."""

    messages: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...] | None = None


@dataclass(slots=True)
class ChatResponse:
    """
    Response from chat completion.

    Attributes:
        content: The text response from the model
        cost: Token usage and cost information
        routing: Model routing information
        finish_reason: Why the model stopped (stop, tool_calls, length)
        data: Parsed structured output if response_format was provided
        tool_calls: Tool calls if finish_reason is "tool_calls"
        turns: Turn history if tools were executed
    """

    content: str | None
    cost: CostInfo
    routing: RoutingInfo
    finish_reason: str = "stop"
    data: Any = None  # Pydantic model instance if structured output
    tool_calls: tuple[ToolCall, ...] | None = None
    turns: list[Turn] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        *,
        parsed_data: Any = None,
        turns: list[Turn] | None = None,
    ) -> ChatResponse:
        """Create from API response."""
        tool_calls = None
        if raw_calls := data.get("tool_calls"):
            tool_calls = tuple(ToolCall.from_api(tc) for tc in raw_calls)

        cost_data = data.get("cost") or {}
        routing_data = data.get("routing") or {}

        # Construct routing from cost if not present
        if not routing_data:
            routing_data = {
                "requested_model": "",
                "resolved_model": cost_data.get("model_name", ""),
                "resolved_provider": cost_data.get("provider", ""),
            }

        return cls(
            content=data.get("content"),
            cost=CostInfo.from_api(cost_data),
            routing=RoutingInfo.from_api(routing_data),
            finish_reason=data.get("finish_reason", "stop"),
            data=parsed_data,
            tool_calls=tool_calls,
            turns=turns,
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class Embedding:
    """A single embedding result."""

    index: int
    embedding: tuple[float, ...]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Embedding:
        """Create from API response."""
        return cls(
            index=data.get("index", 0),
            embedding=tuple(data.get("embedding", [])),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Response from embedding creation."""

    data: tuple[Embedding, ...]
    model: str
    cost: CostInfo

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> EmbeddingResponse:
        """Create from API response."""
        return cls(
            data=tuple(Embedding.from_api(e) for e in data.get("data", [])),
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


@dataclass(frozen=True, slots=True)
class ImageResponse:
    """Response from image generation."""

    images: tuple[GeneratedImage, ...]
    cost: CostInfo

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ImageResponse:
        """Create from API response."""
        return cls(
            images=tuple(GeneratedImage.from_api(img) for img in data.get("data", [])),
            cost=CostInfo.from_api(data.get("cost", data.get("usage", {}))),
        )


@dataclass(frozen=True, slots=True)
class GeneratedAudio:
    """Generated audio data."""

    content: bytes
    format: str
    url: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any], content: bytes) -> GeneratedAudio:
        """Create from API response."""
        return cls(
            content=content,
            format=data.get("format", "mp3"),
            url=data.get("url"),
        )


@dataclass(slots=True)
class AudioResponse:
    """Response from text-to-speech."""

    audio: GeneratedAudio
    cost: CostInfo

    @property
    def content(self) -> bytes:
        """Get audio content bytes."""
        return self.audio.content

    @property
    def format(self) -> str:
        """Get audio format."""
        return self.audio.format

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


@dataclass(frozen=True, slots=True)
class TranscriptionResponse:
    """Response from audio transcription."""

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
