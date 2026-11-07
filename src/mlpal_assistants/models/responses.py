"""Response models as frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlpal_assistants.models.common import CostInfo, RoutingInfo
from mlpal_assistants.models.tools import ToolCall

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class Turn:
    """A single turn in an agentic conversation."""

    messages: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...] | None = None


def _flatten_openai(data: dict[str, Any]) -> dict[str, Any]:
    """OpenAI `chat.completion` → the flat dict the dataclasses read. The
    `mlpal` extra is the gateway's own information (cost, routing, parsed
    structured output, metadata, trace id); `usage` backs `cost.tokens` when
    the extra is absent (a plain OpenAI-compatible server)."""
    choices = data.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    extra = data.get("mlpal") or {}
    usage = data.get("usage") or {}
    cost = dict(extra.get("cost") or {})
    if "tokens" not in cost and usage:
        cost["tokens"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        }
    cost.setdefault("model_name", data.get("model", ""))
    return {
        "content": message.get("content"),
        "role": message.get("role", "assistant"),
        "tool_calls": message.get("tool_calls"),
        "finish_reason": (choices[0].get("finish_reason") if choices else None) or "stop",
        "cost": cost,
        "routing": extra.get("routing") or {},
        "data": extra.get("data"),
        "metadata": {
            **(extra.get("metadata") or {}),
            **({"trace_id": extra["trace_id"]} if extra.get("trace_id") else {}),
        },
    }


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

    Example:
        # Building conversation history:
        messages = [{"role": "user", "content": "Hello"}]
        response = await assistant.chat(messages=messages)

        # Add response to history using .message property
        messages.append(response.message)

        # Continue conversation
        messages.append({"role": "user", "content": "What's my name?"})
    """

    content: str | None
    cost: CostInfo
    routing: RoutingInfo
    finish_reason: str = "stop"
    role: str = "assistant"
    data: Any = None  # Pydantic model instance if structured output
    tool_calls: tuple[ToolCall, ...] | None = None
    turns: list[Turn] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> dict[str, Any]:
        """
        Get the response as a message dict for conversation history.

        Returns a dict that can be directly appended to the messages list
        for multi-turn conversations.

        Example:
            messages = [{"role": "user", "content": "Hello"}]
            response = await assistant.chat(messages=messages)
            messages.append(response.message)  # Add assistant's response
            messages.append({"role": "user", "content": "Follow up"})
        """
        msg: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        *,
        parsed_data: Any = None,
        turns: list[Turn] | None = None,
    ) -> ChatResponse:
        """Create from an API response in either shape: OpenAI's
        `chat.completion` (what the gateway sends this SDK, with the `mlpal`
        extra carrying cost, routing, parsed data and metadata) or the
        gateway's flat shape (older gateways)."""
        if "choices" in data:
            data = _flatten_openai(data)
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
            role=data.get("role", "assistant"),
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
class ImageJob:
    """A background image generation job (submit -> poll)."""

    id: str
    status: str  # queued | running | succeeded | failed
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: ImageResponse | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def done(self) -> bool:
        return self.status in ("succeeded", "failed")

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ImageJob:
        """Create from API response (submit 202 or poll payload)."""
        err = data.get("error") or {}
        result = data.get("result")
        return cls(
            id=data["id"],
            status=data["status"],
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=ImageResponse.from_api(result) if result else None,
            error_code=err.get("code"),
            error_message=err.get("message"),
        )


@dataclass(frozen=True, slots=True)
class GeneratedAudio:
    """Generated audio data."""

    content: bytes
    format: str
    url: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any], content: bytes) -> GeneratedAudio:
        """Create from API response."""
        # Infer format from content_type if not explicitly provided
        audio_format = data.get("format") or data.get("response_format", "mp3")
        content_type = data.get("content_type")
        if not audio_format and content_type:
            # Extract format from content_type like "audio/mpeg" -> "mp3"
            format_map = {
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
                "audio/wav": "wav",
                "audio/flac": "flac",
                "audio/ogg": "ogg",
                "audio/aac": "aac",
                "audio/opus": "opus",
            }
            audio_format = format_map.get(content_type, "mp3")

        return cls(
            content=content,
            format=audio_format or "mp3",
            url=data.get("url"),
            size_bytes=data.get("size_bytes"),
            content_type=content_type,
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
