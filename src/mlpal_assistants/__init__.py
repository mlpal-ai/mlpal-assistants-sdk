"""
MLPal Assistants SDK - Official Python SDK for MLPal Assistants API.

A production-grade SDK providing a unified interface for chat, embeddings,
image generation, text-to-speech, and transcription.

Example:
    from mlpal_assistants import Assistant, File

    async def main():
        async with Assistant() as assistant:
            # Simple string shorthand
            response = await assistant.chat("Hello!")
            print(response.content)

            # Full messages format
            response = await assistant.chat(
                messages=[{"role": "user", "content": "Hello!"}]
            )

            # Building conversation history
            messages = [{"role": "user", "content": "My name is Alice"}]
            response = await assistant.chat(messages=messages)
            messages.append(response.message)  # Use .message for history

            # With image
            response = await assistant.chat(
                messages=[{
                    "role": "user",
                    "content": "What's in this image?",
                    "files": [File.from_path("photo.jpg")]
                }]
            )

            # Structured output
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            response = await assistant.chat(
                "John is 30 years old",
                response_format=Person,
            )
            print(response.data.name)  # "John"
"""

from mlpal_assistants._version import __version__
from mlpal_assistants.client import AsyncMLPal, MLPal
from mlpal_assistants.exceptions import (
    AssistantsError,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    MaxTurnsExceededError,
    MCPConnectionError,
    MCPError,
    MCPToolNotFoundError,
    ModelNotAvailableError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    TimeoutError,
    ToolExecutionError,
    UnsupportedCapabilityError,
    ValidationError,
)
from mlpal_assistants.facade import Assistant
from mlpal_assistants.models.files import File
from mlpal_assistants.tools import ToolRegistry, ToolResult
from mlpal_assistants.transport.streaming import ChatStream, ChatStreamDelta
from mlpal_assistants.types import (
    AudioResponse,
    ChatResponse,
    CostInfo,
    Embedding,
    EmbeddingResponse,
    FunctionCall,
    GeneratedAudio,
    GeneratedImage,
    ImageResponse,
    Message,
    RoutingInfo,
    TokenUsage,
    ToolCall,
    TranscriptionResponse,
    Turn,
)

# Backward compatibility alias
Assistants = Assistant

__all__ = [
    # Version
    "__version__",
    # v2 clients (native inference + management; the go-forward surface)
    "MLPal",
    "AsyncMLPal",
    # v1 facade (agentic convenience layer; kept for compatibility)
    "Assistant",
    "Assistants",  # Backward compatibility alias
    # Files
    "File",
    # Tools
    "ToolRegistry",
    "ToolResult",
    # Response types
    "ChatResponse",
    "EmbeddingResponse",
    "Embedding",
    "ImageResponse",
    "GeneratedImage",
    "AudioResponse",
    "GeneratedAudio",
    "TranscriptionResponse",
    "ToolCall",
    "FunctionCall",
    "Turn",
    # Streaming
    "ChatStream",
    "ChatStreamDelta",
    # Common types
    "TokenUsage",
    "CostInfo",
    "RoutingInfo",
    "Message",
    # Exceptions
    "AssistantsError",
    "AuthenticationError",
    "PermissionDeniedError",
    "RateLimitError",
    "QuotaExceededError",
    "BadRequestError",
    "ValidationError",
    "NotFoundError",
    "ModelNotAvailableError",
    "UnsupportedCapabilityError",
    "ServerError",
    "ServiceUnavailableError",
    "ProviderError",
    "TimeoutError",
    "ConnectionError",
    "ToolExecutionError",
    "MaxTurnsExceededError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolNotFoundError",
]
