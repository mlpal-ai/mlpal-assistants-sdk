"""
MLPal Assistants SDK - Official Python SDK for MLPal Assistants API.

A production-grade SDK providing a unified interface for chat, embeddings,
image generation, text-to-speech, and transcription.

Example:
    from mlpal_assistants import Assistants, File

    async def main():
        assistant = Assistants()

        # Simple chat
        response = await assistant.chat(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content)

        # With image
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "What's in this image?",
                "files": [File("photo.jpg")]
            }]
        )

        # Structured output
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        response = await assistant.chat(
            messages=[{"role": "user", "content": "John is 30"}],
            response_format=Person,
        )
        print(response.data.name)  # "John"
"""

from mlpal_assistants._version import __version__
from mlpal_assistants.facade import Assistants
from mlpal_assistants.models.files import File
from mlpal_assistants.tools import ToolRegistry, ToolResult
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

__all__ = [
    # Version
    "__version__",
    # Main client
    "Assistants",
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
