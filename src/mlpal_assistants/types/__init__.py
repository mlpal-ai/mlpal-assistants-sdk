"""Type definitions for mlpal-assistants SDK.

Re-exports from mlpal_assistants.models for backward compatibility.
"""

from mlpal_assistants.models.common import CostInfo, RoutingInfo, TokenUsage
from mlpal_assistants.models.files import File
from mlpal_assistants.models.messages import Message
from mlpal_assistants.models.responses import (
    AudioResponse,
    ChatResponse,
    Embedding,
    EmbeddingResponse,
    GeneratedAudio,
    GeneratedImage,
    ImageResponse,
    TranscriptionResponse,
    Turn,
)
from mlpal_assistants.models.tools import FunctionCall, ToolCall, ToolResult

# Backward compatibility alias
FileAttachment = File

__all__ = [
    "TokenUsage",
    "CostInfo",
    "RoutingInfo",
    "Message",
    "File",
    "FileAttachment",
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
    "ToolResult",
    "Turn",
]
