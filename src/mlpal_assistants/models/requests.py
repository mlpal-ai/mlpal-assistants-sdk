"""Request models with Pydantic validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Validated chat completion request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, ge=0, le=1)
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


class EmbedRequest(BaseModel):
    """Validated embedding request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: list[str]


class ImageRequest(BaseModel):
    """Validated image generation request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = Field(1, ge=1, le=10)
    style: str | None = None
    reference_images: list[dict[str, Any]] | None = None


class SpeechRequest(BaseModel):
    """Validated text-to-speech request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
