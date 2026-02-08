"""Thin Assistants facade - composes services into a unified API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable

from pydantic import BaseModel

from mlpal_assistants.config import Config
from mlpal_assistants.models.responses import (
    AudioResponse,
    ChatResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
)
from mlpal_assistants.services import AudioService, ChatService, EmbeddingService, ImageService
from mlpal_assistants.transport import HTTPTransport

if TYPE_CHECKING:
    from mlpal_assistants.models.messages import Message
    from mlpal_assistants.tools.registry import ToolRegistry


class Assistants:
    """
    Unified facade for the MLPal Assistants API.

    Thin wrapper that composes services for chat, embeddings, images, and audio.

    Example:
        async with Assistants() as assistant:
            response = await assistant.chat(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Initialize the Assistants client.

        Args:
            api_key: MLPal API key. Falls back to MLPAL_API_KEY env var.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        self._config = Config.from_env(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._transport = HTTPTransport(self._config)

        # Compose services
        self._chat = ChatService(self._transport)
        self._embeddings = EmbeddingService(self._transport)
        self._images = ImageService(self._transport)
        self._audio = AudioService(self._transport)

    async def chat(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        model: str = "mlpal",
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
            messages: Conversation messages
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet", "mlpal")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            response_format: Pydantic model for structured output
            tools: Tool definitions - can be:
                - list[dict]: Raw OpenAI-format tool definitions (no auto-execution)
                - ToolRegistry: Registry with auto-execution
                - dict[str, Callable]: Dict of name->function (auto-converted to registry)
            tool_choice: Tool selection mode
            parallel_tool_calls: Allow parallel tool calls
            max_turns: Max turns for agentic tool execution (only with ToolRegistry)

        Returns:
            ChatResponse with content, cost, routing, and optional structured data

        Example:
            # Simple chat
            response = await assistant.chat(
                messages=[{"role": "user", "content": "Hello!"}]
            )

            # With automatic tool execution
            def get_weather(city: str) -> str:
                '''Get weather for a city.'''
                return f"Sunny in {city}"

            response = await assistant.chat(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools={"get_weather": get_weather},
            )
        """
        return await self._chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            max_turns=max_turns,
        )

    async def embed(
        self,
        input: str | list[str],
        *,
        model: str = "mlpal",
    ) -> EmbeddingResponse:
        """
        Create embeddings for text.

        Args:
            model: Embedding model identifier
            input: Text or list of texts to embed

        Returns:
            EmbeddingResponse with embeddings and cost info
        """
        return await self._embeddings.create(model=model, input=input)

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = "mlpal",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        style: str | None = None,
        reference_images: list[dict[str, Any]] | None = None,
    ) -> ImageResponse:
        """
        Generate images from a prompt.

        Args:
            model: Image model identifier
            prompt: Text description of the image
            size: Image size (e.g., "1024x1024")
            quality: Generation quality
            n: Number of images to generate
            style: Style hint
            reference_images: Reference images for style transfer

        Returns:
            ImageResponse with generated images and cost info
        """
        return await self._images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            style=style,
            reference_images=reference_images,
        )

    async def generate_speech(
        self,
        input: str,
        *,
        model: str = "mlpal",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> AudioResponse:
        """
        Generate speech from text.

        Args:
            model: TTS model identifier
            input: Text to convert to speech
            voice: Voice to use
            response_format: Audio format
            speed: Speech speed (0.25 - 4.0)

        Returns:
            AudioResponse with audio bytes and cost info
        """
        return await self._audio.speech(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

    async def transcribe(
        self,
        file: str | Path | bytes | BinaryIO,
        *,
        model: str = "mlpal",
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            model: Transcription model identifier
            file: Audio file path, bytes, or file-like object
            language: Language hint
            prompt: Optional context prompt

        Returns:
            TranscriptionResponse with text and cost info
        """
        return await self._audio.transcribe(
            model=model,
            file=file,
            language=language,
            prompt=prompt,
        )

    async def close(self) -> None:
        """Close the client and release resources."""
        await self._transport.close()

    async def __aenter__(self) -> Assistants:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()
