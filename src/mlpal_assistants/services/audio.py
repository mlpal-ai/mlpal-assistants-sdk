"""Audio service - handles text-to-speech and transcription."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from mlpal_assistants.models.requests import SpeechRequest
from mlpal_assistants.models.responses import AudioResponse, TranscriptionResponse
from mlpal_assistants.services._base import BaseService


class AudioService(BaseService):
    """
    Audio service for TTS and transcription.

    Single responsibility: Convert between text and audio.
    """

    async def speech(
        self,
        model: str,
        input: str,
        *,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> AudioResponse:
        """
        Generate speech from text.

        Args:
            model: TTS model identifier
            input: Text to convert to speech
            voice: Voice to use (e.g., "alloy", "echo", "nova")
            response_format: Audio format ("mp3", "opus", "aac", "flac")
            speed: Speech speed (0.25 - 4.0)

        Returns:
            AudioResponse with audio bytes and cost info
        """
        body = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        # Validate request
        SpeechRequest.model_validate(body)

        data = await self._transport.post("/v1/audio/speech", body=body)

        # Handle URL-based response (S3) or base64 response
        audio_content = await self._get_audio_content(data)

        return AudioResponse.from_api(data, audio_content)

    async def _get_audio_content(self, data: dict[str, Any]) -> bytes:
        """
        Extract audio content from API response.

        Handles both URL-based (S3) and base64-encoded responses.
        """
        # Priority 1: URL-based response (S3)
        if url := data.get("url"):
            return await self._fetch_audio_from_url(url)

        # Priority 2: Base64-encoded response
        if audio_b64 := data.get("audio"):
            import base64

            return base64.b64decode(audio_b64)

        # No audio content found
        return b""

    async def _fetch_audio_from_url(self, url: str) -> bytes:
        """Fetch audio content from URL (typically S3)."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def transcribe(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            model: Transcription model identifier
            file: Audio file path, bytes, or file-like object
            language: Language hint (ISO code)
            prompt: Optional context prompt

        Returns:
            TranscriptionResponse with text and cost info
        """
        # Prepare file for upload
        if isinstance(file, (str, Path)):
            path = Path(file)
            file_data = (path.name, path.read_bytes(), self._guess_mime(path))
        elif isinstance(file, bytes):
            file_data = ("audio.mp3", file, "audio/mpeg")
        else:
            file_data = ("audio.mp3", file.read(), "audio/mpeg")

        files = {"file": file_data}
        data = {"model": model}

        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt

        response = await self._transport.post_multipart(
            "/v1/audio/transcriptions",
            data=data,
            files=files,
        )

        return TranscriptionResponse.from_api(response)

    def _guess_mime(self, path: Path) -> str:
        """Guess MIME type from file extension."""
        suffix = path.suffix.lower()
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
        }
        return mime_map.get(suffix, "audio/mpeg")
