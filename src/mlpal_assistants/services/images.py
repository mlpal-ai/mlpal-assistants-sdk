"""Image service - handles image generation."""

from __future__ import annotations

from typing import Any

from mlpal_assistants.models.requests import ImageRequest
from mlpal_assistants.models.responses import ImageResponse
from mlpal_assistants.services._base import BaseService


class ImageService(BaseService):
    """
    Image generation service.

    Single responsibility: Generate images from text prompts.

    Supports unified size/aspect ratio across all providers:
    - Semantic presets: "square", "landscape", "portrait", "wide", "tall"
    - Aspect ratios: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
    - Pixel dimensions: "1024x1024", "1792x1024", "1536x1024"
    """

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        style: str | None = None,
        reference_images: list[dict[str, Any]] | None = None,
    ) -> ImageResponse:
        """
        Generate images from a prompt.

        Args:
            model: Image model identifier (e.g., "gpt-image-1.5", "gemini-3-pro-image-preview")
            prompt: Text description of the image
            size: Image size/aspect ratio. Accepts multiple formats:
                - Semantic presets: "square", "landscape", "portrait", "wide", "tall"
                - Aspect ratios: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
                - Pixel dimensions: "1024x1024", "1792x1024", "1536x1024"
                Automatically converted to provider-specific format.
            quality: Generation quality ("standard" or "hd")
            n: Number of images to generate (1-4)
            style: Style hint (model-dependent)
            reference_images: Reference images for image-to-image generation.
                Each dict should have either:
                - {"url": "https://..."} for URL-based images
                - {"base64": "...", "mime_type": "image/png"} for base64 images

        Returns:
            ImageResponse with generated images and cost info
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
        }

        if style is not None:
            body["style"] = style
        if reference_images is not None:
            body["reference_images"] = reference_images

        # Validate request
        ImageRequest.model_validate(body)

        data = await self._transport.post("/v1/images/generations", body=body)

        return ImageResponse.from_api(data)
