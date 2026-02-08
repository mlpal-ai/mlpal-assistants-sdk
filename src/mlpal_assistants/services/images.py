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
            model: Image model identifier (e.g., "dall-e-3")
            prompt: Text description of the image
            size: Image size (e.g., "1024x1024")
            quality: Generation quality ("standard" or "hd")
            n: Number of images to generate (1-10)
            style: Style hint (model-dependent)
            reference_images: Reference images for style transfer

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
