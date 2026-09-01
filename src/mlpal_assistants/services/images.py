"""Image service - image generation via the gateway's async job flow."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from mlpal_assistants.exceptions import ImageJobFailedError, ImageJobTimeoutError
from mlpal_assistants.models.requests import ImageRequest
from mlpal_assistants.models.responses import ImageJob, ImageResponse
from mlpal_assistants.services._base import BaseService

# Poll cadence: first check quickly (fast renders finish in seconds), then at
# the gateway's advertised Retry-After (5s).
_FIRST_POLL_S = 2.0
_POLL_INTERVAL_S = 5.0


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
        deadline: float = 600.0,
    ) -> ImageResponse:
        """
        Generate images from a prompt.

        Internally submits a background job and polls until the result is
        ready — long renders (minutes) work without any connection timeout.
        Raises ImageJobTimeoutError (a TimeoutError subclass, carrying
        .job_id) if ``deadline`` seconds elapse first; the server-side job
        keeps running and can still be retrieved via ``get_job``.

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
        body = self._build_body(
            model=model, prompt=prompt, size=size, quality=quality, n=n,
            style=style, reference_images=reference_images,
        )
        # Background submit + poll: no single HTTP connection ever has to
        # outlive the render (the edge caps a connection at ~120s; hd/4K
        # image edits legitimately run minutes). The idempotency key makes a
        # network-blip retry return the SAME job instead of double-billing.
        body["wait"] = False
        body["idempotency_key"] = f"sdk-{uuid.uuid4().hex}"

        data = await self._transport.post("/v1/images/generations", body=body)

        # Old gateways (pre async-jobs) ignore `wait` and answer synchronously
        # with the image payload itself — pass it straight through.
        if "data" in data:
            return ImageResponse.from_api(data)

        job = ImageJob.from_api(data)
        deadline_at = time.monotonic() + deadline
        delay = _FIRST_POLL_S
        while True:
            if job.done:
                break
            if time.monotonic() >= deadline_at:
                raise ImageJobTimeoutError(
                    f"image job {job.id} still {job.status} after {deadline:.0f}s "
                    "(the job keeps running server-side; poll get_image_job to retrieve it)",
                    job_id=job.id,
                )
            await asyncio.sleep(min(delay, max(0.0, deadline_at - time.monotonic())))
            delay = _POLL_INTERVAL_S
            job = await self.get_job(job.id)

        if job.status == "failed":
            raise ImageJobFailedError(
                job.error_message or "image job failed",
                code=job.error_code or "provider_error",
                job_id=job.id,
            )
        assert job.result is not None  # succeeded jobs always carry the payload
        return job.result

    def _build_body(
        self,
        *,
        model: str,
        prompt: str,
        size: str,
        quality: str,
        n: int,
        style: str | None,
        reference_images: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
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
        # Validate the user-facing request (job-control fields are added after
        # — they belong to the transport contract, not the render).
        ImageRequest.model_validate(body)
        return body

    async def submit_job(
        self,
        model: str,
        prompt: str,
        *,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        style: str | None = None,
        reference_images: list[dict[str, Any]] | None = None,
        idempotency_key: str | None = None,
    ) -> ImageJob:
        """Submit a background image job and return immediately (no polling).

        Pair with :meth:`get_job` for custom progress UX. ``idempotency_key``
        defaults to a fresh one per call; pass your own to make retries safe
        across process restarts.
        """
        body = self._build_body(
            model=model, prompt=prompt, size=size, quality=quality, n=n,
            style=style, reference_images=reference_images,
        )
        body["wait"] = False
        body["idempotency_key"] = idempotency_key or f"sdk-{uuid.uuid4().hex}"
        data = await self._transport.post("/v1/images/generations", body=body)
        if "data" in data:  # pre async-jobs gateway answered synchronously
            return ImageJob(
                id="sync", status="succeeded", result=ImageResponse.from_api(data)
            )
        return ImageJob.from_api(data)

    async def get_job(self, job_id: str) -> ImageJob:
        """Fetch the current state of a background image job."""
        data = await self._transport.get(f"/v1/images/jobs/{job_id}")
        return ImageJob.from_api(data)
