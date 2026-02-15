"""Embedding service - handles text embedding generation."""

from __future__ import annotations

from mlpal_assistants.models.requests import EmbedRequest
from mlpal_assistants.models.responses import EmbeddingResponse
from mlpal_assistants.services._base import BaseService


class EmbeddingService(BaseService):
    """
    Embedding service.

    Single responsibility: Generate embeddings for text inputs.
    """

    async def create(
        self,
        model: str,
        input: str | list[str],
    ) -> EmbeddingResponse:
        """
        Create embeddings for text.

        Args:
            model: Embedding model identifier
            input: Text or list of texts to embed

        Returns:
            EmbeddingResponse with embeddings and cost info
        """
        texts = [input] if isinstance(input, str) else input

        # Validate request
        EmbedRequest.model_validate({"model": model, "input": texts})

        body = {"model": model, "input": texts}
        data = await self._transport.post("/v1/embeddings", body=body)

        return EmbeddingResponse.from_api(data)
