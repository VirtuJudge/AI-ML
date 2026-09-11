"""HTTP text embedding provider for VirtuJudge AI-ML.

Implements EmbeddingProvider using an OpenAI-compatible /v1/embeddings REST API endpoint.
"""

import os
from typing import Any

import httpx

from app.providers.groq_speech import ProviderError


class HttpEmbeddingProviderError(ProviderError):
    """Exception raised when HTTP embedding request fails."""


class HttpEmbeddingProvider:
    """Text embedding adapter powered by OpenAI-compatible embeddings REST API."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        dimensions: int | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get("EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        )
        self.api_key = resolved_key.strip()
        self.model = os.environ.get("EMBEDDING_MODEL", model)
        self.api_url = os.environ.get("EMBEDDING_API_URL", api_url)
        env_dims = os.environ.get("EMBEDDING_DIMENSIONS")
        self.dimensions = int(env_dims) if env_dims and env_dims.isdigit() else dimensions
        self.timeout = timeout
        self._client = http_client

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text strings."""
        if not texts:
            return []

        if not self.api_key:
            raise HttpEmbeddingProviderError(
                "Embedding API key is required. Provide `api_key` to HttpEmbeddingProvider "
                "or set the EMBEDDING_API_KEY environment variable."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        try:
            if self._client is not None:
                response = await self._client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                    )
        except httpx.RequestError as err:
            raise HttpEmbeddingProviderError(
                f"Embedding API request failed due to network error: {type(err).__name__}"
            ) from None

        if response.is_error or response.status_code >= 400:
            raise HttpEmbeddingProviderError(
                f"Embedding API request failed with status {response.status_code}: "
                f"{response.reason_phrase}"
            )

        try:
            resp_data = response.json()
        except Exception:
            raise HttpEmbeddingProviderError(
                "Embedding API returned a response that could not be parsed as JSON."
            ) from None

        data = resp_data.get("data", [])
        if not isinstance(data, list) or len(data) != len(texts):
            raise HttpEmbeddingProviderError(
                f"Embedding API response length mismatch: expected {len(texts)} "
                f"embeddings, got {len(data)}"
            )

        # Sort by index if present to maintain order
        sorted_data = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in sorted_data]


__all__ = [
    "HttpEmbeddingProvider",
    "HttpEmbeddingProviderError",
]
