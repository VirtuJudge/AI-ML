"""Resilient Groq API key pool with multi-key rotation and failover.

Provides thread-safe, coroutine-safe management of multiple Groq API keys,
preferred-key routing for parallel judges, rate-limit (HTTP 429/529) failover,
and strict log hygiene preventing secret and prompt leakage.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_TIMEOUT = 30.0

GROQ_ENV_KEYS = [
    "GROQ_API_KEY",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_4",
]


class GroqPoolError(Exception):
    """Base exception for Groq key pool errors."""


class GroqRateLimitError(GroqPoolError):
    """Raised when a Groq API request encounters rate limiting."""


class AllKeysExhaustedError(GroqPoolError):
    """Raised when all configured Groq API keys are exhausted or unavailable."""


def mask_key(key: str) -> str:
    """Safely mask an API key for safe debugging without leaking secrets.

    Example: 'gsk_1234567890abcdef' -> 'gsk_...cdef'
    """
    if not key:
        return "<empty>"
    clean = key.strip()
    if not clean:
        return "<empty>"
    if len(clean) <= 8:
        return "***"
    return f"{clean[:4]}...{clean[-4:]}"


class GroqKeyPool:
    """Resilient client pool managing multiple Groq API keys.

    Features:
    - Automatic loading of up to 4 keys from environment variables.
    - Preferred key routing per judge persona (0 contention).
    - Transparent failover to backup keys on HTTP 429 (rate limit), 503, or 529.
    - Coroutine-safe round-robin allocation when no preferred key is specified.
    - Strict log hygiene: keys and prompt texts are never logged or included in exceptions.
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        *,
        base_url: str = DEFAULT_GROQ_BASE_URL,
        timeout: float = DEFAULT_GROQ_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize GroqKeyPool.

        Args:
            api_keys: Explicit list of API keys. If None, loaded from GROQ_ENV_KEYS.
            base_url: Base URL for Groq API (default: https://api.groq.com/openai/v1).
            timeout: Default request timeout in seconds.
            http_client: Optional shared httpx.AsyncClient (useful for testing).

        Raises:
            ValueError: If no valid API keys are configured.
        """
        if api_keys is not None:
            resolved_keys = [k.strip() for k in api_keys if k and k.strip()]
        else:
            resolved_keys = []
            for env_var in GROQ_ENV_KEYS:
                val = os.environ.get(env_var, "").strip()
                if val:
                    resolved_keys.append(val)

        if not resolved_keys:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY (or GROQ_API_KEY_2..4) "
                "or pass api_keys to GroqKeyPool."
            )

        self._api_keys = resolved_keys
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._custom_client = http_client
        self._clients: dict[int, httpx.AsyncClient] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()

    @property
    def key_count(self) -> int:
        """Number of configured keys in the pool."""
        return len(self._api_keys)

    @property
    def api_keys(self) -> list[str]:
        """Return a copy of the configured keys list."""
        return list(self._api_keys)

    def _get_client_for_key(self, index: int) -> httpx.AsyncClient:
        """Get or create dedicated httpx.AsyncClient for a specific key index."""
        if self._custom_client is not None:
            return self._custom_client

        if index not in self._clients:
            self._clients[index] = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_keys[index]}"},
            )
        return self._clients[index]

    def _build_attempt_order(self, preferred_key_index: int | None) -> list[int]:
        """Determine key trial order starting with preferred key or round-robin."""
        n = len(self._api_keys)
        if preferred_key_index is not None and 0 <= preferred_key_index < n:
            # Preferred key first, followed by others in order
            return [preferred_key_index] + [i for i in range(n) if i != preferred_key_index]

        # Round-robin starting point
        start = self._current_index % n
        self._current_index = (self._current_index + 1) % n
        return [(start + i) % n for i in range(n)]

    async def call(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        preferred_key_index: int | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a chat completion request with key failover on 429/503/529.

        Args:
            model: Target Groq model identifier (e.g., 'openai/gpt-oss-120b').
            messages: List of chat message objects (role, content).
            preferred_key_index: 0-indexed preferred key to attempt first.
            response_format: Optional format specification, e.g. {"type": "json_object"}.
            temperature: Sampling temperature (default 0.2 for deterministic judging).
            max_tokens: Maximum completion tokens to generate.
            timeout: Per-request override timeout in seconds.
            **kwargs: Additional parameters passed to chat/completions endpoint.

        Returns:
            dict containing raw Groq API JSON response.

        Raises:
            AllKeysExhaustedError: If all keys in the pool fail due to rate limits or errors.
            GroqPoolError: On fatal non-retryable errors.
        """
        async with self._lock:
            attempt_indices = self._build_attempt_order(preferred_key_index)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error_code: int | None = None

        for key_idx in attempt_indices:
            client = self._get_client_for_key(key_idx)
            headers: dict[str, str] = {
                "Authorization": f"Bearer {self._api_keys[key_idx]}",
                "Content-Type": "application/json",
            }
            req_timeout = timeout if timeout is not None else self._timeout

            try:
                # If custom client or base_url difference, handle path correctly
                is_same_base = str(client.base_url).rstrip("/") == self._base_url
                post_url = (
                    "/chat/completions" if is_same_base else f"{self._base_url}/chat/completions"
                )
                response = await client.post(
                    post_url,
                    headers=headers,
                    json=payload,
                    timeout=req_timeout,
                )
            except httpx.RequestError as req_err:
                logger.warning(
                    "Groq key index %d request network error: %s. Rotating to fallback.",
                    key_idx,
                    type(req_err).__name__,
                )
                continue

            if response.status_code == 200:
                try:
                    data: dict[str, Any] = response.json()
                    return data
                except Exception as json_err:
                    logger.warning(
                        "Groq key index %d returned non-JSON 200 response: %s. Rotating.",
                        key_idx,
                        type(json_err).__name__,
                    )
                    continue

            # Transient / rate-limit failure codes that warrant rotation
            if response.status_code in (429, 503, 529):
                last_error_code = response.status_code
                logger.warning(
                    "Groq key index %d exhausted (HTTP %d). Rotating to fallback key.",
                    key_idx,
                    response.status_code,
                )
                continue

            # Auth errors: key might be expired or invalid, rotate to try remaining keys
            if response.status_code in (401, 403):
                logger.warning(
                    "Groq key index %d rejected credentials (HTTP %d). Rotating to fallback key.",
                    key_idx,
                    response.status_code,
                )
                continue

            # Unhandled status code: rotate as well to preserve pipeline availability
            logger.warning(
                "Groq key index %d failed with HTTP %d. Rotating to fallback key.",
                key_idx,
                response.status_code,
            )

        # If all keys failed
        err_detail = f" (last status HTTP {last_error_code})" if last_error_code else ""
        msg = (
            f"All {len(self._api_keys)} configured Groq API keys are exhausted "
            f"or unavailable{err_detail}."
        )
        raise AllKeysExhaustedError(msg)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        preferred_key_index: int | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method returning the text content of the first message choice."""
        res = await self.call(
            model=model,
            messages=messages,
            preferred_key_index=preferred_key_index,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choices = res.get("choices")
        if not choices or not isinstance(choices, list):
            raise GroqPoolError("Groq response missing valid 'choices' array.")

        choice = choices[0]
        message = choice.get("message")
        if not message or not isinstance(message, dict):
            raise GroqPoolError("Groq response choice missing valid 'message' object.")

        content = message.get("content")
        if content is None:
            raise GroqPoolError("Groq response message missing 'content'.")

        return str(content).strip()

    async def post_multipart(
        self,
        endpoint: str,
        *,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        preferred_key_index: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute an arbitrary multipart POST request with multi-key failover.

        Args:
            endpoint: API endpoint path (e.g. '/audio/transcriptions') or full URL.
            data: Form fields dictionary.
            files: Files dictionary in httpx multipart format.
            preferred_key_index: Preferred key index to try first.
            timeout: Per-request override timeout in seconds.

        Returns:
            dict containing parsed JSON response.

        Raises:
            GroqPoolError: On fatal non-retryable errors (e.g. HTTP 413).
            AllKeysExhaustedError: If all keys fail due to rate limits (429), errors, or network.
        """
        async with self._lock:
            attempt_indices = self._build_attempt_order(preferred_key_index)

        last_error_code: int | None = None
        last_reason: str = ""
        network_error_count = 0

        for key_idx in attempt_indices:
            client = self._get_client_for_key(key_idx)
            headers: dict[str, str] = {
                "Authorization": f"Bearer {self._api_keys[key_idx]}",
            }
            req_timeout = timeout if timeout is not None else self._timeout

            try:
                is_same_base = str(client.base_url).rstrip("/") == self._base_url
                if endpoint.startswith("http://") or endpoint.startswith("https://"):
                    post_url = endpoint
                else:
                    norm_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
                    post_url = norm_endpoint if is_same_base else f"{self._base_url}{norm_endpoint}"

                response = await client.post(
                    post_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=req_timeout,
                )
            except httpx.RequestError as req_err:
                network_error_count += 1
                logger.warning(
                    "Groq key index %d multipart network error: %s. Rotating to fallback.",
                    key_idx,
                    type(req_err).__name__,
                )
                continue

            if response.status_code == 200:
                try:
                    data_json: dict[str, Any] = response.json()
                    return data_json
                except Exception as json_err:
                    logger.warning(
                        "Groq key index %d returned non-JSON 200 response: %s. Rotating.",
                        key_idx,
                        type(json_err).__name__,
                    )
                    continue

            last_error_code = response.status_code
            last_reason = response.reason_phrase or "Error"

            # 413: file too large, no point in rotating
            if response.status_code == 413:
                raise GroqPoolError(
                    f"Groq API rejected upload: payload exceeds 25MB request limit "
                    f"(status 413: {last_reason})"
                )

            # 429, 503, 529: rate limits or capacity
            if response.status_code in (429, 503, 529):
                logger.warning(
                    "Groq key index %d exhausted during multipart (HTTP %d). Rotating to fallback.",
                    key_idx,
                    response.status_code,
                )
                continue

            # Auth errors (401, 403): rotate
            if response.status_code in (401, 403):
                logger.warning(
                    "Groq key index %d rejected credentials (HTTP %d). Rotating to fallback.",
                    key_idx,
                    response.status_code,
                )
                continue

            # Other errors: rotate
            logger.warning(
                "Groq key index %d returned status %d during multipart request. Rotating.",
                key_idx,
                response.status_code,
            )

        if network_error_count == len(attempt_indices):
            raise AllKeysExhaustedError(
                f"Groq API request failed due to network error on all "
                f"{len(self._api_keys)} keys."
            )

        if last_error_code is not None:
            raise AllKeysExhaustedError(
                f"Groq API request failed with status {last_error_code}: {last_reason}"
            )

        raise AllKeysExhaustedError(
            f"All Groq API keys ({len(self._api_keys)}) failed for multipart request."
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        *,
        model: str = "whisper-large-v3-turbo",
        mime_type: str = "audio/wav",
        preferred_key_index: int | None = None,
        timeout: float | None = None,
        response_format: str = "verbose_json",
        timestamp_granularities: list[str] | None = None,
        prompt: str | None = None,
        language: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Execute Whisper speech-to-text audio transcription with multi-key failover.

        Delegates to post_multipart with formatted Whisper payload.
        """
        data: dict[str, Any] = {
            "model": model,
            "response_format": response_format,
        }
        if timestamp_granularities is not None:
            data["timestamp_granularities[]"] = timestamp_granularities
        if prompt is not None:
            data["prompt"] = prompt
        if language is not None:
            data["language"] = language
        if temperature is not None:
            data["temperature"] = str(temperature)

        files = {
            "file": (filename, audio_bytes, mime_type),
        }

        return await self.post_multipart(
            "/audio/transcriptions",
            data=data,
            files=files,
            preferred_key_index=preferred_key_index,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close all managed httpx clients."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def close(self) -> None:
        """Alias for aclose()."""
        await self.aclose()

    async def __aenter__(self) -> "GroqKeyPool":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()


__all__ = [
    "DEFAULT_GROQ_BASE_URL",
    "DEFAULT_GROQ_TIMEOUT",
    "GROQ_ENV_KEYS",
    "AllKeysExhaustedError",
    "GroqKeyPool",
    "GroqPoolError",
    "GroqRateLimitError",
    "mask_key",
]
