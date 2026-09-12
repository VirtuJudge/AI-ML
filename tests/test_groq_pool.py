"""Unit tests for app.providers.groq_pool (Resilient GroqKeyPool)."""

import logging
from unittest.mock import AsyncMock

import httpx
import pytest

from app.providers.groq_pool import (
    AllKeysExhaustedError,
    GroqKeyPool,
    mask_key,
)


def test_mask_key_behavior() -> None:
    """Verify mask_key redacts secrets properly."""
    assert mask_key("") == "<empty>"
    assert mask_key("   ") == "<empty>"
    assert mask_key("short") == "***"
    assert mask_key("12345678") == "***"
    masked = mask_key("gsk_1234567890abcdef")
    assert masked == "gsk_...cdef"
    assert "1234567890" not in masked


def test_groq_pool_explicit_keys() -> None:
    """Verify initialization with explicit list of keys."""
    pool = GroqKeyPool(api_keys=["key1", "key2", "  ", "key3"])
    assert pool.key_count == 3
    assert pool.api_keys == ["key1", "key2", "key3"]


def test_groq_pool_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify initialization with environment variables."""
    monkeypatch.setenv("GROQ_API_KEY", "env_key_1")
    monkeypatch.setenv("GROQ_API_KEY_2", "env_key_2")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_4", raising=False)

    pool = GroqKeyPool()
    assert pool.key_count == 2
    assert pool.api_keys == ["env_key_1", "env_key_2"]


def test_groq_pool_no_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ValueError is raised when no keys are configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_4", raising=False)

    with pytest.raises(ValueError, match="No Groq API keys configured"):
        GroqKeyPool()

    with pytest.raises(ValueError, match="No Groq API keys configured"):
        GroqKeyPool(api_keys=[])


@pytest.mark.asyncio
async def test_groq_pool_preferred_key_routing() -> None:
    """Verify preferred_key_index routes request directly to selected key."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")

    client.post.return_value = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "response from preferred key"}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    pool = GroqKeyPool(api_keys=["key0", "key1", "key2"], http_client=client)

    res = await pool.call(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
        preferred_key_index=2,
    )

    assert res["choices"][0]["message"]["content"] == "response from preferred key"
    assert client.post.call_count == 1
    call_headers = client.post.call_args[1]["headers"]
    assert call_headers["Authorization"] == "Bearer key2"


@pytest.mark.asyncio
async def test_groq_pool_failover_on_429(caplog: pytest.LogCaptureFixture) -> None:
    """Verify automatic failover to next key when preferred key returns HTTP 429."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")

    resp_429 = httpx.Response(
        status_code=429,
        text="Rate limit reached",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    resp_200 = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "ok from backup"}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    client.post.side_effect = [resp_429, resp_200]

    pool = GroqKeyPool(api_keys=["keyA", "keyB"], http_client=client)

    with caplog.at_level(logging.WARNING):
        res = await pool.call(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            preferred_key_index=0,
        )

    assert res["choices"][0]["message"]["content"] == "ok from backup"
    assert client.post.call_count == 2
    first_call_headers = client.post.call_args_list[0][1]["headers"]
    second_call_headers = client.post.call_args_list[1][1]["headers"]
    assert first_call_headers["Authorization"] == "Bearer keyA"
    assert second_call_headers["Authorization"] == "Bearer keyB"
    assert "Groq key index 0 exhausted (HTTP 429). Rotating to fallback key." in caplog.text


@pytest.mark.asyncio
async def test_groq_pool_failover_on_network_error(caplog: pytest.LogCaptureFixture) -> None:
    """Verify failover when preferred key encounters a network error."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")

    resp_200 = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "recovered"}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    client.post.side_effect = [
        httpx.ConnectError("Connection refused"),
        resp_200,
    ]

    pool = GroqKeyPool(api_keys=["key1", "key2"], http_client=client)

    with caplog.at_level(logging.WARNING):
        res = await pool.call(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "hi"}],
            preferred_key_index=0,
        )

    assert res["choices"][0]["message"]["content"] == "recovered"
    assert client.post.call_count == 2
    assert "ConnectError" in caplog.text


@pytest.mark.asyncio
async def test_groq_pool_all_keys_exhausted() -> None:
    """Verify AllKeysExhaustedError is raised when all keys return 429."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")

    resp_429 = httpx.Response(
        status_code=429,
        text="Rate limit",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    client.post.return_value = resp_429

    pool = GroqKeyPool(api_keys=["k1", "k2", "k3"], http_client=client)

    with pytest.raises(AllKeysExhaustedError) as exc_info:
        await pool.call(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
        )

    assert "All 3 configured Groq API keys are exhausted" in str(exc_info.value)
    assert client.post.call_count == 3


@pytest.mark.asyncio
async def test_groq_pool_log_and_exception_hygiene(caplog: pytest.LogCaptureFixture) -> None:
    """Verify secrets and prompts are never leaked into logs or exceptions."""
    secret_key = "gsk_SUPER_SECRET_TOKEN_XYZ12345"
    secret_prompt = "CONFIDENTIAL_PITCH_TEXT_NEVER_LEAK"

    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")
    client.post.return_value = httpx.Response(
        status_code=429,
        text="Rate limit",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    pool = GroqKeyPool(api_keys=[secret_key], http_client=client)

    with caplog.at_level(logging.DEBUG), pytest.raises(AllKeysExhaustedError) as exc_info:
        await pool.call(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": secret_prompt}],
        )

    # Check exception message
    assert secret_key not in str(exc_info.value)
    assert secret_prompt not in str(exc_info.value)

    # Check logs
    assert secret_key not in caplog.text
    assert secret_prompt not in caplog.text


@pytest.mark.asyncio
async def test_groq_pool_chat_helper() -> None:
    """Verify chat() convenience method returns string content directly."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("https://api.groq.com/openai/v1")
    client.post.return_value = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "Direct text response"}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    pool = GroqKeyPool(api_keys=["mock_key"], http_client=client)
    content = await pool.chat(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "prompt"}],
    )
    assert content == "Direct text response"


@pytest.mark.asyncio
async def test_groq_pool_context_manager() -> None:
    """Verify async context manager protocol works."""
    async with GroqKeyPool(api_keys=["dummy_key"]) as pool:
        assert pool.key_count == 1
