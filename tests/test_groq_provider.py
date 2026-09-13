"""Unit tests for app.providers.groq_speech (GroqCloud Whisper API adapter)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.providers.base import SpeechProvider
from app.providers.groq_speech import (
    GroqSpeechProvider,
    GroqSpeechProviderError,
    ProviderError,
)
from app.providers.types import TranscriptionResult, WordTimestamp


def test_groq_provider_satisfies_protocol() -> None:
    """Verify GroqSpeechProvider conforms to SpeechProvider Protocol."""
    provider = GroqSpeechProvider(api_key="gsk_mock_api_key")
    # Verify structural subtyping for SpeechProvider protocol
    speech_provider: SpeechProvider = provider
    assert hasattr(speech_provider, "transcribe")
    assert callable(speech_provider.transcribe)
    assert asyncio.iscoroutinefunction(speech_provider.transcribe)
    assert provider.model_name == "whisper-large-v3-turbo"


def test_groq_provider_missing_api_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ValueError is raised when no API key is provided and env var is not set."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Groq API key is required"):
        GroqSpeechProvider()

    with pytest.raises(ValueError, match="Groq API key is required"):
        GroqSpeechProvider(api_key="")

    with pytest.raises(ValueError, match="Groq API key is required"):
        GroqSpeechProvider(api_key="   ")


def test_groq_provider_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GroqSpeechProvider reads GROQ_API_KEY from environment."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_key_123")
    provider = GroqSpeechProvider()
    assert provider.api_key == "gsk_env_key_123"


@pytest.mark.asyncio
async def test_groq_transcribe_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when audio file does not exist."""
    provider = GroqSpeechProvider(api_key="gsk_test")
    non_existent = tmp_path / "non_existent.wav"

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        await provider.transcribe(non_existent)


@pytest.mark.asyncio
async def test_groq_transcribe_success_with_top_level_words(tmp_path: Path) -> None:
    """Verify successful transcription and mapping when words are at the root level."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF dummy wav audio content")

    api_response = {
        "text": "Hello world and welcome to VirtuJudge",
        "language": "en",
        "duration": 4.5,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "avg_logprob": -0.15,
            },
            {
                "id": 1,
                "start": 2.0,
                "end": 4.5,
                "text": "and welcome to VirtuJudge",
                "avg_logprob": -0.22,
            },
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.8},
            {"word": "world", "start": 0.9, "end": 1.9},
            {"word": "and", "start": 2.1, "end": 2.3},
            {"word": "welcome", "start": 2.4, "end": 3.0},
            {"word": "to", "start": 3.1, "end": 3.3},
            {"word": "VirtuJudge", "start": 3.4, "end": 4.5},
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=api_response,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test_key_abc")
        result = await provider.transcribe(audio_file)

        assert isinstance(result, TranscriptionResult)
        assert result.full_text == "Hello world and welcome to VirtuJudge"
        assert result.language == "en"

        # Verify metadata is recorded
        assert result.metadata["provider"] == "groq"
        assert result.metadata["model"] == "whisper-large-v3-turbo"
        assert result.metadata["language"] == "en"
        assert result.metadata["duration"] == 4.5

        # Verify segments
        assert len(result.segments) == 2
        seg0 = result.segments[0]
        assert seg0.start_ms == 0
        assert seg0.end_ms == 2000
        assert seg0.text == "Hello world"
        assert len(seg0.words) == 2
        assert seg0.words[0].word == "Hello"
        assert seg0.words[0].start_ms == 0
        assert seg0.words[0].end_ms == 800
        assert isinstance(seg0.words[0], WordTimestamp)
        assert seg0.words[1].word == "world"
        assert seg0.words[1].start_ms == 900
        assert seg0.words[1].end_ms == 1900

        seg1 = result.segments[1]
        assert seg1.start_ms == 2000
        assert seg1.end_ms == 4500
        assert seg1.text == "and welcome to VirtuJudge"
        assert len(seg1.words) == 4
        assert seg1.words[0].word == "and"
        assert seg1.words[3].word == "VirtuJudge"
        assert seg1.words[3].start_ms == 3400
        assert seg1.words[3].end_ms == 4500

        # Verify HTTP request fields
        mock_post.assert_awaited_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer gsk_test_key_abc"
        assert call_kwargs["data"]["model"] == "whisper-large-v3-turbo"
        assert call_kwargs["data"]["response_format"] == "verbose_json"
        assert call_kwargs["data"]["timestamp_granularities[]"] == ["word", "segment"]
        assert "file" in call_kwargs["files"]


@pytest.mark.asyncio
async def test_groq_transcribe_success_with_nested_words(tmp_path: Path) -> None:
    """Verify successful transcription when words are nested within segment objects."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF dummy wav audio content")

    api_response = {
        "text": "Pitch presentation opening.",
        "language": "en",
        "duration": 2.5,
        "segments": [
            {
                "id": 0,
                "start": 0.2,
                "end": 2.3,
                "text": "Pitch presentation opening.",
                "avg_logprob": -0.05,
                "words": [
                    {"word": "Pitch", "start": 0.2, "end": 0.7, "probability": 0.98},
                    {"word": "presentation", "start": 0.8, "end": 1.6, "probability": 0.95},
                    {"word": "opening.", "start": 1.7, "end": 2.3, "probability": 0.97},
                ],
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=api_response,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test_key_nested")
        result = await provider.transcribe(audio_file)

        assert len(result.segments) == 1
        assert result.segments[0].start_ms == 200
        assert result.segments[0].end_ms == 2300
        words = result.segments[0].words
        assert len(words) == 3
        assert words[0].word == "Pitch"
        assert words[0].start_ms == 200
        assert words[0].end_ms == 700
        assert words[0].confidence == 0.98


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (403, "Forbidden"),
        (429, "Too Many Requests"),
        (500, "Internal Server Error"),
        (503, "Service Unavailable"),
    ],
)
async def test_groq_api_error_handling_no_leakage(
    tmp_path: Path,
    status_code: int,
    reason: str,
) -> None:
    """Verify 4xx/5xx responses raise ProviderError and do NOT leak response bodies into logs."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"dummy")

    secret_raw_body = (
        '{"error":{"message":"Sensitive internal path '
        '/var/secrets/key.pem leaked","type":"auth_error"}}'
    )
    mock_response = httpx.Response(
        status_code=status_code,
        text=secret_raw_body,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test")

        with pytest.raises(ProviderError) as exc_info:
            await provider.transcribe(audio_file)

        err_str = str(exc_info.value)
        # Exception should clearly state status code
        assert str(status_code) in err_str
        # Exception MUST NOT leak raw response body or secrets
        assert "Sensitive internal path" not in err_str
        assert "/var/secrets/key.pem" not in err_str
        assert secret_raw_body not in err_str
        assert isinstance(exc_info.value, GroqSpeechProviderError)


@pytest.mark.asyncio
async def test_groq_network_error_handling(tmp_path: Path) -> None:
    """Verify network connection errors raise GroqSpeechProviderError safely."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"dummy")

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused to internal:8080")
        provider = GroqSpeechProvider(api_key="gsk_test")

        with pytest.raises(GroqSpeechProviderError) as exc_info:
            await provider.transcribe(audio_file)

        err_str = str(exc_info.value)
        assert "network error" in err_str.lower()


@pytest.mark.asyncio
async def test_groq_transcribe_file_size_exceeds_limit_raises_provider_error(
    tmp_path: Path,
) -> None:
    """Verify file exceeding 25MB limit raises ProviderError before opening stream/posting."""
    audio_file = tmp_path / "oversized.wav"
    audio_file.write_bytes(b"dummy")

    provider = GroqSpeechProvider(api_key="gsk_test", max_file_size_bytes=100)

    # Audio file has 5 bytes, let's write 150 bytes to exceed 100-byte test threshold
    audio_file.write_bytes(b"a" * 150)

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        with pytest.raises(ProviderError) as exc_info:
            await provider.transcribe(audio_file)

        mock_post.assert_not_called()
        err_str = str(exc_info.value)
        assert "exceeds Groq's 25MB upload limit" in err_str
        assert isinstance(exc_info.value, GroqSpeechProviderError)


@pytest.mark.asyncio
async def test_groq_api_413_payload_too_large_maps_to_provider_error(tmp_path: Path) -> None:
    """Verify HTTP 413 rejection is cleanly mapped to GroqSpeechProviderError/ProviderError."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"dummy audio content")

    mock_response = httpx.Response(
        status_code=413,
        text='{"error": "Payload Too Large"}',
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test")

        with pytest.raises(ProviderError) as exc_info:
            await provider.transcribe(audio_file)

        err_str = str(exc_info.value)
        assert "413" in err_str
        assert "25MB" in err_str
        assert isinstance(exc_info.value, GroqSpeechProviderError)


@pytest.mark.asyncio
async def test_groq_transcribe_both_top_and_nested_words_prefers_nested(
    tmp_path: Path,
) -> None:
    """Verify that when both top-level and nested words exist, nested words are used."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"dummy audio content")

    api_response = {
        "text": "Hello world",
        "language": "en",
        "duration": 2.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.8, "probability": 0.95},
                    {"word": "world", "start": 0.9, "end": 1.9, "probability": 0.90},
                ],
            }
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.8},
            {"word": "world", "start": 0.9, "end": 1.9},
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=api_response,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test")
        result = await provider.transcribe(audio_file)

        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 2
        assert result.segments[0].words[0].confidence == 0.95


@pytest.mark.asyncio
async def test_groq_transcribe_synthesizes_segment_when_segments_empty(
    tmp_path: Path,
) -> None:
    """Verify single segment is synthesized when response has no segments but has text/words."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"dummy audio content")

    api_response = {
        "text": "Standalone sentence.",
        "language": "en",
        "duration": 1.5,
        "segments": [],
        "words": [
            {"word": "Standalone", "start": 0.1, "end": 0.8, "probability": 0.92},
            {"word": "sentence.", "start": 0.9, "end": 1.4, "probability": 0.88},
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=api_response,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GroqSpeechProvider(api_key="gsk_test")
        result = await provider.transcribe(audio_file)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Standalone sentence."
        assert result.segments[0].start_ms == 100
        assert result.segments[0].end_ms == 1400
        assert len(result.segments[0].words) == 2


@pytest.mark.asyncio
async def test_groq_transcribe_multi_key_failover_on_429(tmp_path: Path) -> None:
    """Verify GroqSpeechProvider automatically fails over to Key 2 when Key 1 hits 429."""
    from app.providers.groq_pool import GroqKeyPool

    audio_file = tmp_path / "pitch.wav"
    audio_file.write_bytes(b"dummy wav content")

    pool = GroqKeyPool(api_keys=["gsk_key_1", "gsk_key_2"])
    provider = GroqSpeechProvider(key_pool=pool)

    # Key 1 returns 429 rate limit, Key 2 returns 200 success
    resp_429 = httpx.Response(
        status_code=429,
        text='{"error": "rate_limit_exceeded"}',
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "text": "Pitch transcribed with backup key.",
            "language": "en",
            "duration": 3.0,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Pitch transcribed with backup key.",
                }
            ],
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp_429, resp_200]

        result = await provider.transcribe(audio_file)

        assert result.full_text == "Pitch transcribed with backup key."
        assert mock_post.await_count == 2
        # Verify first call used Key 1 and second call used Key 2
        first_auth = mock_post.call_args_list[0].kwargs["headers"]["Authorization"]
        second_auth = mock_post.call_args_list[1].kwargs["headers"]["Authorization"]
        assert first_auth == "Bearer gsk_key_1"
        assert second_auth == "Bearer gsk_key_2"


@pytest.mark.asyncio
async def test_groq_speech_with_custom_key_pool() -> None:
    """Verify GroqSpeechProvider accepts an external GroqKeyPool."""
    from app.providers.groq_pool import GroqKeyPool

    pool = GroqKeyPool(api_keys=["gsk_custom_1", "gsk_custom_2"])
    provider = GroqSpeechProvider(key_pool=pool)

    assert provider.key_pool is pool
    assert provider.api_key == "gsk_custom_1"



