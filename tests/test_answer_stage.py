"""Unit tests for app.stages.answers (run_answer_speech_stage)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.contracts import AssetInput
from app.providers.fake_speech import FakeSpeechProvider
from app.stages.answers import AnswerSpeechResult, run_answer_speech_stage
from app.stages.media import MediaNormalizationResult


@pytest.mark.asyncio
async def test_run_answer_speech_stage_normal(tmp_path: Path) -> None:
    """Verify normal answer transcription produces non-empty transcript, segments, and metadata."""
    audio_path = tmp_path / "answer.wav"
    audio_path.write_bytes(b"dummy wav data")
    speech_provider = FakeSpeechProvider()

    result = await run_answer_speech_stage(
        audio_path,
        speech_provider,
        media_duration_ms=15000,
        skip_normalization=True,
    )

    assert isinstance(result, AnswerSpeechResult)
    assert result.transcript.full_text != ""
    assert len(result.transcript.segments) > 0
    assert result.limitations == []
    assert result.metadata["stage"] == "answer_speech"
    assert result.metadata["speech_provider"] == "FakeSpeechProvider"
    assert result.metadata["media_duration_ms"] == 15000


@pytest.mark.asyncio
async def test_run_answer_speech_stage_skipped() -> None:
    """Verify skipped answer (audio_path=None) returns empty transcript and skipped metadata."""
    speech_provider = FakeSpeechProvider()

    result = await run_answer_speech_stage(None, speech_provider)

    assert isinstance(result, AnswerSpeechResult)
    assert result.transcript.full_text == ""
    assert result.transcript.segments == []
    assert result.limitations == []
    assert result.metadata.get("skipped") is True
    assert result.metadata.get("stage") == "answer_speech"


@pytest.mark.asyncio
async def test_run_answer_speech_stage_with_asset_input() -> None:
    """Verify AssetInput extracts object_key and duration_ms correctly."""
    speech_provider = FakeSpeechProvider()
    asset = AssetInput(
        artifact_id="01JTEST0000000000000000014",
        object_key="audio/answer_clip.wav",
        checksum="sha256:" + "b" * 64,
        media_type="audio/wav",
        duration_ms=25000,
    )

    with patch.object(
        speech_provider, "transcribe", wraps=speech_provider.transcribe
    ) as mock_transcribe:
        result = await run_answer_speech_stage(
            asset,
            speech_provider,
            skip_normalization=True,
        )

        mock_transcribe.assert_called_once_with(Path("audio/answer_clip.wav"))
        assert result.metadata["media_duration_ms"] == 25000
        assert result.transcript.full_text != ""
        assert len(result.transcript.segments) > 0
        assert result.limitations == []


@pytest.mark.asyncio
async def test_run_answer_speech_stage_timestamp_validation_limitation(tmp_path: Path) -> None:
    """Verify limitations are produced when transcription timestamps exceed media duration."""
    audio_path = tmp_path / "answer.wav"
    audio_path.write_bytes(b"dummy wav data")
    speech_provider = FakeSpeechProvider()

    # FakeSpeechProvider returns segments up to 14000ms.
    # Setting duration to 5000ms with strict_timestamps=False should produce a Limitation.
    result = await run_answer_speech_stage(
        audio_path,
        speech_provider,
        media_duration_ms=5000,
        skip_normalization=True,
        strict_timestamps=False,
    )

    assert len(result.limitations) > 0
    assert any(lim.code == "timestamp_out_of_bounds" for lim in result.limitations)
    assert any(lim.scope == "speech" for lim in result.limitations)


@pytest.mark.asyncio
async def test_run_answer_speech_stage_timestamp_validation_strict_raises(tmp_path: Path) -> None:
    """Verify strict timestamp validation raises ValueError when timestamps exceed duration."""
    audio_path = tmp_path / "answer.wav"
    audio_path.write_bytes(b"dummy wav data")
    speech_provider = FakeSpeechProvider()

    with pytest.raises(ValueError, match="exceeds media duration"):
        await run_answer_speech_stage(
            audio_path,
            speech_provider,
            media_duration_ms=5000,
            skip_normalization=True,
            strict_timestamps=True,
        )


@pytest.mark.asyncio
async def test_run_answer_speech_stage_with_normalization(tmp_path: Path) -> None:
    """Verify audio normalization path orchestrates properly when skip_normalization is False."""
    dummy_input = tmp_path / "answer.mp4"
    dummy_input.write_bytes(b"dummy video data")
    fake_norm = MediaNormalizationResult(
        output_path=tmp_path / "normalized.wav",
        duration_ms=15000,
        sample_rate=16000,
        channels=1,
    )
    speech_provider = FakeSpeechProvider()

    with patch(
        "app.stages.answers.normalize_media",
        new_callable=AsyncMock,
        return_value=fake_norm,
    ) as mock_norm:
        result = await run_answer_speech_stage(
            dummy_input,
            speech_provider,
            skip_normalization=False,
        )

        mock_norm.assert_called_once_with(dummy_input, output_path=None)
        assert result.metadata["media_duration_ms"] == 15000
        assert result.transcript.full_text != ""
        assert len(result.transcript.segments) > 0
