"""Unit tests for graceful pyannote diarization fallback in run_speech_stage."""

from pathlib import Path

import pytest

from app.contracts import (
    AnalyzeSessionPayload,
    AssetInput,
    Limitation,
    RubricRef,
)
from app.providers.types import (
    DiarizationResult,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from app.stages.speech import SpeechStageResult, run_speech_stage


class MockSpeechProvider:
    """Mock speech provider returning predetermined transcription."""

    def __init__(self, transcription: TranscriptionResult | None = None) -> None:
        self.transcription = transcription or TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=2000,
                    text="First section of pitch",
                    confidence=0.98,
                    words=[
                        WordTimestamp(word="First", start_ms=0, end_ms=500, confidence=0.98),
                        WordTimestamp(word="section", start_ms=600, end_ms=1200, confidence=0.98),
                        WordTimestamp(word="of", start_ms=1300, end_ms=1500, confidence=0.98),
                        WordTimestamp(word="pitch", start_ms=1600, end_ms=2000, confidence=0.98),
                    ],
                ),
                TranscriptionSegment(
                    start_ms=2100,
                    end_ms=4000,
                    text="Second section of pitch",
                    confidence=0.97,
                    words=[
                        WordTimestamp(word="Second", start_ms=2100, end_ms=2700, confidence=0.97),
                        WordTimestamp(word="section", start_ms=2800, end_ms=3300, confidence=0.97),
                        WordTimestamp(word="of", start_ms=3400, end_ms=3600, confidence=0.97),
                        WordTimestamp(word="pitch", start_ms=3700, end_ms=4000, confidence=0.97),
                    ],
                ),
            ],
            full_text="First section of pitch Second section of pitch",
            language="en",
            metadata={"model": "whisper-large-v3-turbo", "provider": "groq"},
        )
        self.model_name = "whisper-large-v3-turbo"

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return self.transcription


class FailingDiarizationProvider:
    """Mock diarization provider that always raises an exception simulating token/model errors."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError(
            "Could not download model pyannote/speaker-diarization-community-1: "
            "HuggingFace token missing or invalid. Cache: /root/.cache/huggingface/hub/models"
        )
        self.model_name = "pyannote-community-1"

    async def diarize(self, audio_path: Path) -> DiarizationResult:
        raise self.error


@pytest.mark.asyncio
async def test_diarization_failure_fallback_with_path_input(tmp_path: Path) -> None:
    """Verify diarization failure gracefully falls back to single SPEAKER_00 with Path input."""
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"RIFF dummy wav audio data")

    speech_provider = MockSpeechProvider()
    diarization_provider = FailingDiarizationProvider()

    result = await run_speech_stage(
        target=dummy_audio,
        speech_provider=speech_provider,
        diarization_provider=diarization_provider,
        media_duration_ms=4500,
        skip_normalization=True,
    )

    assert isinstance(result, SpeechStageResult)

    # 1. Verify transcription succeeded completely
    assert result.transcription == speech_provider.transcription
    assert len(result.transcription.segments) == 2

    # 2. Verify fallback speaker label is SPEAKER_00
    assert result.speaker_labels == ["SPEAKER_00"]
    assert len(result.diarization.segments) == 2
    for seg in result.diarization.segments:
        assert seg.speaker_label == "SPEAKER_00"

    # Segment timing bounds should match transcript segments
    assert result.diarization.segments[0].start_ms == 0
    assert result.diarization.segments[0].end_ms == 2000
    assert result.diarization.segments[1].start_ms == 2100
    assert result.diarization.segments[1].end_ms == 4000

    # 3. Verify limitation was generated
    diar_limitations = [lim for lim in result.limitations if lim.code == "diarization_unavailable"]
    assert len(diar_limitations) == 1
    lim = diar_limitations[0]
    assert isinstance(lim, Limitation)
    assert lim.code == "diarization_unavailable"
    assert lim.scope == "diarization"
    assert (
        lim.message
        == "Speaker diarization was unavailable; all speech attributed to a single speaker."
    )
    assert lim.affected_dimensions == ["individual_feedback", "speaker_attribution"]


@pytest.mark.asyncio
async def test_diarization_failure_fallback_with_payload_input(tmp_path: Path) -> None:
    """Verify diarization failure gracefully falls back with AnalyzeSessionPayload input."""
    dummy_audio = tmp_path / "pitch_video.mp4"
    dummy_audio.write_bytes(b"dummy video content")

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JEXAMPLE000000000000000A",
            object_key=str(dummy_audio),
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
            duration_ms=5000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "diarization"],
    )

    speech_provider = MockSpeechProvider()
    diarization_provider = FailingDiarizationProvider()

    result = await run_speech_stage(
        target=payload,
        speech_provider=speech_provider,
        diarization_provider=diarization_provider,
        skip_normalization=True,
    )

    assert isinstance(result, SpeechStageResult)
    assert result.speaker_labels == ["SPEAKER_00"]
    assert result.transcription.full_text == "First section of pitch Second section of pitch"

    # Check diarization_unavailable limitation is present
    assert any(lim.code == "diarization_unavailable" for lim in result.limitations)


@pytest.mark.asyncio
async def test_diarization_failure_limitation_is_log_safe(tmp_path: Path) -> None:
    """Verify limitation message does not leak sensitive exception details, paths, or internals."""
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"dummy")

    secret_cache_path = "/secret/root/cache/models--pyannote--diarization"
    secret_token = "hf_sensitive_token_abc123"
    custom_error = RuntimeError(f"Failed to auth with {secret_token} accessing {secret_cache_path}")

    speech_provider = MockSpeechProvider()
    diarization_provider = FailingDiarizationProvider(error=custom_error)

    result = await run_speech_stage(
        target=dummy_audio,
        speech_provider=speech_provider,
        diarization_provider=diarization_provider,
        media_duration_ms=4500,
        skip_normalization=True,
    )

    # Check all limitations for leaks
    for limitation in result.limitations:
        assert secret_token not in limitation.message
        assert secret_cache_path not in limitation.message
        assert "RuntimeError" not in limitation.message
        assert str(dummy_audio) not in limitation.message

    # Check metadata for leaks
    diar_metadata_str = str(result.metadata.get("diarization_metadata", {}))
    assert secret_token not in diar_metadata_str
    assert secret_cache_path not in diar_metadata_str


@pytest.mark.asyncio
async def test_diarization_failure_various_exceptions(tmp_path: Path) -> None:
    """Verify fallback activates for different exception types (OSError, ValueError, Exception)."""
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"dummy")

    for err in [
        ValueError("Invalid model config"),
        OSError("Disk read error"),
        ImportError("pyannote.audio not installed"),
        Exception("Unknown failure"),
    ]:
        speech_provider = MockSpeechProvider()
        diarization_provider = FailingDiarizationProvider(error=err)

        result = await run_speech_stage(
            target=dummy_audio,
            speech_provider=speech_provider,
            diarization_provider=diarization_provider,
            media_duration_ms=4500,
            skip_normalization=True,
        )

        assert result.speaker_labels == ["SPEAKER_00"]
        assert any(lim.code == "diarization_unavailable" for lim in result.limitations)
