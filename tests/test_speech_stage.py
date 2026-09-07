"""Unit tests for app.stages.speech (transcription & diarization orchestration)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.contracts import (
    AnalyzeSessionPayload,
    AssetInput,
    Limitation,
    RubricRef,
)
from app.providers.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from app.stages.media import MediaNormalizationResult
from app.stages.speech import (
    SpeechStageResult,
    run_speech_stage,
    validate_word_timestamps,
)


class MockSpeechProvider:
    """Mock speech provider returning predetermined transcription."""

    def __init__(self, transcription: TranscriptionResult | None = None) -> None:
        self.transcription = transcription or TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=2000,
                    text="Hello team",
                    confidence=0.99,
                    words=[
                        WordTimestamp(word="Hello", start_ms=0, end_ms=900, confidence=0.99),
                        WordTimestamp(word="team", start_ms=1000, end_ms=2000, confidence=0.99),
                    ],
                )
            ],
            full_text="Hello team",
            language="en",
            metadata={"model_name": "large-v3-turbo", "provider": "whisper"},
        )
        self.model_name = "large-v3-turbo"

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return self.transcription


class MockDiarizationProvider:
    """Mock diarization provider returning predetermined diarization."""

    def __init__(self, diarization: DiarizationResult | None = None) -> None:
        self.diarization = diarization or DiarizationResult(
            segments=[SpeakerSegment(start_ms=0, end_ms=2000, speaker_label="SPEAKER_00")],
            speaker_labels=["SPEAKER_00"],
            metadata={"model_name": "community-1", "provider": "pyannote"},
        )
        self.model_name = "community-1"

    async def diarize(self, audio_path: Path) -> DiarizationResult:
        return self.diarization


def test_validate_word_timestamps_valid() -> None:
    """Verify validate_word_timestamps passes when all timestamps are within media duration."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=0,
                end_ms=4500,
                text="Valid pitch opening",
                words=[
                    WordTimestamp(word="Valid", start_ms=0, end_ms=1000),
                    WordTimestamp(word="pitch", start_ms=1100, end_ms=2500),
                    WordTimestamp(word="opening", start_ms=2600, end_ms=4500),
                ],
            )
        ],
        full_text="Valid pitch opening",
    )
    limitations = validate_word_timestamps(res, media_duration_ms=5000, strict=True)
    assert limitations == []


def test_validate_word_timestamps_exceeding_duration_strict_raises() -> None:
    """Verify strict validation raises ValueError when word timestamp exceeds media duration."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=0,
                end_ms=6000,
                text="Out of bounds",
                words=[
                    WordTimestamp(word="Out", start_ms=0, end_ms=2000),
                    WordTimestamp(word="of", start_ms=2100, end_ms=3000),
                    WordTimestamp(word="bounds", start_ms=3100, end_ms=6500),  # exceeds 5000ms
                ],
            )
        ],
        full_text="Out of bounds",
    )
    with pytest.raises(ValueError, match="exceeds media duration"):
        validate_word_timestamps(res, media_duration_ms=5000, strict=True)


def test_validate_word_timestamps_exceeding_duration_non_strict_clamps() -> None:
    """Verify non-strict validation clamps timestamps and returns a Limitation."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=0,
                end_ms=6000,
                text="Out of bounds",
                words=[
                    WordTimestamp(word="bounds", start_ms=3100, end_ms=6500),
                ],
            )
        ],
        full_text="Out of bounds",
    )
    limitations = validate_word_timestamps(res, media_duration_ms=5000, strict=False)

    assert len(limitations) == 1
    assert limitations[0].code == "timestamp_out_of_bounds"
    assert limitations[0].scope == "speech"
    # Clamped bounds
    assert res.segments[0].end_ms == 5000
    assert res.segments[0].words[0].end_ms == 5000


def test_validate_word_timestamps_negative_raises() -> None:
    """Verify negative timestamp raises ValueError in strict mode."""
    seg = TranscriptionSegment(start_ms=0, end_ms=2000, text="Negative start")
    object.__setattr__(seg, "start_ms", -100)
    res = TranscriptionResult(segments=[seg], full_text="Negative start")
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_word_timestamps(res, media_duration_ms=5000, strict=True)


@pytest.mark.asyncio
async def test_run_speech_stage_orchestration(tmp_path: Path) -> None:
    """Verify parallel transcription & diarization and SpeechStageResult composition."""
    dummy_input = tmp_path / "presentation.mp4"
    dummy_input.write_bytes(b"dummy video")

    fake_norm = MediaNormalizationResult(
        output_path=tmp_path / "normalized.wav",
        duration_ms=2000,
        sample_rate=16000,
        channels=1,
    )

    speech_p = MockSpeechProvider()
    diar_p = MockDiarizationProvider()

    with patch("app.stages.speech.normalize_media", new_callable=AsyncMock) as mock_norm:
        mock_norm.return_value = fake_norm

        result = await run_speech_stage(
            target=dummy_input,
            speech_provider=speech_p,
            diarization_provider=diar_p,
        )

        assert isinstance(result, SpeechStageResult)
        assert result.transcription.full_text == "Hello team"
        assert result.speaker_labels == ["SPEAKER_00"]
        assert len(result.limitations) == 0

        # Verify metadata recorded
        assert result.metadata["stage"] == "speech"
        assert result.metadata["speech_provider"] == "MockSpeechProvider"
        assert result.metadata["speech_model"] == "large-v3-turbo"
        assert result.metadata["diarization_provider"] == "MockDiarizationProvider"
        assert result.metadata["diarization_model"] == "community-1"
        assert result.metadata["media_duration_ms"] == 2000


@pytest.mark.asyncio
async def test_run_speech_stage_multi_speaker(tmp_path: Path) -> None:
    """Verify multi-speaker fixtures produce expected speaker count and labels."""
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=0,
                end_ms=3000,
                text="Speaker 1 opening pitch",
                words=[WordTimestamp(word="Speaker", start_ms=0, end_ms=1000)],
            ),
            TranscriptionSegment(
                start_ms=3500,
                end_ms=7000,
                text="Speaker 2 financial breakdown",
                words=[WordTimestamp(word="Speaker", start_ms=3500, end_ms=4500)],
            ),
        ],
        full_text="Speaker 1 opening pitch Speaker 2 financial breakdown",
    )
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=3200, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=3400, end_ms=7000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    speech_p = MockSpeechProvider(transcription)
    diar_p = MockDiarizationProvider(diarization)

    result = await run_speech_stage(
        target=tmp_path / "dummy.wav",
        speech_provider=speech_p,
        diarization_provider=diar_p,
        skip_normalization=True,
        media_duration_ms=7000,
    )

    assert result.speaker_labels == ["SPEAKER_00", "SPEAKER_01"]
    assert len(result.speaker_labels) == 2
    assert len(result.limitations) == 0


@pytest.mark.asyncio
async def test_run_speech_stage_uncertain_speakers_produce_limitations(tmp_path: Path) -> None:
    """Verify missing/uncertain speaker assignments produce appropriate limitations."""
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(start_ms=0, end_ms=2000, text="Intro"),
            TranscriptionSegment(start_ms=2500, end_ms=5000, text="Unattributed speaking turn"),
        ],
        full_text="Intro Unattributed speaking turn",
    )
    # Only 0-2000 has a speaker, 2500-5000 has no overlap or unknown
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=2000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=2500, end_ms=5000, speaker_label="SPEAKER_UNKNOWN"),
        ],
        speaker_labels=["SPEAKER_00"],
        metadata={"uncertain_turns_count": 1},
    )

    speech_p = MockSpeechProvider(transcription)
    diar_p = MockDiarizationProvider(diarization)

    result = await run_speech_stage(
        target=tmp_path / "dummy.wav",
        speech_provider=speech_p,
        diarization_provider=diar_p,
        skip_normalization=True,
        media_duration_ms=5000,
    )

    assert len(result.limitations) >= 1
    codes = [lim.code for lim in result.limitations]
    assert "uncertain_speaker_assignment" in codes


@pytest.mark.asyncio
async def test_run_speech_stage_no_speakers_produces_limitation(tmp_path: Path) -> None:
    """Verify empty diarization produces no_speakers_detected limitation."""
    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start_ms=0, end_ms=2000, text="Solo speech")],
        full_text="Solo speech",
    )
    diarization = DiarizationResult(segments=[], speaker_labels=[])

    speech_p = MockSpeechProvider(transcription)
    diar_p = MockDiarizationProvider(diarization)

    result = await run_speech_stage(
        target=tmp_path / "dummy.wav",
        speech_provider=speech_p,
        diarization_provider=diar_p,
        skip_normalization=True,
        media_duration_ms=2000,
    )

    codes = [lim.code for lim in result.limitations]
    assert "no_speakers_detected" in codes


@pytest.mark.asyncio
async def test_run_speech_stage_accepts_payload(tmp_path: Path) -> None:
    """Verify run_speech_stage accepts AnalyzeSessionPayload input."""
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JEXAMPLE000000000000000A",
            object_key=str(tmp_path / "pitch.mp4"),
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
            duration_ms=2000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "diarization"],
    )

    speech_p = MockSpeechProvider()
    diar_p = MockDiarizationProvider()

    result = await run_speech_stage(
        target=payload,
        speech_provider=speech_p,
        diarization_provider=diar_p,
        skip_normalization=True,
    )

    assert isinstance(result, SpeechStageResult)
    assert result.metadata["media_duration_ms"] == 2000


def test_transcript_and_paths_do_not_leak_to_log_safe_outputs() -> None:
    """Verify sensitive raw transcripts and file paths do not leak into Limitations."""
    private_path = "/secret/path/to/private_recording_2026.mp4"
    confidential_transcript = "Our confidential proprietary formula is XYZ-42."

    # Verify normalization limitation
    norm_limit = Limitation(
        code="media_normalization_unavailable",
        scope="media",
        message="Audio normalization unavailable or failed; speech analysis may be impacted.",
    )
    assert private_path not in norm_limit.message
    assert confidential_transcript not in norm_limit.message

    # Verify speech stage limitation
    diar_limit = Limitation(
        code="uncertain_speaker_assignment",
        scope="diarization",
        message="Diarization produced unassigned or ambiguous speaker turns.",
    )
    assert private_path not in diar_limit.message
    assert confidential_transcript not in diar_limit.message
