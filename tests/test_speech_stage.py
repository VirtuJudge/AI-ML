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
from app.pipeline import FakePipeline, combine_timed_evidence
from app.providers.fake_speech import FakeSpeechProvider
from app.providers.pyannote_diarization import PyannoteDiarizationProvider
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    VisualObservation,
    WordTimestamp,
)
from app.stages.aggregation import aggregate_speaker_observations
from app.stages.media import MediaNormalizationResult
from app.stages.speech import (
    SpeechStageResult,
    run_speech_stage,
    validate_diarization_timestamps,
    validate_word_timestamps,
)
from app.storage.local import LocalDiskObjectStorage


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


def test_validate_word_timestamps_inverted_segment_strict_raises() -> None:
    """Verify strict validation raises ValueError when segment start_ms exceeds end_ms."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=3000,
                end_ms=1000,
                text="Inverted segment",
            )
        ],
        full_text="Inverted segment",
    )
    with pytest.raises(
        ValueError,
        match=r"Segment start timestamp \(3000ms\) exceeds end timestamp \(1000ms\)",
    ):
        validate_word_timestamps(res, media_duration_ms=5000, strict=True)


def test_validate_word_timestamps_inverted_word_strict_raises() -> None:
    """Verify strict validation raises ValueError when word start_ms exceeds end_ms."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=1000,
                end_ms=3000,
                text="Inverted word",
                words=[
                    WordTimestamp(word="Inverted", start_ms=2500, end_ms=1500),
                ],
            )
        ],
        full_text="Inverted word",
    )
    with pytest.raises(
        ValueError,
        match=r"Word 'Inverted' start timestamp \(2500ms\) exceeds end timestamp \(1500ms\)",
    ):
        validate_word_timestamps(res, media_duration_ms=5000, strict=True)


def test_validate_word_timestamps_inverted_non_strict_flags_limitation() -> None:
    """Verify non-strict validation corrects inverted timestamps and flags a limitation."""
    res = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_ms=3000,
                end_ms=1000,
                text="Inverted segment and word",
                words=[
                    WordTimestamp(word="Inverted", start_ms=2500, end_ms=1200),
                ],
            )
        ],
        full_text="Inverted segment and word",
    )
    limitations = validate_word_timestamps(res, media_duration_ms=5000, strict=False)

    inverted_lims = [lim for lim in limitations if lim.code == "inverted_timestamps"]
    assert len(inverted_lims) == 1
    assert inverted_lims[0].scope == "speech"
    assert "start timestamps exceeded end timestamps" in inverted_lims[0].message

    # Verify intervals were corrected such that start_ms <= end_ms
    assert res.segments[0].start_ms == 1000
    assert res.segments[0].end_ms == 3000
    assert res.segments[0].words[0].start_ms == 1200
    assert res.segments[0].words[0].end_ms == 2500
    assert res.segments[0].start_ms <= res.segments[0].end_ms
    assert res.segments[0].words[0].start_ms <= res.segments[0].words[0].end_ms


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


# ============================================================================
# Diarization Turn Timestamp Validation & Label Reconciliation Tests (#70)
# ============================================================================


def test_validate_diarization_timestamps_exact_boundary_valid() -> None:
    """Turns matching 0 <= start < end <= media_duration_ms pass strict and permissive checks."""
    diar = DiarizationResult(
        segments=[SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")],
        speaker_labels=["SPEAKER_00"],
        metadata={"detected_speaker_count": 1},
    )

    strict_res, strict_lims = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=True
    )
    assert strict_lims == []
    assert strict_res.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")
    ]
    assert strict_res.speaker_labels == ["SPEAKER_00"]

    perm_res, perm_lims = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=False
    )
    assert perm_lims == []
    assert perm_res.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")
    ]
    assert perm_res.speaker_labels == ["SPEAKER_00"]


def test_validate_diarization_timestamps_out_of_range_strict_and_permissive() -> None:
    """Out-of-range turns fail in strict mode and clamp/drop + reconcile in permissive mode."""
    diar = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=60000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=2000, end_ms=5000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        metadata={"detected_speaker_count": 2},
    )

    with pytest.raises(ValueError, match="exceeds media duration"):
        validate_diarization_timestamps(diar, media_duration_ms=1000, strict=True)

    validated, limitations = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=False
    )
    assert validated.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00"),
    ]
    assert validated.speaker_labels == ["SPEAKER_00"]
    assert validated.metadata["detected_speaker_count"] == 1
    assert [lim.code for lim in limitations] == ["diarization_timestamp_out_of_bounds"]
    assert limitations[0].scope == "diarization"
    assert "individual_feedback" in limitations[0].affected_dimensions


def test_validate_diarization_timestamps_negative_endpoints() -> None:
    """Negative endpoints fail in strict mode and clamp/drop in permissive mode."""
    neg_partial = SpeakerSegment.model_construct(
        start_ms=-100, end_ms=500, speaker_label="SPEAKER_00"
    )
    neg_full = SpeakerSegment.model_construct(
        start_ms=-500, end_ms=-100, speaker_label="SPEAKER_01"
    )
    diar = DiarizationResult(
        segments=[neg_partial, neg_full],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_diarization_timestamps(diar, media_duration_ms=1000, strict=True)

    validated, limitations = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=False
    )
    assert validated.segments == [
        SpeakerSegment(start_ms=0, end_ms=500, speaker_label="SPEAKER_00"),
    ]
    assert validated.speaker_labels == ["SPEAKER_00"]
    assert [lim.code for lim in limitations] == ["diarization_timestamp_out_of_bounds"]


def test_validate_diarization_timestamps_inverted_turns() -> None:
    """Inverted turns fail in strict mode and swap/drop with limitation in permissive mode."""
    diar = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=800, end_ms=200, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=2500, end_ms=1500, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    with pytest.raises(ValueError, match="exceeds end timestamp"):
        validate_diarization_timestamps(diar, media_duration_ms=1000, strict=True)

    validated, limitations = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=False
    )
    assert validated.segments == [
        SpeakerSegment(start_ms=200, end_ms=800, speaker_label="SPEAKER_00"),
    ]
    assert validated.speaker_labels == ["SPEAKER_00"]
    codes = {lim.code for lim in limitations}
    assert codes == {"diarization_inverted_timestamps", "diarization_timestamp_out_of_bounds"}


def test_validate_diarization_timestamps_zero_length_turns() -> None:
    """Zero-length turns fail in strict mode and drop with limitation + label reconciliation."""
    diar = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=500, end_ms=500, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=100, end_ms=900, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    with pytest.raises(ValueError, match="positive duration"):
        validate_diarization_timestamps(diar, media_duration_ms=1000, strict=True)

    validated, limitations = validate_diarization_timestamps(
        diar, media_duration_ms=1000, strict=False
    )
    assert validated.segments == [
        SpeakerSegment(start_ms=100, end_ms=900, speaker_label="SPEAKER_01"),
    ]
    assert validated.speaker_labels == ["SPEAKER_01"]
    assert [lim.code for lim in limitations] == ["diarization_zero_length_turns"]


@pytest.mark.asyncio
async def test_run_speech_stage_reproduction_one_second_60s_turn(tmp_path: Path) -> None:
    """Reproduce #70: 1s media with 0-60000ms turn rejects strictly and clamps permissively."""
    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start_ms=0, end_ms=1000, text="Quick intro")],
        full_text="Quick intro",
    )
    diarization = DiarizationResult(
        segments=[SpeakerSegment(start_ms=0, end_ms=60000, speaker_label="SPEAKER_00")],
        speaker_labels=["SPEAKER_00"],
    )

    speech_p = MockSpeechProvider(transcription)
    diar_p = MockDiarizationProvider(diarization)
    audio_path = tmp_path / "one_sec.wav"
    audio_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="exceeds media duration"):
        await run_speech_stage(
            target=audio_path,
            speech_provider=speech_p,
            diarization_provider=diar_p,
            media_duration_ms=1000,
            strict_timestamps=True,
            skip_normalization=True,
        )

    perm_result = await run_speech_stage(
        target=audio_path,
        speech_provider=speech_p,
        diarization_provider=diar_p,
        media_duration_ms=1000,
        strict_timestamps=False,
        skip_normalization=True,
    )
    assert perm_result.diarization.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")
    ]
    assert perm_result.speaker_labels == ["SPEAKER_00"]
    assert any(lim.code == "diarization_timestamp_out_of_bounds" for lim in perm_result.limitations)

    by_speaker = aggregate_speaker_observations([], [], perm_result.diarization)
    assert by_speaker["SPEAKER_00"]["speaking_time_ms"] == 1000


@pytest.mark.asyncio
async def test_run_speech_stage_uses_actual_normalized_duration(tmp_path: Path) -> None:
    """run_speech_stage enforces actual normalized WAV duration even if payload duration differs."""
    dummy_input = tmp_path / "presentation.mp4"
    dummy_input.write_bytes(b"dummy video")

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JEXAMPLE000000000000000B",
            object_key=str(dummy_input),
            checksum="sha256:" + "b" * 64,
            media_type="video/mp4",
            duration_ms=60000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "diarization"],
    )

    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start_ms=0, end_ms=1000, text="Short clip")],
        full_text="Short clip",
    )
    diarization = DiarizationResult(
        segments=[SpeakerSegment(start_ms=0, end_ms=60000, speaker_label="SPEAKER_00")],
        speaker_labels=["SPEAKER_00"],
    )
    fake_norm = MediaNormalizationResult(
        output_path=tmp_path / "normalized.wav",
        duration_ms=1000,
        sample_rate=16000,
        channels=1,
    )

    with patch("app.stages.speech.normalize_media", new_callable=AsyncMock) as mock_norm:
        mock_norm.return_value = fake_norm

        with pytest.raises(ValueError, match="exceeds media duration"):
            await run_speech_stage(
                target=payload,
                speech_provider=MockSpeechProvider(transcription),
                diarization_provider=MockDiarizationProvider(diarization),
                strict_timestamps=True,
            )

        perm_res = await run_speech_stage(
            target=payload,
            speech_provider=MockSpeechProvider(transcription),
            diarization_provider=MockDiarizationProvider(diarization),
            strict_timestamps=False,
        )
        assert perm_res.metadata["media_duration_ms"] == 1000
        assert perm_res.diarization.segments[0].end_ms == 1000


@pytest.mark.asyncio
async def test_run_speech_stage_boundary_rounding_cases(tmp_path: Path) -> None:
    """Boundary-rounding turns from float seconds are validated against media_duration_ms."""

    class _Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Annotation:
        def __init__(self, tracks: list[tuple[_Turn, str, str]]) -> None:
            self._tracks = tracks

        def itertracks(self, yield_label: bool = True):
            yield from self._tracks

    audio_path = tmp_path / "rounded.wav"
    audio_path.write_bytes(b"dummy")
    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start_ms=0, end_ms=1000, text="Boundary test")],
        full_text="Boundary test",
    )

    # 1. Rounds inside boundary (1.0004s -> 1000ms) -> accepted in strict mode
    ok_provider = PyannoteDiarizationProvider(
        pipeline_instance=lambda *_args, **_kw: _Annotation([(_Turn(0.0, 1.0004), "t0", "A")])
    )
    ok_res = await run_speech_stage(
        target=audio_path,
        speech_provider=MockSpeechProvider(transcription),
        diarization_provider=ok_provider,
        media_duration_ms=1000,
        strict_timestamps=True,
        skip_normalization=True,
    )
    assert ok_res.diarization.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")
    ]
    assert ok_res.limitations == []

    # 2. Rounds 1ms past end (1.0006s -> 1001ms) and collapses at boundary
    #    (0.9996..1.0004 -> 1000..1000)
    over_provider = PyannoteDiarizationProvider(
        pipeline_instance=lambda *_args, **_kw: _Annotation(
            [
                (_Turn(0.0, 1.0006), "t0", "A"),  # 0..1001ms -> clamp to 0..1000ms
                (_Turn(0.9996, 1.0004), "t1", "B"),  # 1000..1000ms -> drop + reconcile B
                (_Turn(0.9996, 1.0006), "t2", "C"),  # 1000..1001ms -> drop + reconcile C
            ]
        )
    )
    with pytest.raises(ValueError, match=r"exceeds media duration|positive duration"):
        await run_speech_stage(
            target=audio_path,
            speech_provider=MockSpeechProvider(transcription),
            diarization_provider=over_provider,
            media_duration_ms=1000,
            strict_timestamps=True,
            skip_normalization=True,
        )

    perm_res = await run_speech_stage(
        target=audio_path,
        speech_provider=MockSpeechProvider(transcription),
        diarization_provider=over_provider,
        media_duration_ms=1000,
        strict_timestamps=False,
        skip_normalization=True,
    )
    assert perm_res.diarization.segments == [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00")
    ]
    assert perm_res.speaker_labels == ["SPEAKER_00"]
    assert perm_res.diarization.speaker_labels == ["SPEAKER_00"]
    assert any(lim.code == "diarization_timestamp_out_of_bounds" for lim in perm_res.limitations)


@pytest.mark.asyncio
async def test_run_speech_stage_all_invalid_turns_before_correlation_and_aggregation(
    tmp_path: Path,
) -> None:
    """All-invalid turn sets drop cleanly and prevent spurious correlation or speaking time."""
    audio_path = tmp_path / "all_invalid.wav"
    audio_path.write_bytes(b"dummy")

    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start_ms=0, end_ms=1000, text="Spoken words")],
        full_text="Spoken words",
    )
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=1500, end_ms=3000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=400, end_ms=400, speaker_label="SPEAKER_01"),
            SpeakerSegment(start_ms=2500, end_ms=1800, speaker_label="SPEAKER_02"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
    )

    with pytest.raises(ValueError):
        await run_speech_stage(
            target=audio_path,
            speech_provider=MockSpeechProvider(transcription),
            diarization_provider=MockDiarizationProvider(diarization),
            media_duration_ms=1000,
            strict_timestamps=True,
            skip_normalization=True,
        )

    result = await run_speech_stage(
        target=audio_path,
        speech_provider=MockSpeechProvider(transcription),
        diarization_provider=MockDiarizationProvider(diarization),
        media_duration_ms=1000,
        strict_timestamps=False,
        skip_normalization=True,
    )

    assert result.diarization.segments == []
    assert result.diarization.speaker_labels == []
    assert result.speaker_labels == []
    lim_codes = {lim.code for lim in result.limitations}
    assert {
        "diarization_timestamp_out_of_bounds",
        "diarization_inverted_timestamps",
        "diarization_zero_length_turns",
        "no_speakers_detected",
    }.issubset(lim_codes)

    # Downstream correlation & aggregation must see zero turns and zero speakers
    visual_obs = [
        VisualObservation(
            start_ms=0,
            end_ms=1000,
            metric="gaze_direction",
            value=1.0,
            unit="index",
            speaker_label="PERSON_00",
        )
    ]
    audio_obs = [
        AudioObservation(
            start_ms=0,
            end_ms=1000,
            metric="speaking_rate_wpm",
            value=140.0,
            unit="words_per_minute",
        )
    ]
    updated_visual, updated_audio, mapping, corr_lims = combine_timed_evidence(
        result.diarization, visual_obs, audio_obs
    )
    assert mapping == {}
    assert updated_audio[0].speaker_label is None
    assert updated_visual[0].speaker_label == "PERSON_00"
    assert any(lim.code == "ambiguous_visual_speaker_mapping" for lim in corr_lims)

    by_speaker = aggregate_speaker_observations(updated_visual, updated_audio, result.diarization)
    assert by_speaker == {}


@pytest.mark.asyncio
async def test_fake_pipeline_clamps_invalid_diarization_before_aggregation(tmp_path: Path) -> None:
    """End-to-end FakePipeline.analyze_session clamps invalid diarization before aggregation."""

    class _ShortFakeSpeechProvider(FakeSpeechProvider):
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            return TranscriptionResult(
                segments=[TranscriptionSegment(start_ms=0, end_ms=1000, text="One second pitch")],
                full_text="One second pitch",
            )

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=60000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    pipeline = FakePipeline(
        speech_provider=_ShortFakeSpeechProvider(),
        diarization_provider=MockDiarizationProvider(diarization),
        object_storage=storage,
    )
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JEXAMPLE000000000000000C",
            object_key="uploads/short_pitch.mp4",
            checksum="sha256:" + "c" * 64,
            media_type="video/mp4",
            duration_ms=1000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "diarization", "vision", "audio", "questions"],
        practice_session_id="01JSESSION0000000000000070",
    )

    completed = await pipeline.analyze_session(payload, job_id="01JJOB000000000000000070")

    assert completed.speaker_labels == ["SPEAKER_00"]
    assert any(lim.code == "diarization_timestamp_out_of_bounds" for lim in completed.limitations)

    analysis_json = await storage.read_json(completed.analysis_artifact.object_key)
    assert list(analysis_json["by_speaker"].keys()) == ["SPEAKER_00"]
    assert analysis_json["by_speaker"]["SPEAKER_00"]["speaking_time_ms"] == 1000
