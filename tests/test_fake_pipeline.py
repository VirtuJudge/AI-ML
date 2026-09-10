"""Unit tests for FakePipeline implementation."""

from pathlib import Path
from typing import Any

import pytest

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    EraseAIDataPayload,
    GenerateReportPayload,
    Limitation,
    RubricRef,
    SpeakerMapping,
)
from app.pipeline import FakePipeline, combine_timed_evidence
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    SpeakerSegment,
    VisualObservation,
)


@pytest.mark.asyncio
async def test_analyze_session_completes(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_session returns 3 primary questions and validates properly."""
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
    )

    result = await fake_pipeline.analyze_session(payload)
    assert len(result.primary_questions) == 3
    assert result.analysis_artifact.artifact_id is not None
    assert result.analysis_artifact.checksum.startswith("sha256:")
    assert len(result.speaker_labels) == 2


@pytest.mark.asyncio
async def test_analyze_session_result_is_stable(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_session returns stable, identical results for repeated calls."""
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )

    result1 = await fake_pipeline.analyze_session(payload)
    result2 = await fake_pipeline.analyze_session(payload)
    assert result1 == result2


@pytest.mark.asyncio
async def test_analyze_answer_completes(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_answer returns transcript and assessment artifact IDs."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=1,
    )

    result = await fake_pipeline.analyze_answer(payload)
    assert result.answer_id == payload.answer_id
    assert result.transcript_artifact_id is not None
    assert result.assessment_artifact_id is not None
    assert result.follow_up is not None


@pytest.mark.asyncio
async def test_analyze_answer_no_followup_when_zero_remaining(
    fake_pipeline: FakePipeline,
) -> None:
    """Test that analyze_answer returns follow_up=None when remaining_follow_ups is 0."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=0,
    )

    result = await fake_pipeline.analyze_answer(payload)
    assert result.follow_up is None


@pytest.mark.asyncio
async def test_generate_report_completes(fake_pipeline: FakePipeline) -> None:
    """Test report generation returns mapped user IDs for feedback."""
    speaker_mappings = [
        SpeakerMapping(
            speaker_label="SPEAKER_00",
            user_id="01JTEST0000000000000000031",
            display_name="Founder Alice",
        ),
        SpeakerMapping(
            speaker_label="SPEAKER_01",
            user_id="01JTEST0000000000000000032",
            display_name="CTO Bob",
        ),
    ]
    payload = GenerateReportPayload(
        report_id="01JTEST0000000000000000030",
        analysis_artifact=ArtifactRef(
            artifact_id="01JTEST0000000000000000033",
            object_key="artifacts/analysis.json",
            checksum="sha256:" + "c" * 64,
            schema_version=1,
        ),
        qa_artifact=ArtifactRef(
            artifact_id="01JTEST0000000000000000034",
            object_key="artifacts/qa.json",
            checksum="sha256:" + "d" * 64,
            schema_version=1,
        ),
        speaker_mappings=speaker_mappings,
    )

    result = await fake_pipeline.generate_report(payload)
    assert len(result.member_feedback_user_ids) == 2
    assert "01JTEST0000000000000000031" in result.member_feedback_user_ids
    assert "01JTEST0000000000000000032" in result.member_feedback_user_ids


@pytest.mark.asyncio
async def test_erase_data_completes(fake_pipeline: FakePipeline) -> None:
    """Test data erasure returns expected record and object counts."""
    payload = EraseAIDataPayload(
        erasure_request_id="01JTEST0000000000000000040",
        scope="practice_session",
        scope_id="01JTEST0000000000000000041",
    )

    result = await fake_pipeline.erase_data(payload)
    assert result.erasure_request_id == "01JTEST0000000000000000040"
    assert result.deleted_records == 14
    assert result.deleted_objects == 6


def test_combine_timed_evidence_audio_attribution() -> None:
    """Verify audio observations without speaker labels are attributed to the active speaker turn."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    audio_obs = [
        AudioObservation(start_ms=0, end_ms=4000, metric="speaking_rate_wpm", value=140.0, unit="wpm"),
        AudioObservation(start_ms=6000, end_ms=9000, metric="speaking_rate_wpm", value=130.0, unit="wpm"),
    ]
    _, updated_audio, _, _ = combine_timed_evidence(diarization, [], audio_obs)
    assert len(updated_audio) == 2
    assert updated_audio[0].speaker_label == "SPEAKER_00"
    assert updated_audio[1].speaker_label == "SPEAKER_01"


def test_combine_timed_evidence_deduplicates_fragmented_tracks() -> None:
    """Verify visual tracks across edit cuts (PERSON_00 and PERSON_02) both map to SPEAKER_00."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=10000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=10000, end_ms=20000, speaker_label="SPEAKER_01"),
            SpeakerSegment(start_ms=20000, end_ms=30000, speaker_label="SPEAKER_00"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    visual_obs = [
        # PERSON_00 speaks during 0-10s
        VisualObservation(
            start_ms=1000, end_ms=2000, metric="mouth_aspect_ratio", value=0.25, unit="ratio", speaker_label="PERSON_00"
        ),
        VisualObservation(
            start_ms=1000, end_ms=2000, metric="gaze_direction", value=1.0, unit="index", speaker_label="PERSON_00"
        ),
        # PERSON_01 speaks during 10-20s
        VisualObservation(
            start_ms=11000, end_ms=12000, metric="mouth_aspect_ratio", value=0.30, unit="ratio", speaker_label="PERSON_01"
        ),
        # Presenter A returns in new track PERSON_02 during SPEAKER_00's turn (20-30s)
        VisualObservation(
            start_ms=21000, end_ms=22000, metric="mouth_aspect_ratio", value=0.28, unit="ratio", speaker_label="PERSON_02"
        ),
        VisualObservation(
            start_ms=21000, end_ms=22000, metric="gaze_direction", value=1.0, unit="index", speaker_label="PERSON_02"
        ),
    ]

    v_res, _, mapping, lims = combine_timed_evidence(diarization, visual_obs, [])

    assert mapping["PERSON_00"] == "SPEAKER_00"
    assert mapping["PERSON_01"] == "SPEAKER_01"
    assert mapping["PERSON_02"] == "SPEAKER_00"
    assert len(lims) == 0

    # All PERSON_00 and PERSON_02 observations must be relabeled as SPEAKER_00
    for obs in v_res:
        if obs.start_ms < 10000 or obs.start_ms >= 20000:
            assert obs.speaker_label == "SPEAKER_00"
        else:
            assert obs.speaker_label == "SPEAKER_01"


def test_combine_timed_evidence_co_presenters_mar_disambiguation() -> None:
    """Verify co-presenters on stage simultaneously are differentiated by mouth_aspect_ratio."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    visual_obs = [
        # During 0-5s: PERSON_00 has high MAR (speaking), PERSON_01 has low MAR (listening)
        VisualObservation(
            start_ms=1000, end_ms=2000, metric="mouth_aspect_ratio", value=0.26, unit="ratio", speaker_label="PERSON_00"
        ),
        VisualObservation(
            start_ms=1000, end_ms=2000, metric="mouth_aspect_ratio", value=0.04, unit="ratio", speaker_label="PERSON_01"
        ),
        # During 5-10s: PERSON_01 has high MAR (speaking), PERSON_00 has low MAR (listening)
        VisualObservation(
            start_ms=6000, end_ms=7000, metric="mouth_aspect_ratio", value=0.05, unit="ratio", speaker_label="PERSON_00"
        ),
        VisualObservation(
            start_ms=6000, end_ms=7000, metric="mouth_aspect_ratio", value=0.29, unit="ratio", speaker_label="PERSON_01"
        ),
    ]

    _, _, mapping, _ = combine_timed_evidence(diarization, visual_obs, [])
    assert mapping["PERSON_00"] == "SPEAKER_00"
    assert mapping["PERSON_01"] == "SPEAKER_01"


def test_combine_timed_evidence_ambiguous_track_limitation() -> None:
    """Verify an ambiguous visual track that never spoke produces a limitation."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    # PERSON_SILENT appears during 15-20s when no speech occurs
    visual_obs = [
        VisualObservation(
            start_ms=15000, end_ms=16000, metric="gaze_direction", value=1.0, unit="index", speaker_label="PERSON_SILENT"
        ),
    ]
    _, _, mapping, lims = combine_timed_evidence(diarization, visual_obs, [])
    assert "PERSON_SILENT" not in mapping
    assert any(l.code == "ambiguous_visual_speaker_mapping" for l in lims)


@pytest.mark.asyncio
async def test_analyze_session_aggregates_limitations() -> None:
    """Verify limitations from all stages are preserved and aggregated."""

    class VisionWithLimitationProvider:
        async def analyze_video(self, video_path: Path, **kwargs: Any) -> list[VisualObservation]:
            return []

    pipeline = FakePipeline(vision_provider=VisionWithLimitationProvider())
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )
    result = await pipeline.analyze_session(payload)
    limitation_codes = [lim.code for lim in result.limitations]
    assert "no_visual_observations" in limitation_codes

