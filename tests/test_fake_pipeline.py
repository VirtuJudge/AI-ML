"""Unit tests for FakePipeline implementation."""

import pytest

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    EraseAIDataPayload,
    GenerateReportPayload,
    RubricRef,
    SpeakerMapping,
)
from app.pipeline import FakePipeline


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
