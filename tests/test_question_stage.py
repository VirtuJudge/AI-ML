"""Unit tests for app.stages.questions (run_question_stage)."""

from unittest.mock import AsyncMock

import pytest

from app.contracts import Limitation, PrimaryQuestion
from app.providers.base import JudgeModelProvider
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    VisualObservation,
)
from app.stages.audio import AudioStageResult
from app.stages.questions import QuestionStageResult, run_question_stage
from app.stages.speech import SpeechStageResult
from app.stages.vision import VisionStageResult


def _make_mock_speech() -> SpeechStageResult:
    return SpeechStageResult(
        transcription=TranscriptionResult(
            full_text="We sell software to enterprises.",
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=4000,
                    text="We sell software to enterprises.",
                    confidence=0.95,
                    words=[],
                )
            ],
            language="en",
            duration_ms=4000,
        ),
        diarization=DiarizationResult(
            speaker_count=1,
            speaker_labels=["SPEAKER_00"],
            segments=[
                SpeakerSegment(start_ms=0, end_ms=4000, speaker_label="SPEAKER_00")
            ],
        ),
        speaker_labels=["SPEAKER_00"],
        limitations=[],
    )


def _make_mock_vision() -> VisionStageResult:
    return VisionStageResult(
        observations=[
            VisualObservation(
                start_ms=0,
                end_ms=2000,
                speaker_label="SPEAKER_00",
                metric="gaze_on_camera_ratio",
                value=0.85,
                unit="ratio",
                confidence=0.9,
            )
        ],
        person_tracks=["PERSON_00"],
        limitations=[],
    )


def _make_mock_audio() -> AudioStageResult:
    return AudioStageResult(
        observations=[
            AudioObservation(
                start_ms=0,
                end_ms=2000,
                speaker_label="SPEAKER_00",
                metric="speaking_rate_wpm",
                value=145.0,
                unit="wpm",
                confidence=0.9,
            )
        ],
        limitations=[],
    )


@pytest.mark.asyncio
async def test_run_question_stage_happy_path() -> None:
    """Verify run_question_stage aggregates evidence and queries judge provider."""
    mock_judge = AsyncMock(spec=JudgeModelProvider)
    mock_questions = [
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001A",
            text="What is your sales cycle length?",
            reason="Assesses market and business model.",
            rubric_dimension="market_and_business_model",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001B",
            text="How is your software protected against competitors?",
            reason="Assesses technology moat.",
            rubric_dimension="technology_and_moat",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001C",
            text="What are your pilot conversion targets?",
            reason="Assesses execution milestones.",
            rubric_dimension="execution_and_milestones",
            evidence_ids=["ev_doc_slide_01"],
        ),
    ]
    mock_judge.generate_questions.return_value = mock_questions

    doc_chunks = [
        DocumentChunk(
            chunk_id="chk_01",
            page_or_slide=1,
            text="Enterprise SaaS Platform Overview",
        )
    ]

    result = await run_question_stage(
        mock_judge,
        speech_result=_make_mock_speech(),
        vision_result=_make_mock_vision(),
        audio_result=_make_mock_audio(),
        document_chunks=doc_chunks,
        rubric_id="startup_pitch",
    )

    assert isinstance(result, QuestionStageResult)
    assert len(result.primary_questions) == 3
    assert result.evidence_bundle.has_documents is True
    assert result.evidence_bundle.has_vision is True
    assert result.evidence_bundle.has_audio is True
    assert result.metadata["question_count"] == 3
    assert result.metadata["has_documents"] is True
    assert mock_judge.generate_questions.call_count == 1


@pytest.mark.asyncio
async def test_run_question_stage_presentation_only() -> None:
    """Verify run_question_stage handles presentation without documents."""
    mock_judge = AsyncMock(spec=JudgeModelProvider)
    mock_judge.generate_questions.return_value = [
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001A",
            text="Question 1",
            reason="Reason 1",
            rubric_dimension="market_and_business_model",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001B",
            text="Question 2",
            reason="Reason 2",
            rubric_dimension="technology_and_moat",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="01JEXAMPLE000000000000001C",
            text="Question 3",
            reason="Reason 3",
            rubric_dimension="execution_and_milestones",
            evidence_ids=["ev_speech_001"],
        ),
    ]

    result = await run_question_stage(
        mock_judge,
        speech_result=_make_mock_speech(),
        rubric_id="startup_pitch",
    )

    assert len(result.primary_questions) == 3
    assert result.evidence_bundle.has_documents is False
    assert result.metadata["has_documents"] is False


@pytest.mark.asyncio
async def test_run_question_stage_merges_extra_limitations() -> None:
    """Verify extra limitations are retained in QuestionStageResult."""
    mock_judge = AsyncMock(spec=JudgeModelProvider)
    mock_judge.generate_questions.return_value = []

    extra_lim = Limitation(
        code="custom_limitation",
        scope="audio",
        message="Acoustic anomalies detected.",
        affected_dimensions=["acoustic_prosody"],
    )

    result = await run_question_stage(
        mock_judge,
        transcript="Test transcript",
        extra_limitations=[extra_lim],
    )

    assert any(lim.code == "custom_limitation" for lim in result.limitations)
