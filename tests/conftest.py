"""Pytest fixtures for VirtuJudge AI-ML tests."""

from datetime import UTC, datetime

import pytest

from app.contracts import JobType, QueueMessage
from app.pipeline import FakePipeline


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    """Fixture providing an instance of FakePipeline."""
    return FakePipeline()


@pytest.fixture
def analyze_session_message() -> QueueMessage:
    """QueueMessage fixture with job_type='analyze_session' and valid AnalyzeSessionPayload dict."""
    return QueueMessage(
        schema_version=1,
        job_id="01JTEST0000000000000000010",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id="01JTEST0000000000000000011",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTEST0000000000000000012",
        payload={
            "presentation": {
                "artifact_id": "01JTEST0000000000000000013",
                "object_key": "recordings/pitch_deck_presentation.mp4",
                "checksum": "sha256:" + "1" * 64,
                "media_type": "video/mp4",
            },
            "supporting_documents": [],
            "rubric": {
                "rubric_id": "startup_pitch",
                "version": 1,
            },
            "requested_capabilities": [
                "speech",
                "diarization",
                "vision",
                "audio",
                "documents",
                "questions",
            ],
        },
    )


@pytest.fixture
def analyze_answer_message() -> QueueMessage:
    """QueueMessage fixture with job_type='analyze_answer' and valid AnalyzeAnswerPayload dict."""
    return QueueMessage(
        schema_version=1,
        job_id="01JTEST0000000000000000020",
        job_type=JobType.ANALYZE_ANSWER,
        practice_session_id="01JTEST0000000000000000021",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTEST0000000000000000022",
        payload={
            "qa_round_id": "01JTEST0000000000000000023",
            "question_id": "01JTEST0000000000000000024",
            "answer_id": "01JTEST0000000000000000025",
            "answered_by": "01JTEST0000000000000000026",
            "audio": {
                "artifact_id": "01JTEST0000000000000000027",
                "object_key": "recordings/qa_answer_round_1.wav",
                "checksum": "sha256:" + "2" * 64,
                "media_type": "audio/wav",
                "duration_ms": 35000,
            },
            "remaining_follow_ups": 2,
        },
    )


@pytest.fixture
def generate_report_message() -> QueueMessage:
    """QueueMessage fixture with job_type='generate_report' and valid GenerateReportPayload dict."""
    return QueueMessage(
        schema_version=1,
        job_id="01JTEST0000000000000000030",
        job_type=JobType.GENERATE_REPORT,
        practice_session_id="01JTEST0000000000000000031",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTEST0000000000000000032",
        payload={
            "report_id": "01JTEST0000000000000000033",
            "analysis_artifact": {
                "artifact_id": "01JTEST0000000000000000034",
                "object_key": "artifacts/session_analysis.json",
                "checksum": "sha256:" + "3" * 64,
                "schema_version": 1,
            },
            "qa_artifact": {
                "artifact_id": "01JTEST0000000000000000035",
                "object_key": "artifacts/qa_round.json",
                "checksum": "sha256:" + "4" * 64,
                "schema_version": 1,
            },
            "speaker_mappings": [
                {
                    "speaker_label": "SPEAKER_00",
                    "user_id": "01JTEST0000000000000000036",
                    "display_name": "Alice Founder",
                },
                {
                    "speaker_label": "SPEAKER_01",
                    "user_id": "01JTEST0000000000000000037",
                    "display_name": "Bob CTO",
                },
            ],
        },
    )


@pytest.fixture
def erase_message() -> QueueMessage:
    """QueueMessage fixture with job_type='erase_ai_data' and valid EraseAIDataPayload dict."""
    return QueueMessage(
        schema_version=1,
        job_id="01JTEST0000000000000000040",
        job_type=JobType.ERASE_AI_DATA,
        practice_session_id="01JTEST0000000000000000041",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTEST0000000000000000042",
        payload={
            "erasure_request_id": "01JTEST0000000000000000043",
            "scope": "practice_session",
            "scope_id": "01JTEST0000000000000000044",
        },
    )
