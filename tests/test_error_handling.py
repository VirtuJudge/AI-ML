"""Tests for safe error handling and input validation in the worker."""

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    EraseAIDataPayload,
    ErasureCompleted,
    ErrorCode,
    FailedPayload,
    GenerateReportPayload,
    JobType,
    QueueMessage,
    ReportCompleted,
    SessionAnalysisCompleted,
    UpdateStatus,
)
from app.pipeline import FakePipeline
from app.worker import process_job


class ThrowingPipeline:
    """A pipeline mock that deliberately raises an internal exception with sensitive data."""

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted:
        raise RuntimeError(
            "Sensitive internal secret: API_KEY_SECRET_12345 in /internal/secrets.json"
        )

    async def analyze_answer(self, job: AnalyzeAnswerPayload) -> AnswerAnalysisCompleted:
        raise RuntimeError("Sensitive answer failure")

    async def generate_report(self, job: GenerateReportPayload) -> ReportCompleted:
        raise RuntimeError("Sensitive report failure")

    async def erase_data(self, job: EraseAIDataPayload) -> ErasureCompleted:
        raise RuntimeError("Sensitive erasure failure")


@pytest.mark.asyncio
async def test_malformed_payload_returns_safe_error(fake_pipeline: FakePipeline) -> None:
    """Test that missing required fields return MISSING_REQUIRED_FIELD with safe message."""
    message = QueueMessage(
        schema_version=1,
        job_id="01JTESTERR000000000000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id="01JTESTERR000000000000002",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTESTERR000000000000003",
        payload={},  # Missing 'presentation' and 'rubric'
    )

    update = await process_job(message, fake_pipeline)
    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.MISSING_REQUIRED_FIELD
    assert update.payload.stage == "validation"
    assert update.payload.retryable is False
    assert "missing required field" in update.payload.message.lower()


@pytest.mark.asyncio
async def test_invalid_checksum_returns_safe_error(fake_pipeline: FakePipeline) -> None:
    """Test that invalid checksum returns INVALID_CHECKSUM."""
    message = QueueMessage(
        schema_version=1,
        job_id="01JTESTERR000000000000004",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id="01JTESTERR000000000000005",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTESTERR000000000000006",
        payload={
            "presentation": {
                "artifact_id": "01JTEST0000000000000000013",
                "object_key": "recordings/pitch.mp4",
                "checksum": "md5:invalidchecksumstring",
                "media_type": "video/mp4",
            },
            "supporting_documents": [],
            "rubric": {"rubric_id": "startup_pitch", "version": 1},
        },
    )

    update = await process_job(message, fake_pipeline)
    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.INVALID_CHECKSUM
    assert update.payload.stage == "validation"
    assert update.payload.retryable is False


@pytest.mark.asyncio
async def test_unknown_job_type_returns_safe_error(fake_pipeline: FakePipeline) -> None:
    """Test that unsupported job type returns INVALID_JOB_TYPE."""
    message: Any = QueueMessage.model_construct(
        schema_version=1,
        job_id="01JTESTERR000000000000007",
        job_type=cast(JobType, "unsupported_operation"),
        practice_session_id="01JTESTERR000000000000008",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTESTERR000000000000009",
        payload={},
    )

    update = await process_job(message, fake_pipeline)
    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.INVALID_JOB_TYPE
    assert update.payload.stage == "dispatch"
    assert update.payload.retryable is False


@pytest.mark.asyncio
async def test_error_message_does_not_leak_internals(
    analyze_session_message: QueueMessage,
) -> None:
    """Verify that unexpected exceptions do not leak stack traces or internal secrets."""
    throwing_pipeline = ThrowingPipeline()
    update = await process_job(analyze_session_message, throwing_pipeline)

    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.INTERNAL_ERROR
    assert update.payload.stage == "dispatch"

    # Crucial security checks: internal details are suppressed
    assert "API_KEY_SECRET" not in update.payload.message
    assert "secrets.json" not in update.payload.message
    assert "Traceback" not in update.payload.message
    assert "RuntimeError" not in update.payload.message
