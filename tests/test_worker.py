"""Unit tests for the worker dispatcher."""

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from app.contracts import (
    AnswerAnalysisCompleted,
    ErasureCompleted,
    FailedPayload,
    JobType,
    QueueMessage,
    ReportCompleted,
    SessionAnalysisCompleted,
    UpdateStatus,
)
from app.pipeline import FakePipeline
from app.worker import create_started_update, process_job


@pytest.mark.asyncio
async def test_process_analyze_session_job(
    analyze_session_message: QueueMessage, fake_pipeline: FakePipeline
) -> None:
    """Test process_job returns completed WorkerUpdate with SessionAnalysisCompleted payload."""
    update = await process_job(analyze_session_message, fake_pipeline)
    assert update.status == UpdateStatus.COMPLETED
    assert update.sequence == 2
    assert update.trace_id == analyze_session_message.trace_id
    assert isinstance(update.payload, SessionAnalysisCompleted)
    assert len(update.payload.primary_questions) == 3


@pytest.mark.asyncio
async def test_process_analyze_answer_job(
    analyze_answer_message: QueueMessage, fake_pipeline: FakePipeline
) -> None:
    """Test process_job returns completed WorkerUpdate with AnswerAnalysisCompleted payload."""
    update = await process_job(analyze_answer_message, fake_pipeline)
    assert update.status == UpdateStatus.COMPLETED
    assert update.sequence == 2
    assert update.trace_id == analyze_answer_message.trace_id
    assert isinstance(update.payload, AnswerAnalysisCompleted)
    assert update.payload.answer_id == analyze_answer_message.payload["answer_id"]


@pytest.mark.asyncio
async def test_process_generate_report_job(
    generate_report_message: QueueMessage, fake_pipeline: FakePipeline
) -> None:
    """Test process_job returns completed WorkerUpdate with ReportCompleted payload."""
    update = await process_job(generate_report_message, fake_pipeline)
    assert update.status == UpdateStatus.COMPLETED
    assert update.sequence == 2
    assert update.trace_id == generate_report_message.trace_id
    assert isinstance(update.payload, ReportCompleted)
    assert len(update.payload.member_feedback_user_ids) == 2


@pytest.mark.asyncio
async def test_process_erase_job(
    erase_message: QueueMessage, fake_pipeline: FakePipeline
) -> None:
    """Test process_job returns completed WorkerUpdate with ErasureCompleted payload."""
    update = await process_job(erase_message, fake_pipeline)
    assert update.status == UpdateStatus.COMPLETED
    assert update.sequence == 2
    assert update.trace_id == erase_message.trace_id
    assert isinstance(update.payload, ErasureCompleted)
    assert update.payload.deleted_records == 14
    assert update.payload.deleted_objects == 6


@pytest.mark.asyncio
async def test_process_unknown_job_type_fails(fake_pipeline: FakePipeline) -> None:
    """Test process_job handles an unknown job_type by returning a failed WorkerUpdate."""
    invalid_message: Any = QueueMessage.model_construct(
        schema_version=1,
        job_id="01JTEST0000000000000000001",
        job_type=cast(JobType, "non_existent_type"),
        practice_session_id="01JTEST0000000000000000002",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTEST0000000000000000003",
        payload={},
    )

    update = await process_job(invalid_message, fake_pipeline)
    assert update.status == UpdateStatus.FAILED
    assert update.sequence == 2
    assert update.trace_id == "01JTEST0000000000000000003"
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.stage == "dispatch"
    assert "Unsupported job type" in update.payload.message


def test_create_started_update(analyze_session_message: QueueMessage) -> None:
    """Test create_started_update creates sequence=1 update with StartedPayload."""
    update = create_started_update(analyze_session_message, pipeline_version="2.0.0")
    assert update.status == UpdateStatus.STARTED
    assert update.sequence == 1
    assert update.trace_id == analyze_session_message.trace_id
    assert update.payload.pipeline_version == "2.0.0"
