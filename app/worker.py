"""Worker dispatcher for VirtuJudge AI-ML pipeline jobs."""

from datetime import UTC, datetime
from typing import Any

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    EraseAIDataPayload,
    FailedPayload,
    GenerateReportPayload,
    JobType,
    QueueMessage,
    StartedPayload,
    UpdateStatus,
    WorkerUpdate,
)
from app.pipeline import PitchAnalysisPipeline


def create_started_update(
    message: QueueMessage, pipeline_version: str = "1.0.0"
) -> WorkerUpdate:
    """Create a sequence=1 started WorkerUpdate for a given queue message."""
    return WorkerUpdate(
        schema_version=1,
        sequence=1,
        status=UpdateStatus.STARTED,
        occurred_at=datetime.now(UTC),
        trace_id=message.trace_id,
        payload=StartedPayload(pipeline_version=pipeline_version),
    )


async def process_job(
    message: QueueMessage, pipeline: PitchAnalysisPipeline
) -> WorkerUpdate:
    """Process an incoming QueueMessage and return the terminal WorkerUpdate."""
    try:
        result: Any
        if message.job_type == JobType.ANALYZE_SESSION:
            session_payload = AnalyzeSessionPayload.model_validate(message.payload)
            result = await pipeline.analyze_session(session_payload)
        elif message.job_type == JobType.ANALYZE_ANSWER:
            answer_payload = AnalyzeAnswerPayload.model_validate(message.payload)
            result = await pipeline.analyze_answer(answer_payload)
        elif message.job_type == JobType.GENERATE_REPORT:
            report_payload = GenerateReportPayload.model_validate(message.payload)
            result = await pipeline.generate_report(report_payload)
        elif message.job_type == JobType.ERASE_AI_DATA:
            erase_payload = EraseAIDataPayload.model_validate(message.payload)
            result = await pipeline.erase_data(erase_payload)
        else:
            raise ValueError(f"Unsupported job type: {message.job_type}")

        return WorkerUpdate(
            schema_version=1,
            sequence=2,
            status=UpdateStatus.COMPLETED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=result,
        )
    except Exception as exc:
        return WorkerUpdate(
            schema_version=1,
            sequence=2,
            status=UpdateStatus.FAILED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=FailedPayload(
                stage="dispatch",
                code="JOB_PROCESSING_FAILED",
                retryable=False,
                attempts=message.analysis_attempt,
                message=str(exc),
            ),
        )


__all__ = ["create_started_update", "process_job"]
