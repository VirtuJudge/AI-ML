"""Worker dispatcher for VirtuJudge AI-ML pipeline jobs."""

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from typing import Any

from pydantic import ValidationError

from app.backend_client import BackendClientProtocol
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    EraseAIDataPayload,
    ErrorCode,
    FailedPayload,
    GenerateReportPayload,
    JobType,
    QueueMessage,
    StartedPayload,
    UpdateStatus,
    WorkerUpdate,
)
from app.pipeline import PitchAnalysisPipeline


def _map_validation_error(exc: ValidationError) -> tuple[ErrorCode, str]:
    """Map a Pydantic ValidationError to a safe, structured ErrorCode and message."""
    errors = exc.errors()
    has_checksum = any(
        "checksum" in str(e.get("loc", ())) or "checksum" in e.get("msg", "").lower()
        for e in errors
    )
    if has_checksum:
        return (
            ErrorCode.INVALID_CHECKSUM,
            "Payload contains an invalid checksum format.",
        )

    has_missing = any(e.get("type") == "missing" for e in errors)
    if has_missing:
        missing_fields = [
            str(e["loc"][-1]) for e in errors if e.get("type") == "missing" and e.get("loc")
        ]
        detail = f": {', '.join(missing_fields)}" if missing_fields else ""
        return (
            ErrorCode.MISSING_REQUIRED_FIELD,
            f"Payload missing required field(s){detail}.",
        )

    return (
        ErrorCode.MALFORMED_PAYLOAD,
        "Payload does not conform to expected schema.",
    )


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
    message: QueueMessage,
    pipeline: PitchAnalysisPipeline,
    backend_client: BackendClientProtocol | None = None,
) -> WorkerUpdate:
    """Process an incoming QueueMessage and return the terminal WorkerUpdate."""
    if backend_client is not None:
        started_update = create_started_update(message)
        await backend_client.send_update(message.job_id, started_update)

    terminal_update: WorkerUpdate
    try:
        result: Any
        if message.job_type == JobType.ANALYZE_SESSION:
            payload_dict = dict(message.payload)
            if "practice_session_id" not in payload_dict and message.practice_session_id:
                payload_dict["practice_session_id"] = message.practice_session_id
            session_payload = AnalyzeSessionPayload.model_validate(payload_dict)
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
            terminal_update = WorkerUpdate(
                schema_version=1,
                sequence=2,
                status=UpdateStatus.FAILED,
                occurred_at=datetime.now(UTC),
                trace_id=message.trace_id,
                payload=FailedPayload(
                    stage="dispatch",
                    code=ErrorCode.INVALID_JOB_TYPE,
                    retryable=False,
                    attempts=message.analysis_attempt,
                    message=f"Unsupported job type: {message.job_type}",
                ),
            )
            if backend_client is not None:
                await backend_client.send_update(message.job_id, terminal_update)
            return terminal_update

        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=2,
            status=UpdateStatus.COMPLETED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=result,
        )
    except ValidationError as exc:
        code, safe_message = _map_validation_error(exc)
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=2,
            status=UpdateStatus.FAILED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=FailedPayload(
                stage="validation",
                code=code,
                retryable=False,
                attempts=message.analysis_attempt,
                message=safe_message,
            ),
        )
    except Exception:
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=2,
            status=UpdateStatus.FAILED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=FailedPayload(
                stage="dispatch",
                code=ErrorCode.INTERNAL_ERROR,
                retryable=False,
                attempts=message.analysis_attempt,
                message="An internal error occurred during job processing.",
            ),
        )

    if backend_client is not None:
        await backend_client.send_update(message.job_id, terminal_update)

    return terminal_update


__all__ = ["create_started_update", "process_job"]

