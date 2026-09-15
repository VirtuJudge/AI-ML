"""Worker dispatcher and Celery transport task for VirtuJudge AI-ML pipeline jobs."""

import asyncio
import contextlib
import logging
import os
import random
import shutil
from collections.abc import Coroutine
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from app.backend_client import (
    BackendClient,
    BackendClientProtocol,
    BackendUnauthorizedError,
    JobConflictError,
    JobNotFoundError,
)
from app.celery_app import celery_app
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    CancelledPayload,
    EraseAIDataPayload,
    ErrorCode,
    FailedPayload,
    GenerateReportPayload,
    JobCancelledError,
    JobType,
    ProgressPayload,
    QueueMessage,
    StartedPayload,
    UpdateStatus,
    WorkerUpdate,
)
from app.pipeline import PitchAnalysisPipeline
from app.stages.common import StageTransientError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_default_pipeline: PitchAnalysisPipeline | None = None
_default_backend_client: BackendClientProtocol | None = None


def set_default_pipeline(pipeline: PitchAnalysisPipeline | None) -> None:
    """Set global pipeline instance for Celery worker (useful in tests)."""
    global _default_pipeline
    _default_pipeline = pipeline


def set_default_backend_client(client: BackendClientProtocol | None) -> None:
    """Set global backend client instance for Celery worker (useful in tests)."""
    global _default_backend_client
    _default_backend_client = client


async def get_or_create_pipeline() -> PitchAnalysisPipeline:
    """Get active pipeline or instantiate live/fake pipeline from environment."""
    global _default_pipeline
    if _default_pipeline is None:
        from app.__main__ import build_pipeline

        _default_pipeline = await build_pipeline()
    return _default_pipeline


def get_or_create_backend_client() -> BackendClientProtocol:
    """Get active backend client or instantiate from environment."""
    global _default_backend_client
    if _default_backend_client is None:
        _default_backend_client = BackendClient()
    return _default_backend_client


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Execute an async coroutine safely whether inside or outside an existing loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


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


class MonotonicUpdateTracker:
    """Client wrapper tracking strictly monotonic sequence numbers for a job."""

    def __init__(self, backend_client: BackendClientProtocol, trace_id: str) -> None:
        self.backend_client = backend_client
        self.trace_id = trace_id
        self.sequence = 0

    @property
    def next_sequence(self) -> int:
        return self.sequence + 1

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None:
        if update.status == UpdateStatus.STARTED:
            self.sequence = 1
            update.sequence = 1
        else:
            self.sequence += 1
            update.sequence = self.sequence
        return await self.backend_client.send_update(job_id, update)

    async def send_progress(self, job_id: str, stage: str, progress: float, message: str) -> None:
        self.sequence += 1
        update = WorkerUpdate(
            schema_version=1,
            sequence=self.sequence,
            status=UpdateStatus.PROGRESS,
            occurred_at=datetime.now(UTC),
            trace_id=self.trace_id,
            payload=ProgressPayload(stage=stage, progress=progress, message=message),
        )
        await self.backend_client.send_update(job_id, update)

    async def check_cancellation(self, job_id: str) -> bool:
        return await self.backend_client.check_cancellation(job_id)

    async def check_backend_reachability(self) -> bool:
        return await self.backend_client.check_backend_reachability()

    async def close(self) -> None:
        await self.backend_client.close()


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
    tracker: MonotonicUpdateTracker | None = None
    client_to_use: BackendClientProtocol | None = backend_client

    if backend_client is not None:
        tracker = MonotonicUpdateTracker(backend_client, trace_id=message.trace_id)
        client_to_use = tracker
        started_update = create_started_update(message)
        try:
            await tracker.send_update(message.job_id, started_update)
        except BackendUnauthorizedError:
            logger.error("Fatal worker authorization failure sending started update.")
            raise
        except (JobNotFoundError, JobConflictError) as exc:
            logger.warning("Job %s rejected at start: %s", message.job_id, exc)
            raise

    terminal_update: WorkerUpdate
    try:
        result: Any
        if message.job_type == JobType.ANALYZE_SESSION:
            payload_dict = dict(message.payload)
            if "practice_session_id" not in payload_dict and message.practice_session_id:
                payload_dict["practice_session_id"] = message.practice_session_id
            session_payload = AnalyzeSessionPayload.model_validate(payload_dict)
            result = await pipeline.analyze_session(
                session_payload,
                job_id=message.job_id,
                backend_client=client_to_use,
                attempt=message.analysis_attempt,
            )
        elif message.job_type == JobType.ANALYZE_ANSWER:
            payload_dict = dict(message.payload)
            if "practice_session_id" not in payload_dict and message.practice_session_id:
                payload_dict["practice_session_id"] = message.practice_session_id
            answer_payload = AnalyzeAnswerPayload.model_validate(payload_dict)
            result = await pipeline.analyze_answer(
                answer_payload,
                job_id=message.job_id,
                backend_client=client_to_use,
                attempt=message.analysis_attempt,
            )
        elif message.job_type == JobType.GENERATE_REPORT:
            report_payload = GenerateReportPayload.model_validate(message.payload)
            result = await pipeline.generate_report(
                report_payload,
                job_id=message.job_id,
                backend_client=client_to_use,
                attempt=message.analysis_attempt,
                practice_session_id=message.practice_session_id,
            )
        elif message.job_type == JobType.ERASE_AI_DATA:
            erase_payload = EraseAIDataPayload.model_validate(message.payload)
            result = await pipeline.erase_data(
                erase_payload,
                job_id=message.job_id,
                backend_client=client_to_use,
            )
        else:
            terminal_seq = tracker.next_sequence if tracker else 2
            terminal_update = WorkerUpdate(
                schema_version=1,
                sequence=terminal_seq,
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
            if client_to_use is not None:
                await client_to_use.send_update(message.job_id, terminal_update)
            return terminal_update

        terminal_seq = tracker.next_sequence if tracker else 2
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=terminal_seq,
            status=UpdateStatus.COMPLETED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=result,
        )
    except JobCancelledError as exc:
        if message.practice_session_id:
            temp_dir = Path(".storage/temp_media") / message.practice_session_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        asset_ids: list[str] = []
        presentation = message.payload.get("presentation", {})
        audio = message.payload.get("audio", {})
        if isinstance(presentation, dict) and presentation.get("artifact_id"):
            asset_ids.append(str(presentation["artifact_id"]))
        if isinstance(audio, dict) and audio.get("artifact_id"):
            asset_ids.append(str(audio["artifact_id"]))
        for asset_id in asset_ids:
            temp_dir = Path(".storage/temp_media") / asset_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        terminal_seq = tracker.next_sequence if tracker else 2
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=terminal_seq,
            status=UpdateStatus.CANCELLED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=CancelledPayload(
                stage=exc.stage,
                message=exc.message,
            ),
        )
    except ValidationError as exc:
        code, safe_message = _map_validation_error(exc)
        terminal_seq = tracker.next_sequence if tracker else 2
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=terminal_seq,
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
    except StageTransientError as exc:
        terminal_seq = tracker.next_sequence if tracker else 2
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=terminal_seq,
            status=UpdateStatus.FAILED,
            occurred_at=datetime.now(UTC),
            trace_id=message.trace_id,
            payload=FailedPayload(
                stage=getattr(exc, "stage", "speech"),
                code=ErrorCode.PROVIDER_ERROR,
                retryable=True,
                attempts=message.analysis_attempt,
                message="A transient external provider error occurred.",
            ),
        )
    except Exception:
        terminal_seq = tracker.next_sequence if tracker else 2
        terminal_update = WorkerUpdate(
            schema_version=1,
            sequence=terminal_seq,
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

    if client_to_use is not None:
        try:
            await client_to_use.send_update(message.job_id, terminal_update)
        except (BackendUnauthorizedError, JobNotFoundError, JobConflictError) as exc:
            logger.warning("Terminal callback for job %s terminated with %s", message.job_id, exc)

    return terminal_update


async def _async_process_job_task(
    task: Any,
    envelope: dict[str, Any],
    pipeline: PitchAnalysisPipeline | None = None,
    backend_client: BackendClientProtocol | None = None,
) -> WorkerUpdate | None:
    """Inner coroutine executing one Celery job task."""
    if pipeline is None:
        pipeline = await get_or_create_pipeline()
    if backend_client is None:
        backend_client = get_or_create_backend_client()

    # 1. Validate envelope model
    try:
        message = QueueMessage.model_validate(envelope)
    except ValidationError as val_exc:
        code, safe_msg = _map_validation_error(val_exc)
        job_id = str(envelope.get("job_id", ""))
        trace_id = str(envelope.get("trace_id", "trc_validation_error"))
        attempt = int(envelope.get("analysis_attempt", 1))
        if job_id and backend_client:
            update = WorkerUpdate(
                schema_version=1,
                sequence=1,
                status=UpdateStatus.FAILED,
                occurred_at=datetime.now(UTC),
                trace_id=trace_id,
                payload=FailedPayload(
                    stage="validation",
                    code=code,
                    retryable=False,
                    attempts=attempt,
                    message=safe_msg,
                ),
            )
            with contextlib.suppress(Exception):
                await backend_client.send_update(job_id, update)
        # Class 1: non-retryable invalid input. Acknowledge and exit.
        return None
    except Exception as exc:
        logger.error("Non-retryable envelope parsing error: %s", exc)
        return None

    # 2. Dispatch job
    try:
        update = await process_job(message, pipeline, backend_client)
        if (
            update is not None
            and update.status == UpdateStatus.FAILED
            and isinstance(update.payload, FailedPayload)
            and update.payload.retryable
            and task is not None
        ):
            retries = getattr(task.request, "retries", 0)
            max_retries = getattr(task, "max_retries", 3)
            if retries < max_retries:
                countdown = int(min(60, (2**retries) * 5 + random.uniform(0, 2)))
                logger.warning(
                    "Transient error on job %s (attempt %d/%d). Retrying in %ds.",
                    message.job_id,
                    retries + 1,
                    max_retries,
                    countdown,
                )
                raise task.retry(countdown=countdown)
        return update
    except BackendUnauthorizedError as exc:
        logger.error("Fatal worker authorization error on job %s: %s. Not retrying.", message.job_id, exc)
        return None
    except (JobNotFoundError, JobConflictError) as exc:
        logger.warning("Job %s stopped due to status mismatch: %s. Not retrying.", message.job_id, exc)
        return None
    except StageTransientError as exc:
        # Class 2: transient error. Exponential backoff retry in Celery
        retries = getattr(task.request, "retries", 0) if task else 0
        max_retries = getattr(task, "max_retries", 3) if task else 3
        if retries < max_retries:
            countdown = int(min(60, (2**retries) * 5 + random.uniform(0, 2)))
            logger.warning(
                "Transient error processing job %s (attempt %d/%d). Retrying in %ds: %s",
                message.job_id,
                retries + 1,
                max_retries,
                countdown,
                exc,
            )
            if task is not None:
                raise task.retry(exc=exc, countdown=countdown)
            raise
        else:
            logger.error("Max retries exceeded on job %s: %s", message.job_id, exc)
            terminal_update = WorkerUpdate(
                schema_version=1,
                sequence=99,
                status=UpdateStatus.FAILED,
                occurred_at=datetime.now(UTC),
                trace_id=message.trace_id,
                payload=FailedPayload(
                    stage=getattr(exc, "stage", "speech"),
                    code=ErrorCode.PROVIDER_ERROR,
                    retryable=False,
                    attempts=message.analysis_attempt,
                    message="Transient retries exhausted.",
                ),
            )
            with contextlib.suppress(Exception):
                await backend_client.send_update(message.job_id, terminal_update)
            return terminal_update


@celery_app.task(
    name="app.worker.process_job",
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)
def process_job_task(self: Any, envelope: dict[str, Any]) -> None:
    """Celery task entry point consuming jobs from ai_jobs queue."""
    _run_async(_async_process_job_task(self, envelope))


__all__ = [
    "MonotonicUpdateTracker",
    "create_started_update",
    "get_or_create_backend_client",
    "get_or_create_pipeline",
    "process_job",
    "process_job_task",
    "set_default_backend_client",
    "set_default_pipeline",
]
