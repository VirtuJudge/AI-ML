"""Worker dispatcher and Celery transport task for VirtuJudge AI-ML pipeline jobs."""

import asyncio
import atexit
import concurrent.futures
import logging
import os
import random
import re
import shutil
import threading
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from app.backend_client import (
    BackendClient,
    BackendClientProtocol,
    BackendUnauthorizedError,
    BackendUnavailableError,
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
ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)
MAX_PROVIDER_RETRIES = 3
MAX_CALLBACK_RETRIES = 3
MAX_TASK_RETRIES = MAX_PROVIDER_RETRIES + MAX_CALLBACK_RETRIES

_default_pipeline: PitchAnalysisPipeline | None = None
_default_backend_client: BackendClientProtocol | None = None
_worker_event_loop: asyncio.AbstractEventLoop | None = None
_worker_event_loop_lock = threading.Lock()


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


def _get_worker_event_loop() -> asyncio.AbstractEventLoop:
    """Create the worker's reusable event loop on first use."""
    global _worker_event_loop

    if _worker_event_loop is None or _worker_event_loop.is_closed():
        _worker_event_loop = asyncio.new_event_loop()
    return _worker_event_loop


def _run_on_worker_event_loop(coro: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine to completion while serializing access to the worker loop."""
    with _worker_event_loop_lock:
        return _get_worker_event_loop().run_until_complete(coro)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Execute every Celery coroutine on one persistent event loop.

    Async HTTP, database, and provider connection pools are bound to the loop where
    they are first used. Reusing one loop prevents cached clients from retaining
    transports that belonged to an already-closed ``asyncio.run`` loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_on_worker_event_loop(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_on_worker_event_loop, coro).result()


def _shutdown_worker_event_loop() -> None:
    """Close the reusable worker event loop during interpreter shutdown."""
    global _worker_event_loop

    with _worker_event_loop_lock:
        loop = _worker_event_loop
        if loop is None or loop.is_closed():
            return
        loop.close()
        _worker_event_loop = None


atexit.register(_shutdown_worker_event_loop)


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

    def __init__(
        self,
        backend_client: BackendClientProtocol,
        trace_id: str,
        initial_sequence: int = 0,
    ) -> None:
        self.backend_client = backend_client
        self.trace_id = trace_id
        self.sequence = initial_sequence

    @property
    def next_sequence(self) -> int:
        return self.sequence + 1

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None:
        self.sequence += 1
        update.sequence = self.sequence
        result = await self.backend_client.send_update(job_id, update)
        # A redelivered job may already have posted updates before its worker died.
        # Resume the backend's durable sequence instead of replaying 1, 2 forever.
        if isinstance(result, dict):
            self.sequence = max(self.sequence, int(result.get("last_update_sequence", 0)))
        return result

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

    async def get_last_update_sequence(self, job_id: str) -> int:
        return await self.backend_client.get_last_update_sequence(job_id)

    async def check_backend_reachability(self) -> bool:
        return await self.backend_client.check_backend_reachability()

    async def close(self) -> None:
        await self.backend_client.close()


def create_started_update(message: QueueMessage, pipeline_version: str = "1.0.0") -> WorkerUpdate:
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
    *,
    publish_terminal_update: bool = True,
) -> WorkerUpdate:
    """Process an incoming QueueMessage and return the terminal WorkerUpdate."""
    tracker: MonotonicUpdateTracker | None = None
    client_to_use: BackendClientProtocol | None = backend_client
    cancelled_before_dispatch = False

    if backend_client is not None:
        if message.job_type != JobType.ERASE_AI_DATA:
            cancelled_before_dispatch = await backend_client.check_cancellation(message.job_id)
        last_sequence = await backend_client.get_last_update_sequence(message.job_id)
        tracker = MonotonicUpdateTracker(
            backend_client,
            trace_id=message.trace_id,
            initial_sequence=last_sequence,
        )
        client_to_use = tracker
        started_update = create_started_update(message)
        try:
            if not cancelled_before_dispatch:
                await tracker.send_update(message.job_id, started_update)
        except BackendUnauthorizedError:
            logger.error("Fatal worker authorization failure sending started update.")
            raise
        except (JobNotFoundError, JobConflictError):
            logger.warning("Job %s rejected at start", message.job_id)
            raise

    terminal_update: WorkerUpdate
    try:
        if cancelled_before_dispatch:
            raise JobCancelledError(job_id=message.job_id, stage="dispatch")
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
        if message.job_type not in {
            JobType.ANALYZE_SESSION,
            JobType.ANALYZE_ANSWER,
            JobType.GENERATE_REPORT,
            JobType.ERASE_AI_DATA,
        }:
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
        else:
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
    except (BackendUnavailableError, BackendUnauthorizedError, JobNotFoundError, JobConflictError):
        raise
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

    if client_to_use is not None and publish_terminal_update:
        await client_to_use.send_update(message.job_id, terminal_update)

    return terminal_update


async def _async_process_job_task(
    task: Any,
    envelope: dict[str, Any],
    pipeline: PitchAnalysisPipeline | None = None,
    backend_client: BackendClientProtocol | None = None,
    delivery_update: dict[str, Any] | None = None,
) -> WorkerUpdate | None:
    """Inner coroutine executing one Celery job task."""
    if backend_client is None:
        backend_client = get_or_create_backend_client()

    retries = getattr(getattr(task, "request", None), "retries", 0)
    max_retries = getattr(task, "max_retries", 3) if task is not None else 0

    def retry_or_quarantine(
        *,
        job_id: str,
        reason: str,
        pending_update: dict[str, Any] | None,
        retry_args: list[Any],
        retry_kwargs: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        if retryable and task is not None and retries < max_retries:
            countdown = int(min(60, (2**retries) * 5 + random.uniform(0, 2)))
            logger.warning(
                "Backend callback for job %s will retry (attempt %d/%d, reason=%s).",
                job_id,
                retries + 1,
                max_retries,
                reason,
            )
            raise task.retry(
                args=retry_args,
                kwargs=retry_kwargs,
                countdown=countdown,
            )

        sender = getattr(task, "app", celery_app) if task is not None else celery_app
        record = {
            "job_id": job_id,
            "reason": reason,
            "queue_message": envelope,
            "terminal_update": pending_update,
        }
        sender.send_task(
            "app.worker.quarantine_callback_delivery",
            args=[record],
            queue=os.getenv("AI_QUARANTINE_QUEUE", "ai_jobs_quarantine"),
            ignore_result=True,
        )
        logger.error("Callback for job %s was quarantined (reason=%s).", job_id, reason)

    if delivery_update is not None:
        job_id = str(envelope.get("job_id", ""))
        pending_update = WorkerUpdate.model_validate(delivery_update)
        try:
            await backend_client.send_update(job_id, pending_update)
        except BackendUnauthorizedError:
            retry_or_quarantine(
                job_id=job_id,
                reason="backend_unauthorized",
                pending_update=delivery_update,
                retry_args=[envelope],
                retryable=False,
            )
            return pending_update
        except JobNotFoundError:
            retry_or_quarantine(
                job_id=job_id,
                reason="job_not_found",
                pending_update=delivery_update,
                retry_args=[envelope],
                retryable=False,
            )
            return pending_update
        except Exception as exc:
            reason = "job_conflict" if isinstance(exc, JobConflictError) else "callback_unavailable"
            retry_or_quarantine(
                job_id=job_id,
                reason=reason,
                pending_update=delivery_update,
                retry_args=[envelope],
                retry_kwargs={"delivery_update": delivery_update},
            )
            return pending_update
        return pending_update

    # 1. Validate envelope model
    try:
        message = QueueMessage.model_validate(envelope)
    except ValidationError as val_exc:
        code, safe_msg = _map_validation_error(val_exc)
        raw_job_id = envelope.get("job_id")
        if not isinstance(raw_job_id, str) or not ULID_PATTERN.fullmatch(raw_job_id):
            retry_or_quarantine(
                job_id="unknown",
                reason="malformed_queue_message",
                pending_update=None,
                retry_args=[envelope],
                retryable=False,
            )
            return None

        raw_trace_id = envelope.get("trace_id")
        trace_id = (
            raw_trace_id
            if isinstance(raw_trace_id, str) and ULID_PATTERN.fullmatch(raw_trace_id)
            else raw_job_id
        )
        raw_attempt = envelope.get("analysis_attempt", 1)
        attempt = (
            raw_attempt
            if isinstance(raw_attempt, int)
            and not isinstance(raw_attempt, bool)
            and raw_attempt >= 0
            else 1
        )
        try:
            sequence = await backend_client.get_last_update_sequence(raw_job_id)
        except BackendUnauthorizedError:
            retry_or_quarantine(
                job_id=raw_job_id,
                reason="backend_unauthorized",
                pending_update=None,
                retry_args=[envelope],
                retryable=False,
            )
            return None
        except JobNotFoundError:
            retry_or_quarantine(
                job_id=raw_job_id,
                reason="job_not_found",
                pending_update=None,
                retry_args=[envelope],
                retryable=False,
            )
            return None
        except BackendUnavailableError:
            retry_or_quarantine(
                job_id=raw_job_id,
                reason="backend_state_unavailable",
                pending_update=None,
                retry_args=[envelope],
            )
            return None

        update = WorkerUpdate(
            schema_version=1,
            sequence=sequence + 1,
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
        serialized_update = update.model_dump(mode="json")
        try:
            await backend_client.send_update(raw_job_id, update)
        except BackendUnauthorizedError:
            retry_or_quarantine(
                job_id=raw_job_id,
                reason="backend_unauthorized",
                pending_update=serialized_update,
                retry_args=[envelope],
                retry_kwargs={"delivery_update": serialized_update},
                retryable=False,
            )
        except JobNotFoundError:
            retry_or_quarantine(
                job_id=raw_job_id,
                reason="job_not_found",
                pending_update=serialized_update,
                retry_args=[envelope],
                retry_kwargs={"delivery_update": serialized_update},
                retryable=False,
            )
        except Exception as exc:
            reason = "job_conflict" if isinstance(exc, JobConflictError) else "callback_unavailable"
            retry_or_quarantine(
                job_id=raw_job_id,
                reason=reason,
                pending_update=serialized_update,
                retry_args=[envelope],
                retry_kwargs={"delivery_update": serialized_update},
            )
        return None
    except Exception as exc:
        logger.error("Non-retryable envelope parsing error (error_type=%s).", type(exc).__name__)
        return None

    # 2. Dispatch job. The terminal callback is delivered separately so a retry
    # can resend the same result without running the pipeline again.
    try:
        if pipeline is None:
            pipeline = await get_or_create_pipeline()
        update = await process_job(
            message,
            pipeline,
            backend_client,
            publish_terminal_update=False,
        )
        if (
            update is not None
            and update.status == UpdateStatus.FAILED
            and isinstance(update.payload, FailedPayload)
            and update.payload.retryable
            and task is not None
            and retries < MAX_PROVIDER_RETRIES
        ):
            countdown = int(min(60, (2**retries) * 5 + random.uniform(0, 2)))
            logger.warning(
                "Transient error on job %s (attempt %d/%d). Retrying in %ds.",
                message.job_id,
                retries + 1,
                max_retries,
                countdown,
            )
            raise task.retry(countdown=countdown)
        if update is None:
            return None

        if (
            update.status == UpdateStatus.FAILED
            and isinstance(update.payload, FailedPayload)
            and update.payload.retryable
        ):
            update = update.model_copy(
                update={
                    "payload": update.payload.model_copy(
                        update={
                            "retryable": False,
                            "message": "Transient retries exhausted.",
                        }
                    )
                }
            )

        try:
            await backend_client.send_update(message.job_id, update)
        except BackendUnauthorizedError:
            retry_or_quarantine(
                job_id=message.job_id,
                reason="backend_unauthorized",
                pending_update=update.model_dump(mode="json"),
                retry_args=[envelope],
                retry_kwargs={"delivery_update": update.model_dump(mode="json")},
                retryable=False,
            )
        except JobNotFoundError:
            retry_or_quarantine(
                job_id=message.job_id,
                reason="job_not_found",
                pending_update=update.model_dump(mode="json"),
                retry_args=[envelope],
                retry_kwargs={"delivery_update": update.model_dump(mode="json")},
                retryable=False,
            )
        except Exception as exc:
            reason = "job_conflict" if isinstance(exc, JobConflictError) else "callback_unavailable"
            retry_or_quarantine(
                job_id=message.job_id,
                reason=reason,
                pending_update=update.model_dump(mode="json"),
                retry_args=[envelope],
                retry_kwargs={"delivery_update": update.model_dump(mode="json")},
            )
        return update
    except BackendUnauthorizedError:
        retry_or_quarantine(
            job_id=message.job_id,
            reason="backend_unauthorized",
            pending_update=None,
            retry_args=[envelope],
            retryable=False,
        )
        return None
    except (JobNotFoundError, JobConflictError):
        retry_or_quarantine(
            job_id=message.job_id,
            reason="job_status_conflict",
            pending_update=None,
            retry_args=[envelope],
            retryable=False,
        )
        return None
    except BackendUnavailableError:
        retry_or_quarantine(
            job_id=message.job_id,
            reason="backend_state_unavailable",
            pending_update=None,
            retry_args=[envelope],
        )
        return None


@celery_app.task(
    name="app.worker.process_job",
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=MAX_TASK_RETRIES,
)
def process_job_task(
    self: Any,
    envelope: dict[str, Any],
    delivery_update: dict[str, Any] | None = None,
) -> None:
    """Celery task entry point consuming jobs from ai_jobs queue."""
    _run_async(_async_process_job_task(self, envelope, delivery_update=delivery_update))


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
