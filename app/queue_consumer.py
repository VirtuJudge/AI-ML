"""Redis queue consumer worker loop for VirtuJudge AI-ML pipeline.

Listens for incoming analysis jobs from Redis, executes the pipeline,
reports status updates to Backend via HTTP, writes smoke correlation
results to Redis, and maintains a worker heartbeat.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
import ulid

from app.backend_client import (
    BackendClient,
    BackendClientProtocol,
    JobConflictError,
    JobNotFoundError,
)
from app.contracts import ErrorCode, FailedPayload, QueueMessage, UpdateStatus, WorkerUpdate
from app.pipeline import PitchAnalysisPipeline
from app.worker import process_job

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_NAMES: list[str] = ["virtujudge:local:jobs", "virtujudge:jobs"]
DEFAULT_HEARTBEAT_KEY: str = "virtujudge:local:worker:heartbeat"
DEFAULT_RESULT_PREFIX: str = "virtujudge:local:result:"
DEFAULT_RESULT_TTL_SECONDS: int = 3600
DEFAULT_HEARTBEAT_TTL_SECONDS: int = 15
DEFAULT_HEARTBEAT_INTERVAL_SECONDS: float = 5.0
DEFAULT_DEAD_LETTER_LIMIT: int = 1000
DEFAULT_DEAD_LETTER_TTL_SECONDS: int = 7 * 24 * 60 * 60
ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)


class RedisQueueConsumer:
    """Async Redis queue consumer executing pipeline jobs."""

    def __init__(
        self,
        pipeline: PitchAnalysisPipeline,
        backend_client: BackendClientProtocol | None = None,
        redis_client: aioredis.Redis | None = None,
        redis_url: str | None = None,
        queue_names: list[str] | None = None,
        heartbeat_key: str = DEFAULT_HEARTBEAT_KEY,
        result_prefix: str = DEFAULT_RESULT_PREFIX,
        result_ttl: int = DEFAULT_RESULT_TTL_SECONDS,
        dead_letter_limit: int = DEFAULT_DEAD_LETTER_LIMIT,
        dead_letter_ttl: int = DEFAULT_DEAD_LETTER_TTL_SECONDS,
        heartbeat_ttl: int = DEFAULT_HEARTBEAT_TTL_SECONDS,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        worker_id: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.backend_client = backend_client or BackendClient()
        self._redis = redis_client
        self._redis_url = redis_url or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        self._owns_redis = redis_client is None

        configured_queue = os.getenv("AI_QUEUE_NAME")
        if queue_names:
            self.queue_names = list(queue_names)
        elif configured_queue:
            self.queue_names = [configured_queue]
            if configured_queue != "virtujudge:jobs":
                self.queue_names.append("virtujudge:jobs")
        else:
            self.queue_names = list(DEFAULT_QUEUE_NAMES)

        self.heartbeat_key = heartbeat_key
        self.result_prefix = result_prefix
        self.result_ttl = result_ttl
        self.dead_letter_limit = dead_letter_limit
        self.dead_letter_ttl = dead_letter_ttl
        self.heartbeat_ttl = heartbeat_ttl
        self.heartbeat_interval = heartbeat_interval
        self.worker_id = worker_id or f"worker-{ulid.new().str[:8]}"

        self._stopping = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def _get_redis(self) -> aioredis.Redis:
        """Get or lazily initialize the Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                health_check_interval=300,  # Ping every 5 minutes to keep socket warm
                socket_keepalive=True,
            )
        return self._redis

    async def _heartbeat_loop(self) -> None:
        """Periodically emit worker heartbeat to Redis."""
        while not self._stopping:
            try:
                r = await self._get_redis()
                payload = {
                    "status": "alive",
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "queues": self.queue_names,
                }
                await r.set(self.heartbeat_key, json.dumps(payload), ex=self.heartbeat_ttl)
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.warning("Heartbeat update failed: %s", err)

            try:
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break

    async def process_one_message(self, raw_message: str) -> WorkerUpdate | None:
        """Parse and execute a single message."""
        try:
            message = QueueMessage.model_validate_json(raw_message)
        except Exception as exc:
            await self._handle_malformed_message(raw_message, exc)
            return None

        logger.info(
            "Processing job %s (type=%s, attempt=%d, worker=%s)",
            message.job_id,
            message.job_type,
            message.analysis_attempt,
            self.worker_id,
        )

        update = await process_job(
            message,
            self.pipeline,
            self.backend_client,
        )

        # Write safe correlation result metadata to Redis
        # (redacting confidential content and payloads)
        try:
            r = await self._get_redis()
            result_key = f"{self.result_prefix}{message.job_id}"
            safe_payload: dict[str, Any] = {}
            if hasattr(update.payload, "analysis_artifact") and update.payload.analysis_artifact:
                safe_payload["artifact_id"] = update.payload.analysis_artifact.artifact_id
            safe_update = WorkerUpdate(
                schema_version=update.schema_version,
                sequence=update.sequence,
                status=update.status,
                occurred_at=update.occurred_at,
                trace_id=update.trace_id,
                payload=safe_payload,
            )
            await r.set(result_key, safe_update.model_dump_json(), ex=self.result_ttl)
            logger.info("Saved result to Redis key: %s (status=%s)", result_key, update.status)
        except Exception as exc:
            logger.warning(
                "Failed to store result for job %s (error_type=%s).",
                message.job_id,
                type(exc).__name__,
            )

        return update

    async def _handle_malformed_message(self, raw_message: str, error: Exception) -> None:
        """Report invalid work safely or retain a bounded dead-letter record."""
        digest = hashlib.sha256(raw_message.encode("utf-8", errors="replace")).hexdigest()
        raw_envelope: Any = None
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            raw_envelope = json.loads(raw_message)

        job_id = raw_envelope.get("job_id") if isinstance(raw_envelope, dict) else None
        trace_id = raw_envelope.get("trace_id") if isinstance(raw_envelope, dict) else None
        if isinstance(job_id, str) and ULID_PATTERN.fullmatch(job_id):
            safe_trace_id = (
                trace_id
                if isinstance(trace_id, str) and ULID_PATTERN.fullmatch(trace_id)
                else job_id
            )
            attempt = raw_envelope.get("analysis_attempt", 1)
            if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
                attempt = 1
            try:
                sequence = await self.backend_client.get_last_update_sequence(job_id)
                update = WorkerUpdate(
                    schema_version=1,
                    sequence=sequence + 1,
                    status=UpdateStatus.FAILED,
                    occurred_at=datetime.now(UTC),
                    trace_id=safe_trace_id,
                    payload=FailedPayload(
                        stage="queue",
                        code=ErrorCode.MALFORMED_PAYLOAD,
                        retryable=False,
                        attempts=attempt,
                        message="The queued job did not match the required message schema.",
                    ),
                )
                await self.backend_client.send_update(job_id, update)
                logger.warning(
                    "Malformed work item reported to backend (job=%s, digest=%s, error_type=%s).",
                    job_id,
                    digest,
                    type(error).__name__,
                )
                return None
            except (JobConflictError, JobNotFoundError) as backend_error:
                error = backend_error
            except Exception:
                # Keep the item pending until the backend has accepted the failure.
                raise

        record = {
            "digest": digest,
            "byte_length": len(raw_message.encode("utf-8", errors="replace")),
            "job_id": job_id
            if isinstance(job_id, str) and ULID_PATTERN.fullmatch(job_id)
            else None,
            "trace_id": trace_id
            if isinstance(trace_id, str) and ULID_PATTERN.fullmatch(trace_id)
            else None,
            "envelope_fields": sorted(raw_envelope) if isinstance(raw_envelope, dict) else [],
            "error_type": type(error).__name__,
            "failure_code": ErrorCode.MALFORMED_PAYLOAD.value,
        }
        redis = await self._get_redis()
        dead_letter_key = f"{self.queue_names[0]}:dead-letter"
        await redis.lpush(dead_letter_key, json.dumps(record, separators=(",", ":")))
        await redis.ltrim(dead_letter_key, 0, max(1, self.dead_letter_limit) - 1)
        await redis.expire(dead_letter_key, self.dead_letter_ttl)
        logger.error("Malformed work item retained in dead-letter queue (digest=%s).", digest)
        return None

    @staticmethod
    def _processing_queue_name(queue_name: str) -> str:
        return f"{queue_name}:processing"

    async def _recover_pending_messages(self, redis: aioredis.Redis) -> int:
        """Move messages left pending by an interrupted consumer back to ready queues."""
        recovered = 0
        for queue_name in self.queue_names:
            processing_queue = self._processing_queue_name(queue_name)
            while True:
                raw_message = await redis.rpoplpush(processing_queue, queue_name)
                if raw_message is None:
                    break
                recovered += 1
        if recovered:
            logger.warning("Recovered %d pending Redis job message(s).", recovered)
        return recovered

    async def run(self, max_jobs: int | None = None) -> int:
        """Run the consumer polling loop until stopped or max_jobs processed."""
        r = await self._get_redis()
        self._stopping = False
        jobs_processed = 0
        await self._recover_pending_messages(r)

        # Start background heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "AI Worker %s started. Listening on queues: %s",
            self.worker_id,
            ", ".join(self.queue_names),
        )

        try:
            while not self._stopping:
                if max_jobs is not None and jobs_processed >= max_jobs:
                    break

                try:
                    selected_queue: str | None = None
                    raw_data: str | bytes | None = None
                    for queue_name in self.queue_names:
                        processing_queue = self._processing_queue_name(queue_name)
                        raw_data = await r.blmove(
                            queue_name,
                            processing_queue,
                            timeout=1,
                            src="LEFT",
                            dest="RIGHT",
                        )
                        if raw_data is not None:
                            selected_queue = queue_name
                            break
                except asyncio.CancelledError:
                    break
                except Exception as err:
                    if self._stopping:
                        break
                    err_str = str(err).lower()
                    if "timeout reading" in err_str:
                        logger.debug("Redis idle socket timeout. Re-polling...")
                        await asyncio.sleep(0.5)
                        continue
                    backoff = 10 if "allowlist" in err_str else 1
                    logger.warning(
                        "Redis claim failed (error_type=%s). Retrying in %ds...",
                        type(err).__name__,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                if raw_data is None or selected_queue is None:
                    continue

                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8", errors="replace")

                processing_queue = self._processing_queue_name(selected_queue)
                try:
                    await self.process_one_message(raw_data)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # BLMOVE claims into the right side of the pending list;
                    # LMOVE atomically returns the failed item to the ready head.
                    await r.lmove(
                        processing_queue,
                        selected_queue,
                        src="RIGHT",
                        dest="LEFT",
                    )
                    logger.warning(
                        "Redis job remains queued after processing failure (error_type=%s).",
                        type(exc).__name__,
                    )
                    await asyncio.sleep(1)
                    continue

                await r.lrem(processing_queue, 1, raw_data)
                jobs_processed += 1

        finally:
            await self.shutdown()

        return jobs_processed

    async def shutdown(self) -> None:
        """Gracefully stop consumer loop, heartbeat, and close connections."""
        self._stopping = True

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        if self._owns_redis and self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.aclose()
            self._redis = None

        if self.backend_client is not None:
            with contextlib.suppress(Exception):
                await self.backend_client.close()

        logger.info("AI Worker %s stopped gracefully.", self.worker_id)

    def attach_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))


__all__ = [
    "DEFAULT_DEAD_LETTER_LIMIT",
    "DEFAULT_DEAD_LETTER_TTL_SECONDS",
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_HEARTBEAT_KEY",
    "DEFAULT_HEARTBEAT_TTL_SECONDS",
    "DEFAULT_QUEUE_NAMES",
    "DEFAULT_RESULT_PREFIX",
    "DEFAULT_RESULT_TTL_SECONDS",
    "RedisQueueConsumer",
]
