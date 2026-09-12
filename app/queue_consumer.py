"""Redis queue consumer worker loop for VirtuJudge AI-ML pipeline.

Listens for incoming analysis jobs from Redis, executes the pipeline,
reports status updates to Backend via HTTP, writes smoke correlation
results to Redis, and maintains a worker heartbeat.
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
from datetime import UTC, datetime

import redis.asyncio as aioredis
import ulid

from app.backend_client import BackendClient, BackendClientProtocol
from app.contracts import QueueMessage, WorkerUpdate
from app.pipeline import PitchAnalysisPipeline
from app.worker import process_job

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_NAMES: list[str] = ["virtujudge:local:jobs", "virtujudge:jobs"]
DEFAULT_HEARTBEAT_KEY: str = "virtujudge:local:worker:heartbeat"
DEFAULT_RESULT_PREFIX: str = "virtujudge:local:result:"
DEFAULT_RESULT_TTL_SECONDS: int = 3600
DEFAULT_HEARTBEAT_TTL_SECONDS: int = 15
DEFAULT_HEARTBEAT_INTERVAL_SECONDS: float = 5.0


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
        heartbeat_ttl: int = DEFAULT_HEARTBEAT_TTL_SECONDS,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        worker_id: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.backend_client = backend_client or BackendClient()
        self._redis = redis_client
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
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
        except Exception as err:
            logger.error("Failed to deserialize QueueMessage: %s", err)
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

        # Write smoke/diagnostic correlation result to Redis
        try:
            r = await self._get_redis()
            result_key = f"{self.result_prefix}{message.job_id}"
            await r.set(result_key, update.model_dump_json(), ex=self.result_ttl)
            logger.info("Saved result to Redis key: %s (status=%s)", result_key, update.status)
        except Exception as err:
            logger.warning("Failed to store result in Redis for job %s: %s", message.job_id, err)

        return update

    async def run(self, max_jobs: int | None = None) -> int:
        """Run the consumer polling loop until stopped or max_jobs processed."""
        r = await self._get_redis()
        self._stopping = False
        jobs_processed = 0

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
                    # BLPOP with 1-second timeout to allow responsive shutdown
                    item = await r.blpop(self.queue_names, timeout=1)
                except asyncio.CancelledError:
                    break
                except Exception as err:
                    if self._stopping:
                        break
                    logger.warning("Redis blpop error: %s. Retrying...", err)
                    await asyncio.sleep(1)
                    continue

                if item is None:
                    continue

                _queue_name, raw_data = item
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8")

                await self.process_one_message(raw_data)
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
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_HEARTBEAT_KEY",
    "DEFAULT_HEARTBEAT_TTL_SECONDS",
    "DEFAULT_QUEUE_NAMES",
    "DEFAULT_RESULT_PREFIX",
    "DEFAULT_RESULT_TTL_SECONDS",
    "RedisQueueConsumer",
]
