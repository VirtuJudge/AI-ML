"""consumer.py: Background consumer pulling jobs from Redis 24/7."""

import asyncio
import logging
import os

import app.compat  # noqa: F401
from app.__main__ import _load_env, build_pipeline
from app.backend_client import BackendClient
from app.queue_consumer import RedisQueueConsumer

logger = logging.getLogger("virtujudge.consumer")


async def run_redis_consumer() -> None:
    """Initialize live pipeline and run Redis queue consumer loop."""
    _load_env()
    redis_url = os.getenv("REDIS_URL")
    while not redis_url:
        logger.warning("REDIS_URL is not configured. Queue consumer standing by...")
        await asyncio.sleep(15)
        redis_url = os.getenv("REDIS_URL")

    logger.info("Initializing Live AI Pipeline (Groq, MediaPipe, Librosa, Supabase pgvector, Cloudflare R2)...")
    pipeline = await build_pipeline()
    backend_client = BackendClient()

    consumer = RedisQueueConsumer(
        pipeline=pipeline,
        backend_client=backend_client,
        redis_url=redis_url,
    )
    consumer.attach_signal_handlers()

    logger.info(
        "AI Worker is LIVE. Listening to Redis queues: %s (Worker ID: %s)",
        consumer.queue_names,
        consumer.worker_id,
    )
    await consumer.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_redis_consumer())
