"""Readiness and health check probes for VirtuJudge AI-ML worker."""

import logging
import os
from typing import Any

from app.backend_client import BackendClient, BackendClientProtocol

logger = logging.getLogger(__name__)

DEFAULT_TASK_NAME: str = os.getenv("AI_TASK_NAME", "app.worker.process_job")
DEFAULT_QUEUE_NAME: str = os.getenv("AI_QUEUE_NAME", "ai_jobs")


async def check_readiness(
    backend_client: BackendClientProtocol | None = None,
    redis_url: str | None = None,
    celery_app: Any | None = None,
) -> dict[str, Any]:
    """Perform deep readiness check verifying all external dependencies and registered task."""
    queue_name = os.getenv("AI_QUEUE_NAME", DEFAULT_QUEUE_NAME)
    task_name = os.getenv("AI_TASK_NAME", DEFAULT_TASK_NAME)
    provider_mode = os.getenv("AI_PROVIDER_MODE", "fake")

    # 1. Broker connectivity
    broker_url = redis_url or os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    broker_connected = False
    try:
        import redis.asyncio as aioredis

        # Mask credentials in any logging
        r = aioredis.from_url(broker_url, socket_connect_timeout=2.0)
        await r.ping()
        await r.aclose()
        broker_connected = True
    except Exception as exc:
        logger.debug("Broker readiness check failed: %s", exc)
        broker_connected = False

    # 2. Celery task registration
    task_registered = False
    try:
        app_to_check = celery_app
        if app_to_check is None:
            from app.celery_app import celery_app as default_app

            app_to_check = default_app

        if hasattr(app_to_check, "loader") and hasattr(app_to_check.loader, "import_default_modules"):
            app_to_check.loader.import_default_modules()
        else:
            import app.worker  # noqa: F401

        task_registered = task_name in app_to_check.tasks
    except Exception as exc:
        logger.debug("Task registration check failed: %s", exc)
        task_registered = False

    # 3. Backend reachability
    client = backend_client
    owns_client = False
    if client is None:
        client = BackendClient(strict_production=False)
        owns_client = True

    try:
        backend_reachable = await client.check_backend_reachability()
    except Exception as exc:
        logger.debug("Backend reachability check failed: %s", exc)
        backend_reachable = False
    finally:
        if owns_client:
            await client.close()

    # 4. Callback credentials configured
    shared_secret = os.getenv("AI_WORKER_SHARED_SECRET")
    callback_credentials_configured = bool(shared_secret and shared_secret.strip())

    # 5. Object storage configured
    bucket = os.getenv("OBJECT_STORAGE_BUCKET")
    endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT")
    object_storage_configured = bool(bucket or endpoint)

    # In dev/test with fake provider, relax broker/backend requirements for readiness if not configured
    is_prod = os.getenv("APP_ENV") in ("production", "prod") or bool(os.getenv("SPACE_ID"))
    if is_prod:
        is_ready = bool(
            broker_connected
            and task_registered
            and backend_reachable
            and callback_credentials_configured
            and object_storage_configured
        )
    else:
        is_ready = bool(task_registered and callback_credentials_configured)

    return {
        "status": "ready" if is_ready else "unready",
        "broker_connected": broker_connected,
        "queue": queue_name,
        "task_registered": task_registered,
        "backend_reachable": backend_reachable,
        "callback_credentials_configured": callback_credentials_configured,
        "object_storage_configured": object_storage_configured,
        "provider_mode": provider_mode,
    }


__all__ = [
    "DEFAULT_QUEUE_NAME",
    "DEFAULT_TASK_NAME",
    "check_readiness",
]
