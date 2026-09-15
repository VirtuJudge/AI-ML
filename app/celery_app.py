"""Celery application configuration for VirtuJudge AI-ML worker."""

import os
from typing import Any

from celery import Celery

DEFAULT_QUEUE_NAME: str = "ai_jobs"
DEFAULT_TASK_NAME: str = "app.worker.process_job"


def create_celery_app(
    broker_url: str | None = None,
    app_name: str = "virtujudge_ai_ml",
) -> Celery:
    """Create and configure Celery application for the AI-ML worker."""
    effective_broker = (
        broker_url
        if broker_url is not None
        else os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    app = Celery(app_name, broker=effective_broker)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_default_queue=DEFAULT_QUEUE_NAME,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        broker_connection_retry_on_startup=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        include=["app.worker"],
    )
    return app


celery_app: Celery = create_celery_app()

__all__ = [
    "DEFAULT_QUEUE_NAME",
    "DEFAULT_TASK_NAME",
    "celery_app",
    "create_celery_app",
]
