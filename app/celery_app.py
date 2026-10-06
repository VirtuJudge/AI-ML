"""Celery application configuration for VirtuJudge AI-ML worker."""

import os
import ssl
from typing import Any
from urllib.parse import parse_qs, urlparse

from celery import Celery

DEFAULT_QUEUE_NAME: str = os.getenv("AI_QUEUE_NAME", "ai_jobs")
DEFAULT_TASK_NAME: str = os.getenv("AI_TASK_NAME", "app.worker.process_job")


def create_celery_app(
    broker_url: str | None = None,
    app_name: str = "virtujudge_ai_ml",
) -> Celery:
    """Create and configure Celery application for the AI-ML worker."""
    effective_broker = (
        broker_url
        if broker_url is not None
        else (
            os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        )
    )
    queue_name = os.getenv("AI_QUEUE_NAME", DEFAULT_QUEUE_NAME)
    prefetch = int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1"))
    broker_use_ssl: dict[str, Any] | None = None
    if effective_broker.startswith("rediss://"):
        configured_cert_reqs = os.getenv("REDIS_TLS_CERT_REQS", "required").strip().lower()
        url_cert_reqs = parse_qs(urlparse(effective_broker).query).get("ssl_cert_reqs", [])
        if configured_cert_reqs != "required" or (
            any(
                value.strip().lower() not in {"required", "cert_required", "2"}
                for value in url_cert_reqs
            )
        ):
            raise ValueError("Redis TLS certificate verification cannot be disabled.")

        certificate_file = os.getenv("REDIS_SSL_CERTFILE")
        key_file = os.getenv("REDIS_SSL_KEYFILE")
        if bool(certificate_file) != bool(key_file):
            raise ValueError("REDIS_SSL_CERTFILE and REDIS_SSL_KEYFILE must be set together.")

        broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
        ca_bundle = os.getenv("REDIS_SSL_CA_CERTS")
        if ca_bundle:
            broker_use_ssl["ssl_ca_certs"] = ca_bundle
        if certificate_file and key_file:
            broker_use_ssl["ssl_certfile"] = certificate_file
            broker_use_ssl["ssl_keyfile"] = key_file

    app = Celery(app_name, broker=effective_broker)
    conf: dict[str, Any] = {
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        "task_default_queue": queue_name,
        "task_ignore_result": True,
        "task_store_errors_even_if_ignored": False,
        "broker_connection_retry_on_startup": True,
        "worker_prefetch_multiplier": prefetch,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "include": ["app.worker"],
    }
    if broker_use_ssl:
        conf["broker_use_ssl"] = broker_use_ssl
        conf["redis_backend_use_ssl"] = broker_use_ssl

    app.conf.update(**conf)
    return app


celery_app: Celery = create_celery_app()

__all__ = [
    "DEFAULT_QUEUE_NAME",
    "DEFAULT_TASK_NAME",
    "celery_app",
    "create_celery_app",
]
