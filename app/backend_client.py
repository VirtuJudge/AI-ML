"""Backend HTTP callback client for VirtuJudge AI-ML worker."""

import asyncio
import logging
import os
from typing import Any, Protocol

import httpx

from app.contracts import UpdateStatus, WorkerUpdate

logger = logging.getLogger(__name__)


class BackendClientError(Exception):
    """Base exception for BackendClient errors."""


class BackendConfigurationError(BackendClientError):
    """Raised when required production backend configuration is missing."""


class BackendUnauthorizedError(BackendClientError):
    """Raised when backend rejects callback credentials with HTTP 401."""


class JobNotFoundError(BackendClientError):
    """Raised when job is not found on backend (HTTP 404)."""


class JobConflictError(BackendClientError):
    """Raised when an invalid state transition or conflict occurs (HTTP 409)."""


class BackendClientProtocol(Protocol):
    """Protocol defining methods for backend callback communication."""

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None: ...

    async def check_cancellation(self, job_id: str) -> bool: ...

    async def check_backend_reachability(self) -> bool: ...

    async def close(self) -> None: ...


class BackendClient:
    """Production HTTP client for reporting job status and checking cancellation."""

    def __init__(
        self,
        base_url: str | None = None,
        shared_secret: str | None = None,
        timeout: float = 10.0,
        strict_production: bool | None = None,
    ) -> None:
        env_name = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower()
        is_prod = env_name in ("production", "prod") or bool(os.getenv("SPACE_ID"))
        if strict_production is None:
            strict_production = is_prod

        raw_base_url = (
            base_url
            if base_url is not None
            else os.getenv("BACKEND_INTERNAL_URL")
        )
        secret = (
            shared_secret
            if shared_secret is not None
            else os.getenv("AI_WORKER_SHARED_SECRET")
        )

        if strict_production:
            if not raw_base_url:
                raise BackendConfigurationError(
                    "BACKEND_INTERNAL_URL is required in production environment."
                )
            if "localhost" in raw_base_url or "127.0.0.1" in raw_base_url:
                raise BackendConfigurationError(
                    f"BACKEND_INTERNAL_URL cannot use localhost/loopback in production: '{raw_base_url}'"
                )
            if not secret:
                raise BackendConfigurationError(
                    "AI_WORKER_SHARED_SECRET is required in production environment."
                )

        if not raw_base_url:
            raw_base_url = "http://localhost:3000"
        if secret is None:
            secret = ""

        clean_base = raw_base_url.rstrip("/")
        if clean_base.endswith("/internal/v1"):
            clean_base = clean_base[:-len("/internal/v1")].rstrip("/")
        self.base_url: str = clean_base
        self.shared_secret: str = secret
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {self.shared_secret}",
            "X-Worker-Secret": self.shared_secret,
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=self.headers,
        )

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None:
        """POST job status update to the internal backend endpoint."""
        url = f"/internal/v1/ai-jobs/{job_id}/updates"
        payload = update.model_dump(mode="json")
        # Ensure cancelled payload is empty dict per backend contract
        if update.status == UpdateStatus.CANCELLED or payload.get("status") == "cancelled":
            payload["payload"] = {}

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = await self.client.post(url, json=payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == max_attempts - 1:
                    logger.error("Failed to connect to backend callback endpoint %s: %s", url, exc)
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error(
                    "Worker credentials rejected by backend (401 Unauthorized) for job %s.",
                    job_id,
                )
                raise BackendUnauthorizedError(
                    f"Backend rejected worker credentials (HTTP 401) for job {job_id}."
                )
            elif response.status_code == 404:
                logger.warning(
                    "Job %s not found on backend (HTTP 404). Stopping processing.",
                    job_id,
                )
                raise JobNotFoundError(
                    f"Job {job_id} not found on backend (HTTP 404). Stopping processing."
                )
            elif response.status_code == 409:
                logger.warning(
                    "Conflict recording update for job %s (HTTP 409). Job may be cancelled or superseded.",
                    job_id,
                )
                raise JobConflictError(
                    f"Conflict recording update for job {job_id} (HTTP 409)."
                )
            elif response.status_code >= 500:
                if attempt == max_attempts - 1:
                    response.raise_for_status()
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            else:
                response.raise_for_status()
                return response.json()
        return None

    async def check_cancellation(self, job_id: str) -> bool:
        """Query the internal backend endpoint to check if job was cancelled."""
        url = f"/internal/v1/ai-jobs/{job_id}"
        try:
            response = await self.client.get(url)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

        if response.status_code == 401:
            raise BackendUnauthorizedError(
                f"Backend rejected credentials checking cancellation for job {job_id}."
            )
        if response.status_code == 404:
            raise JobNotFoundError(f"Job {job_id} not found on backend (HTTP 404).")
        if response.status_code != 200:
            return False

        data: dict[str, Any] = response.json()
        return bool(data.get("cancel_requested", False)) or data.get("status") == "cancelled"

    async def check_backend_reachability(self) -> bool:
        """Check if backend service is reachable."""
        try:
            response = await self.client.get("/health", timeout=3.0)
            return response.status_code in (200, 404)
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()


class FakeBackendClient:
    """In-memory backend client for tests."""

    def __init__(
        self,
        cancel_requested: bool = False,
        fail_with_status: int | None = None,
    ) -> None:
        self.updates: list[WorkerUpdate] = []
        self.cancel_requested = cancel_requested
        self.fail_with_status = fail_with_status
        self.reachable = True

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None:
        if self.fail_with_status == 401:
            raise BackendUnauthorizedError(f"Backend rejected credentials (401) for job {job_id}")
        if self.fail_with_status == 404:
            raise JobNotFoundError(f"Job {job_id} not found (404)")
        if self.fail_with_status == 409:
            raise JobConflictError(f"Conflict for job {job_id} (409)")
        if self.fail_with_status and self.fail_with_status >= 500:
            raise httpx.HTTPStatusError(
                f"Server error {self.fail_with_status}",
                request=httpx.Request("POST", f"http://test/internal/v1/ai-jobs/{job_id}/updates"),
                response=httpx.Response(self.fail_with_status),
            )
        self.updates.append(update)
        return {"status": "ok", "job_id": job_id, "cancel_requested": self.cancel_requested}

    async def check_cancellation(self, job_id: str) -> bool:
        if self.fail_with_status == 401:
            raise BackendUnauthorizedError(f"Backend rejected credentials (401) for job {job_id}")
        if self.fail_with_status == 404:
            raise JobNotFoundError(f"Job {job_id} not found (404)")
        return self.cancel_requested

    async def check_backend_reachability(self) -> bool:
        return self.reachable

    async def close(self) -> None:
        pass


__all__ = [
    "BackendClient",
    "BackendClientError",
    "BackendClientProtocol",
    "BackendConfigurationError",
    "BackendUnauthorizedError",
    "FakeBackendClient",
    "JobConflictError",
    "JobNotFoundError",
]
