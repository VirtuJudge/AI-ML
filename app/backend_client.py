"""Backend HTTP callback client for VirtuJudge AI-ML worker."""

import asyncio
import logging
import os
from typing import Any, Protocol, cast

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


class BackendUnavailableError(BackendClientError):
    """Raised when canonical backend state cannot be read or updated safely."""


JOB_STATUS_VALUES = frozenset(
    {"pending", "queued", "running", "completed", "failed", "cancelled", "superseded"}
)


class BackendClientProtocol(Protocol):
    """Protocol defining methods for backend callback communication."""

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None: ...

    async def check_cancellation(self, job_id: str) -> bool: ...

    async def get_last_update_sequence(self, job_id: str) -> int: ...

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

        raw_base_url = base_url if base_url is not None else os.getenv("BACKEND_INTERNAL_URL")
        secret = (
            shared_secret if shared_secret is not None else os.getenv("AI_WORKER_SHARED_SECRET")
        )

        if strict_production:
            if not raw_base_url:
                raise BackendConfigurationError(
                    "BACKEND_INTERNAL_URL is required in production environment."
                )
            if "localhost" in raw_base_url or "127.0.0.1" in raw_base_url:
                raise BackendConfigurationError(
                    "BACKEND_INTERNAL_URL cannot use localhost/loopback in production: "
                    f"'{raw_base_url}'"
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
            clean_base = clean_base[: -len("/internal/v1")].rstrip("/")
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
                    logger.error(
                        "Failed to connect to backend callback endpoint for job %s.", job_id
                    )
                    raise BackendUnavailableError(
                        f"Backend callback unavailable for job {job_id}."
                    ) from exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code == 200:
                return cast(dict[str, Any], response.json())
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
                if await self._should_retry_terminal_conflict(job_id, update):
                    continue
                logger.warning(
                    "Conflict recording update for job %s (HTTP 409). "
                    "Job may be cancelled or superseded.",
                    job_id,
                )
                raise JobConflictError(f"Conflict recording update for job {job_id} (HTTP 409).")
            elif response.status_code >= 500:
                if attempt == max_attempts - 1:
                    raise BackendUnavailableError(
                        f"Backend callback unavailable for job {job_id} "
                        f"(HTTP {response.status_code})."
                    )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            else:
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
        return None

    async def _get_job_status(self, job_id: str) -> dict[str, Any]:
        """Fetch canonical job status with bounded retries and fail closed."""
        url = f"/internal/v1/ai-jobs/{job_id}"
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == max_attempts - 1:
                    raise BackendUnavailableError(
                        f"Backend job status unavailable for job {job_id}."
                    ) from exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code == 401:
                raise BackendUnauthorizedError(
                    f"Backend rejected credentials checking job {job_id}."
                )
            if response.status_code == 404:
                raise JobNotFoundError(f"Job {job_id} not found on backend (HTTP 404).")
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise BackendUnavailableError(
                        f"Backend returned invalid job status for job {job_id}."
                    ) from exc
                if not isinstance(data, dict):
                    raise BackendUnavailableError(
                        f"Backend returned invalid job status for job {job_id}."
                    )
                status = data.get("status")
                cancel_requested = data.get("cancel_requested")
                if (
                    not isinstance(status, str)
                    or status not in JOB_STATUS_VALUES
                    or not isinstance(cancel_requested, bool)
                ):
                    raise BackendUnavailableError(
                        f"Backend returned invalid job status for job {job_id}."
                    )
                return cast(dict[str, Any], data)

            if attempt == max_attempts - 1:
                raise BackendUnavailableError(
                    f"Backend job status unavailable for job {job_id} "
                    f"(HTTP {response.status_code})."
                )
            await asyncio.sleep(0.5 * (2**attempt))

        raise BackendUnavailableError(f"Backend job status unavailable for job {job_id}.")

    async def get_last_update_sequence(self, job_id: str) -> int:
        """Read the backend's accepted sequence before starting or retrying a job."""
        data = await self._get_job_status(job_id)
        sequence = data.get("last_update_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise BackendUnavailableError(
                f"Backend returned invalid update sequence for job {job_id}."
            )
        return sequence

    async def _should_retry_terminal_conflict(self, job_id: str, update: WorkerUpdate) -> bool:
        if update.status not in {
            UpdateStatus.COMPLETED,
            UpdateStatus.FAILED,
            UpdateStatus.CANCELLED,
        }:
            return False

        data = await self._get_job_status(job_id)
        return (
            data.get("status") in {"pending", "queued", "running"}
            and not data.get("cancel_requested", False)
            and int(data.get("last_update_sequence", 0)) < update.sequence
        )

    async def check_cancellation(self, job_id: str) -> bool:
        """Query canonical cancellation state; unavailable state raises safely."""
        data = await self._get_job_status(job_id)
        return data["cancel_requested"] or data["status"] in {"cancelled", "superseded"}

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
        self._last_update_sequence: dict[str, int] = {}

    async def send_update(self, job_id: str, update: WorkerUpdate) -> dict[str, Any] | None:
        if self.fail_with_status == 401:
            raise BackendUnauthorizedError(f"Backend rejected credentials (401) for job {job_id}")
        if self.fail_with_status == 404:
            raise JobNotFoundError(f"Job {job_id} not found (404)")
        if self.fail_with_status == 409:
            raise JobConflictError(f"Conflict for job {job_id} (409)")
        if self.fail_with_status and self.fail_with_status >= 500:
            raise BackendUnavailableError(
                f"Backend callback unavailable (HTTP {self.fail_with_status}) for job {job_id}."
            )
        self.updates.append(update)
        self._last_update_sequence[job_id] = max(
            self._last_update_sequence.get(job_id, 0), update.sequence
        )
        return {"status": "ok", "job_id": job_id, "cancel_requested": self.cancel_requested}

    async def get_last_update_sequence(self, job_id: str) -> int:
        if self.fail_with_status == 401:
            raise BackendUnauthorizedError(f"Backend rejected credentials (401) for job {job_id}")
        if self.fail_with_status == 404:
            raise JobNotFoundError(f"Job {job_id} not found (404)")
        if self.fail_with_status and self.fail_with_status >= 500:
            raise BackendUnavailableError(
                f"Backend status unavailable (HTTP {self.fail_with_status})"
            )
        return self._last_update_sequence.get(job_id, 0)

    async def check_cancellation(self, job_id: str) -> bool:
        if self.fail_with_status == 401:
            raise BackendUnauthorizedError(f"Backend rejected credentials (401) for job {job_id}")
        if self.fail_with_status == 404:
            raise JobNotFoundError(f"Job {job_id} not found (404)")
        if self.fail_with_status and self.fail_with_status >= 500:
            raise BackendUnavailableError(
                f"Backend status unavailable (HTTP {self.fail_with_status})"
            )
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
    "BackendUnavailableError",
    "FakeBackendClient",
    "JobConflictError",
    "JobNotFoundError",
]
