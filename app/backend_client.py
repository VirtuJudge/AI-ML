"""Backend HTTP callback client for VirtuJudge AI-ML worker."""

import os
from typing import Any, Protocol

import httpx

from app.contracts import WorkerUpdate


class BackendClientProtocol(Protocol):
    """Protocol defining methods for backend callback communication."""

    async def send_update(self, job_id: str, update: WorkerUpdate) -> None: ...

    async def check_cancellation(self, job_id: str) -> bool: ...

    async def close(self) -> None: ...


class BackendClient:
    """Production HTTP client for reporting job status and checking cancellation."""

    def __init__(
        self,
        base_url: str | None = None,
        shared_secret: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        raw_base_url = (
            base_url
            if base_url is not None
            else os.getenv("BACKEND_INTERNAL_URL", "http://localhost:3000")
        )
        self.base_url: str = raw_base_url.rstrip("/")
        self.shared_secret: str = (
            shared_secret if shared_secret is not None else os.getenv("AI_WORKER_SHARED_SECRET", "")
        )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.shared_secret}",
            "X-Worker-Secret": self.shared_secret,
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    async def send_update(self, job_id: str, update: WorkerUpdate) -> None:
        """POST job status update to the internal backend endpoint."""
        url = f"/internal/v1/ai-jobs/{job_id}/updates"
        payload = update.model_dump(mode="json")

        max_attempts = 3
        for attempt in range(max_attempts):
            response = await self.client.post(url, json=payload)
            if response.status_code < 500:
                response.raise_for_status()
                return
            if attempt == max_attempts - 1:
                response.raise_for_status()

    async def check_cancellation(self, job_id: str) -> bool:
        """Query the internal backend endpoint to check if job was cancelled."""
        url = f"/internal/v1/ai-jobs/{job_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return bool(data.get("cancel_requested", False))

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()


class FakeBackendClient:
    """In-memory backend client for tests."""

    def __init__(self, cancel_requested: bool = False) -> None:
        self.updates: list[WorkerUpdate] = []
        self.cancel_requested = cancel_requested

    async def send_update(self, job_id: str, update: WorkerUpdate) -> None:
        self.updates.append(update)

    async def check_cancellation(self, job_id: str) -> bool:
        return self.cancel_requested

    async def close(self) -> None:
        pass


__all__ = [
    "BackendClient",
    "BackendClientProtocol",
    "FakeBackendClient",
]
