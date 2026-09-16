"""Tests for BackendClient and worker callback integration."""

import json

import httpx
import pytest

from app.backend_client import BackendClient, FakeBackendClient
from app.contracts import (
    QueueMessage,
    StartedPayload,
    UpdateStatus,
    WorkerUpdate,
)
from app.pipeline import FakePipeline
from app.worker import process_job


@pytest.mark.asyncio
async def test_fake_backend_client_records_updates() -> None:
    """Test that FakeBackendClient accumulates updates in memory."""
    fake_client = FakeBackendClient()
    dummy_update = WorkerUpdate(
        schema_version=1,
        sequence=1,
        status=UpdateStatus.STARTED,
        occurred_at="2026-09-02T12:00:00Z",
        trace_id="test_trace",
        payload=StartedPayload(pipeline_version="1.0.0"),
    )

    await fake_client.send_update("job_123", dummy_update)
    assert len(fake_client.updates) == 1
    assert fake_client.updates[0].sequence == 1
    assert fake_client.updates[0].status == UpdateStatus.STARTED


@pytest.mark.asyncio
async def test_fake_backend_client_cancellation() -> None:
    """Test cancellation flag return on FakeBackendClient."""
    client_not_cancelled = FakeBackendClient(cancel_requested=False)
    assert await client_not_cancelled.check_cancellation("job_1") is False

    client_cancelled = FakeBackendClient(cancel_requested=True)
    assert await client_cancelled.check_cancellation("job_1") is True


@pytest.mark.asyncio
async def test_process_job_with_backend_client(
    analyze_session_message: QueueMessage, fake_pipeline: FakePipeline
) -> None:
    """Test process_job sends both started and completed updates to BackendClient."""
    backend_client = FakeBackendClient()
    terminal_update = await process_job(
        analyze_session_message,
        fake_pipeline,
        backend_client=backend_client,
    )

    assert terminal_update.status == UpdateStatus.COMPLETED
    assert len(backend_client.updates) >= 2

    # Check strictly monotonic update sequence
    for idx, u in enumerate(backend_client.updates, 1):
        assert u.sequence == idx

    assert backend_client.updates[0].sequence == 1
    assert backend_client.updates[0].status == UpdateStatus.STARTED
    assert backend_client.updates[-1].sequence == len(backend_client.updates)
    assert backend_client.updates[-1].status == UpdateStatus.COMPLETED


@pytest.mark.asyncio
async def test_backend_client_http_send_update() -> None:
    """Test BackendClient sends proper HTTP request with headers and payload."""
    recorded_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    client = BackendClient(base_url="http://testbackend:4000", shared_secret="secret_123")
    client.client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testbackend:4000",
        headers={
            "Authorization": "Bearer secret_123",
            "X-Worker-Secret": "secret_123",
            "Content-Type": "application/json",
        },
    )

    update = WorkerUpdate(
        schema_version=1,
        sequence=1,
        status=UpdateStatus.STARTED,
        occurred_at="2026-09-02T12:00:00Z",
        trace_id="test_trace",
        payload=StartedPayload(pipeline_version="1.0.0"),
    )

    try:
        await client.send_update("job_abc", update)
        assert len(recorded_requests) == 1
        req = recorded_requests[0]
        assert req.method == "POST"
        assert req.url.path == "/internal/v1/ai-jobs/job_abc/updates"
        assert req.headers["authorization"] == "Bearer secret_123"
        body = json.loads(req.content)
        assert body["status"] == "started"
        assert body["sequence"] == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_backend_client_http_check_cancellation() -> None:
    """Test BackendClient parses cancellation flag from backend response."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/v1/ai-jobs/job_cancelled":
            return httpx.Response(200, json={"cancel_requested": True})
        return httpx.Response(200, json={"cancel_requested": False})

    transport = httpx.MockTransport(handler)
    client = BackendClient(base_url="http://testbackend:4000", shared_secret="secret_123")
    client.client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testbackend:4000",
        headers={"Authorization": "Bearer secret_123"},
    )

    try:
        is_cancelled = await client.check_cancellation("job_cancelled")
        assert is_cancelled is True

        not_cancelled = await client.check_cancellation("job_running")
        assert not_cancelled is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_backend_client_production_config_validation() -> None:
    """Test BackendClient fails fast in production when URL or secret is missing/invalid."""
    from app.backend_client import BackendConfigurationError

    with pytest.raises(BackendConfigurationError, match="BACKEND_INTERNAL_URL is required"):
        BackendClient(base_url="", shared_secret="sec", strict_production=True)

    with pytest.raises(BackendConfigurationError, match="cannot use localhost"):
        BackendClient(base_url="http://localhost:8000", shared_secret="sec", strict_production=True)

    with pytest.raises(BackendConfigurationError, match="AI_WORKER_SHARED_SECRET is required"):
        BackendClient(base_url="https://virtujudge-backend.onrender.com", shared_secret="", strict_production=True)

    # Valid production config succeeds
    client = BackendClient(
        base_url="https://virtujudge-backend.onrender.com",
        shared_secret="valid_secret",
        strict_production=True,
    )
    assert client.base_url == "https://virtujudge-backend.onrender.com"
    await client.close()


@pytest.mark.asyncio
async def test_backend_client_http_status_errors() -> None:
    """Test BackendClient maps 401, 404, 409 to typed exceptions."""
    from app.backend_client import (
        BackendUnauthorizedError,
        JobConflictError,
        JobNotFoundError,
    )
    from app.contracts import CancelledPayload

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "job_401" in path:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        if "job_404" in path:
            return httpx.Response(404, json={"detail": "Not found"})
        if "job_409" in path:
            return httpx.Response(409, json={"detail": "Conflict"})
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    client = BackendClient(base_url="http://testbackend:4000", shared_secret="sec")
    client.client = httpx.AsyncClient(transport=transport, base_url="http://testbackend:4000")

    dummy_update = WorkerUpdate(
        schema_version=1,
        sequence=1,
        status=UpdateStatus.STARTED,
        occurred_at="2026-09-02T12:00:00Z",
        trace_id="trc_1",
        payload=StartedPayload(pipeline_version="1.0.0"),
    )

    try:
        with pytest.raises(BackendUnauthorizedError):
            await client.send_update("job_401", dummy_update)

        with pytest.raises(JobNotFoundError):
            await client.send_update("job_404", dummy_update)

        with pytest.raises(JobConflictError):
            await client.send_update("job_409", dummy_update)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_backend_client_retries_terminal_update_after_active_job_conflict() -> None:
    """A callback race must not orphan a report that was already uploaded."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and len([call for call in calls if call[0] == "POST"]) == 1:
            return httpx.Response(409, json={"detail": "Concurrent update"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "status": "running",
                    "last_update_sequence": 4,
                    "cancel_requested": False,
                },
            )
        return httpx.Response(200, json={"status": "completed"})

    client = BackendClient(base_url="http://testbackend:4000", shared_secret="sec")
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=client.base_url
    )
    update = WorkerUpdate(
        schema_version=1,
        sequence=5,
        status=UpdateStatus.COMPLETED,
        occurred_at="2026-09-02T12:00:00Z",
        trace_id="trc_report",
        payload={
            "evaluation_artifact": {
                "artifact_id": "eval_1",
                "object_key": "ai/session/session_1/evaluation.json",
                "checksum": "sha256:" + "a" * 64,
                "schema_version": 1,
            },
            "report_artifact": {
                "artifact_id": "report_1",
                "object_key": "ai/session/session_1/report.md",
                "checksum": "sha256:" + "b" * 64,
                "schema_version": 1,
            },
            "member_feedback_user_ids": [],
            "limitations": [],
        },
    )

    try:
        result = await client.send_update("job_report", update)
    finally:
        await client.close()

    assert result == {"status": "completed"}
    assert calls == [
        ("POST", "/internal/v1/ai-jobs/job_report/updates"),
        ("GET", "/internal/v1/ai-jobs/job_report"),
        ("POST", "/internal/v1/ai-jobs/job_report/updates"),
    ]


@pytest.mark.asyncio
async def test_backend_client_cancelled_payload_is_empty_dict() -> None:
    """Test that cancelled update sends empty dict payload to backend."""
    from app.contracts import CancelledPayload

    recorded_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recorded_body
        recorded_body = json.loads(request.content)
        return httpx.Response(200, json={"status": "cancelled"})

    transport = httpx.MockTransport(handler)
    client = BackendClient(base_url="http://testbackend:4000", shared_secret="sec")
    client.client = httpx.AsyncClient(transport=transport, base_url="http://testbackend:4000")

    cancelled_update = WorkerUpdate(
        schema_version=1,
        sequence=2,
        status=UpdateStatus.CANCELLED,
        occurred_at="2026-09-02T12:00:00Z",
        trace_id="trc_cancel",
        payload=CancelledPayload(stage="speech", message="User cancelled"),
    )

    try:
        await client.send_update("job_cancel", cancelled_update)
        assert recorded_body["status"] == "cancelled"
        assert recorded_body["payload"] == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_backend_client_reachability() -> None:
    """Test check_backend_reachability handles health response."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = BackendClient(base_url="http://testbackend:4000", shared_secret="sec")
    client.client = httpx.AsyncClient(transport=transport, base_url="http://testbackend:4000")

    try:
        assert await client.check_backend_reachability() is True
    finally:
        await client.close()
