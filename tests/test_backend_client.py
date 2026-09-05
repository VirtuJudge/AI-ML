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
    assert len(backend_client.updates) == 2

    # Check update sequence
    assert backend_client.updates[0].sequence == 1
    assert backend_client.updates[0].status == UpdateStatus.STARTED
    assert backend_client.updates[1].sequence == 2
    assert backend_client.updates[1].status == UpdateStatus.COMPLETED


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
