from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.backend_client import BackendClient, FakeBackendClient
from app.contracts import EraseAIDataPayload, QueueMessage, WorkerUpdate
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.types import DocumentChunk
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job


@pytest.mark.asyncio
async def test_retention_preserves_reports_transcripts_and_other_sessions(tmp_path: Path) -> None:
    storage = LocalDiskObjectStorage(tmp_path)
    documents = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=documents)
    for key in (
        "report.md",
        "evaluation.json",
        "answers/a/transcript.json",
        "checkpoints/speech.json",
        "analysis.json",
    ):
        await storage.upload_json(f"ai/session/target/{key}", {"synthetic": True})
    await storage.upload_json("ai/session/unrelated/analysis.json", {"synthetic": True})
    await storage.upload_json("uploads/backend-owned.mp4", {"synthetic": True})
    await documents.store_chunks(
        "target",
        [
            DocumentChunk(
                chunk_id="raw", asset_version_id="raw-version", page_or_slide=1, text="synthetic"
            ),
            DocumentChunk(
                chunk_id="doc",
                asset_version_id="document-version",
                page_or_slide=1,
                text="synthetic",
            ),
        ],
    )
    job = EraseAIDataPayload(
        erasure_request_id="request",
        scope="asset",
        scope_id="raw",
        practice_session_ids=["target"],
        asset_version_ids=["raw-version"],
        retention_only=True,
    )
    result = await pipeline.erase_data(job)
    assert result.deleted_records == 1
    assert result.deleted_objects == 2
    assert await storage.list_objects("ai/session/target/") == [
        "ai/session/target/answers/a/transcript.json",
        "ai/session/target/evaluation.json",
        "ai/session/target/report.md",
    ]
    assert len(await documents.retrieve_top_k("target", [])) == 1
    assert await storage.list_objects("ai/session/unrelated/")
    assert await storage.list_objects("uploads/")
    retry = await pipeline.erase_data(job)
    assert retry.deleted_objects == retry.deleted_records == 0


@pytest.mark.asyncio
async def test_project_scope_requires_explicit_session_inventory(tmp_path: Path) -> None:
    storage = LocalDiskObjectStorage(tmp_path)
    await storage.upload_json("ai/session/project-id/analysis.json", {"synthetic": True})
    result = await FakePipeline(object_storage=storage).erase_data(
        EraseAIDataPayload(erasure_request_id="request", scope="project", scope_id="project-id")
    )
    assert result.deleted_objects == 0
    assert await storage.list_objects("ai/session/project-id/")


@pytest.mark.asyncio
async def test_redelivery_resumes_backend_sequence(tmp_path: Path) -> None:
    backend = AsyncMock()
    backend.send_update.side_effect = [{"last_update_sequence": 8}, {"last_update_sequence": 9}]
    message = QueueMessage(
        schema_version=1,
        job_id="job",
        job_type="erase_ai_data",
        created_at=datetime.now(UTC),
        trace_id="trace",
        payload={"erasure_request_id": "request", "scope": "project", "scope_id": "project"},
    )
    result = await process_job(
        message, FakePipeline(object_storage=LocalDiskObjectStorage(tmp_path)), backend
    )
    assert result.sequence == 9
    updates = [call.args[1] for call in backend.send_update.await_args_list]
    assert all(isinstance(update, WorkerUpdate) for update in updates)
    assert [update.sequence for update in updates] == [1, 9]


@pytest.mark.asyncio
async def test_cancelled_job_never_starts_or_writes_outputs(
    analyze_session_message, tmp_path: Path
) -> None:
    storage = LocalDiskObjectStorage(tmp_path)
    backend = FakeBackendClient(cancel_requested=True)
    result = await process_job(
        analyze_session_message, FakePipeline(object_storage=storage), backend
    )
    assert result.status.value == "cancelled"
    assert result.payload.stage == "dispatch"
    assert [update.status.value for update in backend.updates] == ["cancelled"]
    assert await storage.list_objects("ai/") == []


@pytest.mark.asyncio
async def test_unreachable_cancellation_check_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Synthetic outage", request=request)

    client = BackendClient(base_url="http://synthetic", shared_secret="synthetic")
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="http://synthetic", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(httpx.ConnectError):
            await client.check_cancellation("job")
    finally:
        await client.close()
