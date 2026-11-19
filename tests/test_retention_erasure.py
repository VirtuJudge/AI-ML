from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.backend_client import BackendClient, BackendUnavailableError, FakeBackendClient
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
    backend.get_last_update_sequence.return_value = 0
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
        with pytest.raises(BackendUnavailableError) as error:
            await client.check_cancellation("job")
        assert isinstance(error.value.__cause__, httpx.ConnectError)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retention_and_explicit_erasure_cover_immutable_multi_attempt_layout(
    tmp_path: Path,
) -> None:
    """Retention and explicit erasure cover all immutable attempt/job/artifact prefixes."""
    storage = LocalDiskObjectStorage(tmp_path)
    documents = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=documents)

    session_id = "target-multi"
    for attempt in (1, 2):
        job_id = f"job-{attempt}"
        await storage.upload_json(
            f"ai/session/{session_id}/attempts/{attempt}/jobs/{job_id}/art-a-{attempt}/analysis.json",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/attempts/{attempt}/jobs/{job_id}/analyze_session.result.json",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/attempts/{attempt}/jobs/rep-{attempt}/art-e-{attempt}/evaluation.json",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/attempts/{attempt}/jobs/rep-{attempt}/art-r-{attempt}/report.md",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/answers/ans-{attempt}/attempts/{attempt}/jobs/aj-{attempt}/art-t-{attempt}/transcript.json",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/answers/ans-{attempt}/attempts/{attempt}/jobs/aj-{attempt}/art-s-{attempt}/assessment.json",
            {"attempt": attempt},
            overwrite=False,
        )
        await storage.upload_json(
            f"ai/session/{session_id}/answers/ans-{attempt}/attempts/{attempt}/jobs/aj-{attempt}/analyze_answer.result.json",
            {"attempt": attempt},
            overwrite=False,
        )
    await storage.upload_json(
        f"ai/session/{session_id}/checkpoints/speech_abc.json",
        {"synthetic": True},
    )
    await storage.upload_json(
        "ai/answer/ans-orphan/attempts/1/jobs/j-orphan/art-o/transcript.json",
        {"synthetic": True},
        overwrite=False,
    )
    await storage.upload_json("ai/session/unrelated/attempts/1/jobs/j/a/analysis.json", {"s": 1})
    await storage.upload_json("uploads/backend-owned.mp4", {"synthetic": True})

    retention_job = EraseAIDataPayload(
        erasure_request_id="req-retention",
        scope="asset",
        scope_id="raw",
        practice_session_ids=[session_id],
        asset_version_ids=["raw-version"],
        retention_only=True,
    )
    retention_res = await pipeline.erase_data(retention_job)
    # 2 analysis.json + 2 analyze_session.result.json + 2 analyze_answer.result.json + 1 ckpt = 7
    assert retention_res.deleted_objects == 7
    remaining = await storage.list_objects(f"ai/session/{session_id}/")
    assert len(remaining) == 8
    assert all(
        key.endswith(("evaluation.json", "report.md", "transcript.json", "assessment.json"))
        for key in remaining
    )
    assert await storage.list_objects("ai/session/unrelated/")
    assert await storage.list_objects("uploads/")

    explicit_job = EraseAIDataPayload(
        erasure_request_id="req-explicit",
        scope="practice_session",
        scope_id=session_id,
        answer_ids=["ans-1", "ans-2", "ans-orphan"],
        retention_only=False,
    )
    explicit_res = await pipeline.erase_data(explicit_job)
    assert explicit_res.deleted_objects == 9
    assert await storage.list_objects(f"ai/session/{session_id}/") == []
    assert await storage.list_objects("ai/answer/ans-orphan/") == []
    assert await storage.list_objects("ai/session/unrelated/")
    assert await storage.list_objects("uploads/")
