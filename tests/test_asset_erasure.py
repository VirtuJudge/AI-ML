from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.contracts import ErasureCompleted, QueueMessage, UpdateStatus
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.types import DocumentChunk
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job


@pytest.mark.asyncio
@pytest.mark.parametrize("sessions", [[], ["draft"]])
async def test_explicit_asset_erasure_purges_all_source_versions_without_live_session_links(
    tmp_path: Path, sessions: list[str]
) -> None:
    storage = LocalDiskObjectStorage(tmp_path)
    documents = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=documents)
    for session_id in ("historical", "other-historical", "draft"):
        await documents.store_chunks(
            session_id,
            [
                DocumentChunk(
                    chunk_id=version,
                    asset_version_id=version,
                    page_or_slide=1,
                    text="Synthetic source",
                    embedding=[1.0, 0.0],
                )
                for version in ("old-version", "current-version", "unrelated-version")
            ],
        )
        await storage.upload_json(f"ai/session/{session_id}/report.md", {"synthetic": True})
    await storage.upload_json("uploads/backend-owned.pdf", {"synthetic": True})
    message = QueueMessage(
        schema_version=1,
        job_id="asset-erasure-job",
        job_type="erase_ai_data",
        created_at=datetime.now(UTC),
        trace_id="synthetic-trace",
        payload={
            "erasure_request_id": "asset-erasure-request",
            "scope": "asset",
            "scope_id": "asset",
            "practice_session_ids": sessions,
            "asset_version_ids": ["old-version", "current-version"],
            "retention_only": False,
        },
    )

    result = await process_job(message, pipeline)

    assert result.status is UpdateStatus.COMPLETED
    assert isinstance(result.payload, ErasureCompleted)
    assert result.payload.deleted_records == 6 + len(sessions)
    assert result.payload.deleted_objects == len(sessions)
    for session_id in ("historical", "other-historical", "draft"):
        remaining = await documents.retrieve_top_k(session_id, [])
        assert [chunk.asset_version_id for chunk in remaining] == (
            [] if session_id in sessions else ["unrelated-version"]
        )
        assert bool(await storage.list_objects(f"ai/session/{session_id}/")) == (
            session_id not in sessions
        )
    assert await storage.list_objects("uploads/") == ["uploads/backend-owned.pdf"]

    retry = await process_job(message, pipeline)
    assert retry.status is UpdateStatus.COMPLETED
    assert isinstance(retry.payload, ErasureCompleted)
    assert retry.payload.deleted_records == retry.payload.deleted_objects == 0
