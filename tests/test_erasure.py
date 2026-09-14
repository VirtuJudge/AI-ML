"""Comprehensive test suite for ADR 0007-compliant AI data erasure.

Verifies:
- Scoped erasure for scope="practice_session":
  - Intermediate artifacts (analysis.json, checkpoints/*, answers/*) are physically removed.
  - Document chunks in document_store are deleted.
  - Local scratch media (.storage/temp_media/{session_id}) is removed.
  - CRITICAL: report.md and evaluation.json under ai/session/{session_id}/ are
    STRICTLY PRESERVED for the 30-day student access window.
- Scoped erasure for scope="project" and scope="team":
  - Full purge: ALL objects under ai/session/{session_id}/* are removed
    (including report.md and evaluation.json).
- Erasure idempotency:
  - Calling erase_data multiple times for the same session succeeds cleanly.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.contracts import (
    EraseAIDataPayload,
    ErasureCompleted,
    JobType,
    QueueMessage,
    UpdateStatus,
)
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.types import DocumentChunk
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job


async def _populate_session_data(
    storage: LocalDiskObjectStorage,
    document_store: FakeDocumentStore,
    session_id: str,
) -> Path:
    """Populate full suite of artifacts, vector chunks, and scratch media for a session."""
    # 1. Intermediate artifacts in storage:
    await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        {"artifact_id": f"{session_id}:analysis", "status": "analyzed"},
    )
    await storage.upload_json(
        f"ai/session/{session_id}/checkpoints/speech_abc123.json",
        {"stage": "speech", "data": {"transcription": "hello"}},
    )
    await storage.upload_json(
        f"ai/session/{session_id}/checkpoints/vision_def456.json",
        {"stage": "vision", "data": {"observations": []}},
    )
    await storage.upload_json(
        f"ai/session/{session_id}/checkpoints/audio_ghi789.json",
        {"stage": "audio", "data": {"observations": []}},
    )
    await storage.upload_json(
        f"ai/session/{session_id}/answers/ans_001/transcript.json",
        {"answer_id": "ans_001", "text": "Our customer acquisition cost is low."},
    )

    # 2. Retained final artifacts in storage:
    report_file = storage.base_dir / f"ai/session/{session_id}/report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        "# Pitch Evaluation Report\n\nPreserved strictly for 30-day student access.",
        encoding="utf-8",
    )

    await storage.upload_json(
        f"ai/session/{session_id}/evaluation.json",
        {"report_id": session_id, "overall_score": 92.0, "status": "published"},
    )

    # 3. Document chunks in document store:
    chunks = [
        DocumentChunk(
            chunk_id=f"{session_id}_chunk_0",
            page_or_slide=1,
            text="Executive summary of the startup pitch.",
        ),
        DocumentChunk(
            chunk_id=f"{session_id}_chunk_1",
            page_or_slide=2,
            text="Unit economics and competitive differentiation.",
        ),
    ]
    await document_store.store_chunks(session_id, chunks)

    # 4. Scratch media in .storage/temp_media/{session_id}:
    scratch_dir = Path(".storage/temp_media") / session_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "temp_audio.wav").write_bytes(b"temporary audio bytes")
    (scratch_dir / "temp_video.mp4").write_bytes(b"temporary video bytes")

    return scratch_dir


# ---------------------------------------------------------------------------
# 1. Scoped erasure for scope="practice_session" (ADR 0007)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_erasure_practice_session(tmp_path: Path) -> None:
    """Test practice_session erasure purges intermediates while preserving report & evaluation."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000001"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        # Pre-conditions
        assert (tmp_path / f"ai/session/{session_id}/analysis.json").is_file()
        assert (tmp_path / f"ai/session/{session_id}/checkpoints/speech_abc123.json").is_file()
        assert (tmp_path / f"ai/session/{session_id}/answers/ans_001/transcript.json").is_file()
        assert (tmp_path / f"ai/session/{session_id}/report.md").is_file()
        assert (tmp_path / f"ai/session/{session_id}/evaluation.json").is_file()
        assert len(await document_store.retrieve_top_k(session_id, query_embedding=[])) == 2
        assert scratch_dir.exists()

        job = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000001",
            scope="practice_session",
            scope_id=session_id,
        )
        erasure_result = await pipeline.erase_data(job)

        # 1. Result contract conformance
        assert isinstance(erasure_result, ErasureCompleted)
        assert erasure_result.erasure_request_id == "01JERASUREREQ000000000001"

        # 2. Intermediate artifacts are physically removed
        assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()
        assert not (
            tmp_path / f"ai/session/{session_id}/checkpoints/speech_abc123.json"
        ).exists()
        assert not (
            tmp_path / f"ai/session/{session_id}/checkpoints/vision_def456.json"
        ).exists()
        assert not (
            tmp_path / f"ai/session/{session_id}/checkpoints/audio_ghi789.json"
        ).exists()
        assert not (
            tmp_path / f"ai/session/{session_id}/answers/ans_001/transcript.json"
        ).exists()

        # 3. Document chunks in document_store are deleted
        remaining_chunks = await document_store.retrieve_top_k(session_id, query_embedding=[])
        assert remaining_chunks == []

        # 4. Local scratch media is removed
        assert not scratch_dir.exists()

        # 5. CRITICAL: report.md and evaluation.json are STRICTLY PRESERVED
        # for 30-day student window
        report_path = tmp_path / f"ai/session/{session_id}/report.md"
        evaluation_path = tmp_path / f"ai/session/{session_id}/evaluation.json"

        assert report_path.is_file()
        report_content = report_path.read_text(encoding="utf-8")
        assert "Preserved strictly for 30-day student access." in report_content

        assert evaluation_path.is_file()
        eval_data = await storage.read_json(f"ai/session/{session_id}/evaluation.json")
        assert eval_data["overall_score"] == 92.0
        assert eval_data["status"] == "published"
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Scoped erasure for scope="project" and scope="team" (Full Purge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_purge_project_scope(tmp_path: Path) -> None:
    """Test project scope erasure performs a full purge removing report and evaluation."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000002"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        job = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000002",
            scope="project",
            scope_id=session_id,
        )
        erasure_result = await pipeline.erase_data(job)

        assert isinstance(erasure_result, ErasureCompleted)
        assert erasure_result.erasure_request_id == "01JERASUREREQ000000000002"

        # Full purge: ALL objects under ai/session/{session_id}/* are removed
        assert not (tmp_path / f"ai/session/{session_id}/report.md").exists()
        assert not (tmp_path / f"ai/session/{session_id}/evaluation.json").exists()
        assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()
        assert not (tmp_path / f"ai/session/{session_id}/checkpoints").exists()

        stored_keys = await storage.list_objects(f"ai/session/{session_id}/")
        assert stored_keys == []

        # Vector document chunks are purged
        remaining_chunks = await document_store.retrieve_top_k(session_id, query_embedding=[])
        assert remaining_chunks == []
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_purge_team_scope(tmp_path: Path) -> None:
    """Test team scope erasure performs a full purge removing report and evaluation."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000003"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        job = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000003",
            scope="team",
            scope_id=session_id,
        )
        erasure_result = await pipeline.erase_data(job)

        assert isinstance(erasure_result, ErasureCompleted)
        assert erasure_result.erasure_request_id == "01JERASUREREQ000000000003"

        # Full purge: ALL objects under ai/session/{session_id}/* are removed
        assert not (tmp_path / f"ai/session/{session_id}/report.md").exists()
        assert not (tmp_path / f"ai/session/{session_id}/evaluation.json").exists()
        assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()
        assert not (tmp_path / f"ai/session/{session_id}/checkpoints").exists()

        stored_keys = await storage.list_objects(f"ai/session/{session_id}/")
        assert stored_keys == []

        # Vector document chunks are purged
        remaining_chunks = await document_store.retrieve_top_k(session_id, query_embedding=[])
        assert remaining_chunks == []
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Erasure idempotency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_erasure_idempotency_practice_session(tmp_path: Path) -> None:
    """Test calling erase_data multiple times for the same practice_session succeeds cleanly."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000004"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        job = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000004",
            scope="practice_session",
            scope_id=session_id,
        )

        # 1st execution
        res1 = await pipeline.erase_data(job)
        assert isinstance(res1, ErasureCompleted)
        assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()
        assert (tmp_path / f"ai/session/{session_id}/report.md").is_file()
        assert (tmp_path / f"ai/session/{session_id}/evaluation.json").is_file()

        # 2nd execution (idempotent retry)
        res2 = await pipeline.erase_data(job)
        assert isinstance(res2, ErasureCompleted)
        assert res2.erasure_request_id == job.erasure_request_id
        # Preserved artifacts remain intact
        assert (tmp_path / f"ai/session/{session_id}/report.md").is_file()
        assert (tmp_path / f"ai/session/{session_id}/evaluation.json").is_file()

        # 3rd execution (idempotent retry)
        res3 = await pipeline.erase_data(job)
        assert isinstance(res3, ErasureCompleted)
        assert (tmp_path / f"ai/session/{session_id}/report.md").is_file()
        assert (tmp_path / f"ai/session/{session_id}/evaluation.json").is_file()
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_erasure_idempotency_full_purge(tmp_path: Path) -> None:
    """Test calling erase_data multiple times on project/team scope succeeds cleanly."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000005"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        job_project = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000005",
            scope="project",
            scope_id=session_id,
        )

        # 1st purge
        res1 = await pipeline.erase_data(job_project)
        assert isinstance(res1, ErasureCompleted)
        assert await storage.list_objects(f"ai/session/{session_id}/") == []

        # 2nd purge (idempotent retry)
        res2 = await pipeline.erase_data(job_project)
        assert isinstance(res2, ErasureCompleted)
        assert await storage.list_objects(f"ai/session/{session_id}/") == []

        # 3rd purge with team scope
        job_team = EraseAIDataPayload(
            erasure_request_id="01JERASUREREQ000000000006",
            scope="team",
            scope_id=session_id,
        )
        res3 = await pipeline.erase_data(job_team)
        assert isinstance(res3, ErasureCompleted)
        assert await storage.list_objects(f"ai/session/{session_id}/") == []
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Worker integration for erasure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_process_job_erasure(tmp_path: Path) -> None:
    """Test process_job correctly processes an ERASE_AI_DATA QueueMessage."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    session_id = "01JERASURETEST000000000006"
    scratch_dir = await _populate_session_data(storage, document_store, session_id)

    try:
        message = QueueMessage(
            schema_version=1,
            job_id="01JWORKERERASE000000000001",
            job_type=JobType.ERASE_AI_DATA,
            practice_session_id=session_id,
            analysis_attempt=1,
            created_at=datetime.now(UTC),
            trace_id="01JTRACEERASE0000000000001",
            payload={
                "erasure_request_id": "01JERASUREREQ000000000007",
                "scope": "practice_session",
                "scope_id": session_id,
            },
        )

        update = await process_job(message, pipeline)

        assert update.status == UpdateStatus.COMPLETED
        assert update.sequence == 2
        assert isinstance(update.payload, ErasureCompleted)
        assert update.payload.erasure_request_id == "01JERASUREREQ000000000007"

        # Intermediate artifacts removed; final report and evaluation strictly preserved
        assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()
        assert (tmp_path / f"ai/session/{session_id}/report.md").is_file()
        assert (tmp_path / f"ai/session/{session_id}/evaluation.json").is_file()
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
