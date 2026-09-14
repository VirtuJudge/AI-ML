"""Comprehensive reliability and cancellation test suite for AI-08.

Verifies:
- Inter-stage cancellation for analyze_session, analyze_answer, and generate_report.
- Local scratch directory cleanup upon cancellation.
- Stage checkpoint reuse across attempt N+1 with checksum validation.
- Transient error handling in worker with safe, non-leaking payloads.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.contracts import (
    AnalyzeSessionPayload,
    CancelledPayload,
    ErrorCode,
    FailedPayload,
    QueueMessage,
    UpdateStatus,
    WorkerUpdate,
)
from app.pipeline import FakePipeline
from app.providers.fake_speech import FakeSpeechProvider
from app.providers.types import TranscriptionResult
from app.stages.checkpoint import get_stage_checkpoint, save_stage_checkpoint
from app.stages.common import StageTransientError
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job


class StepCancellationBackendClient:
    """Backend client simulating job cancellation at a designated call index."""

    def __init__(self, cancel_at_call: int = 1) -> None:
        self.cancel_at_call = cancel_at_call
        self.call_count = 0
        self.updates: list[WorkerUpdate] = []

    async def send_update(self, job_id: str, update: WorkerUpdate) -> None:
        self.updates.append(update)

    async def check_cancellation(self, job_id: str) -> bool:
        self.call_count += 1
        return self.call_count >= self.cancel_at_call

    async def close(self) -> None:
        pass


class SpySpeechProvider(FakeSpeechProvider):
    """Spy speech provider tracking transcription invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self.call_count += 1
        return await super().transcribe(audio_path)


class FailingTransientPipeline(FakePipeline):
    """Pipeline variant raising StageTransientError with sensitive details."""

    def __init__(self, sensitive_error_message: str) -> None:
        super().__init__()
        self.sensitive_error_message = sensitive_error_message

    async def analyze_session(self, *args: Any, **kwargs: Any) -> Any:
        raise StageTransientError(self.sensitive_error_message)

    async def analyze_answer(self, *args: Any, **kwargs: Any) -> Any:
        raise StageTransientError(self.sensitive_error_message)

    async def generate_report(self, *args: Any, **kwargs: Any) -> Any:
        raise StageTransientError(self.sensitive_error_message)


# ---------------------------------------------------------------------------
# 1. Inter-stage cancellation tests for analyze_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_session_cancelled_before_speech(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test cancellation before speech stops worker, emits cancelled status, without upload."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    backend_client = StepCancellationBackendClient(cancel_at_call=1)

    terminal_update = await process_job(
        analyze_session_message, pipeline, backend_client=backend_client
    )

    # Worker stops and emits sequence=2, status="cancelled"
    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert terminal_update.status.value == "cancelled"
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "speech"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )

    # Backend received sequence=1 (started) and sequence=2 (cancelled)
    assert len(backend_client.updates) == 2
    assert backend_client.updates[0].sequence == 1
    assert backend_client.updates[0].status == UpdateStatus.STARTED
    assert backend_client.updates[1].sequence == 2
    assert backend_client.updates[1].status == UpdateStatus.CANCELLED
    assert isinstance(backend_client.updates[1].payload, CancelledPayload)
    assert backend_client.updates[1].payload.stage == "speech"

    # CRITICAL: no analysis.json was uploaded
    session_id = analyze_session_message.practice_session_id
    stored_keys = await storage.list_objects(f"ai/session/{session_id}/")
    assert not any("analysis.json" in k for k in stored_keys)
    assert not (tmp_path / f"ai/session/{session_id}/analysis.json").exists()


@pytest.mark.asyncio
async def test_analyze_session_cancelled_after_speech_before_vision(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test cancellation after speech and before vision reports stage='vision'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    # Inspection call 1 is speech (False), call 2 is vision (True)
    backend_client = StepCancellationBackendClient(cancel_at_call=2)

    terminal_update = await process_job(
        analyze_session_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "vision"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


@pytest.mark.asyncio
async def test_analyze_session_cancelled_before_questions(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test cancellation after documents and before questions reports stage='questions'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    # Inspection call 1: speech, call 2: vision, call 3: documents, call 4: questions
    backend_client = StepCancellationBackendClient(cancel_at_call=4)

    terminal_update = await process_job(
        analyze_session_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "questions"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


@pytest.mark.asyncio
async def test_analyze_session_cancelled_before_upload(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test cancellation before final analysis.json upload reports stage='aggregation'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    # Inspection calls: 1 speech, 2 vision, 3 docs, 4 questions, 5 aggregation
    backend_client = StepCancellationBackendClient(cancel_at_call=5)

    terminal_update = await process_job(
        analyze_session_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "aggregation"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )

    # Confirm final analysis.json upload was aborted
    session_id = analyze_session_message.practice_session_id
    stored_keys = await storage.list_objects(f"ai/session/{session_id}/")
    assert not any("analysis.json" in k for k in stored_keys)


# ---------------------------------------------------------------------------
# 2. Cancellation tests for analyze_answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_answer_cancelled_before_transcription(
    analyze_answer_message: QueueMessage, tmp_path: Path
) -> None:
    """Test analyze_answer cancellation before transcription reports stage='speech'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    backend_client = StepCancellationBackendClient(cancel_at_call=1)

    terminal_update = await process_job(
        analyze_answer_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "speech"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


@pytest.mark.asyncio
async def test_analyze_answer_cancelled_before_assessment(
    analyze_answer_message: QueueMessage, tmp_path: Path
) -> None:
    """Test analyze_answer cancellation before assessment reports stage='assessment'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    # Call 1: speech (False), Call 2: assessment (True)
    backend_client = StepCancellationBackendClient(cancel_at_call=2)

    terminal_update = await process_job(
        analyze_answer_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "assessment"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


# ---------------------------------------------------------------------------
# 3. Cancellation tests for generate_report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_report_cancelled_before_ingestion(
    generate_report_message: QueueMessage, tmp_path: Path
) -> None:
    """Test generate_report cancellation before ingestion reports stage='ingestion'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    backend_client = StepCancellationBackendClient(cancel_at_call=1)

    terminal_update = await process_job(
        generate_report_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "ingestion"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


@pytest.mark.asyncio
async def test_generate_report_cancelled_before_synthesis(
    generate_report_message: QueueMessage, tmp_path: Path
) -> None:
    """Test generate_report cancellation before synthesis reports stage='reporting'."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)
    # Call 1: ingestion (False), Call 2: reporting synthesis (True)
    backend_client = StepCancellationBackendClient(cancel_at_call=2)

    terminal_update = await process_job(
        generate_report_message, pipeline, backend_client=backend_client
    )

    assert terminal_update.sequence == 2
    assert terminal_update.status == UpdateStatus.CANCELLED
    assert isinstance(terminal_update.payload, CancelledPayload)
    assert terminal_update.payload.stage == "reporting"
    assert (
        terminal_update.payload.message
        == "Analysis cancelled by user request."
    )


# ---------------------------------------------------------------------------
# 4. Scratch media cleanup upon cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_cleans_local_scratch_directory(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test cancellation removes local scratch directory .storage/temp_media/{session_id}."""
    session_id = analyze_session_message.practice_session_id
    assert session_id is not None
    scratch_dir = Path(".storage/temp_media") / session_id

    try:
        # Pre-create scratch directory with mock media chunks
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / "audio_extracted.wav").write_bytes(b"dummy audio wav")
        (scratch_dir / "video_extracted.mp4").write_bytes(b"dummy video mp4")
        assert scratch_dir.exists()
        assert (scratch_dir / "audio_extracted.wav").is_file()

        storage = LocalDiskObjectStorage(base_dir=tmp_path)
        pipeline = FakePipeline(object_storage=storage)
        backend_client = StepCancellationBackendClient(cancel_at_call=1)

        terminal_update = await process_job(
            analyze_session_message, pipeline, backend_client=backend_client
        )

        assert terminal_update.status == UpdateStatus.CANCELLED
        # Scratch directory must be cleaned up
        assert not scratch_dir.exists()
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. Stage checkpoint reuse across attempt N+1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_checkpoint_reuse_across_attempts(
    analyze_session_message: QueueMessage, tmp_path: Path
) -> None:
    """Test stage checkpoint reuse across attempt 1, attempt 2 (same), and attempt 2 (changed)."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    spy_speech = SpySpeechProvider()
    pipeline = FakePipeline(speech_provider=spy_speech, object_storage=storage)

    job_v1 = AnalyzeSessionPayload.model_validate(analyze_session_message.payload)
    session_id = (
        job_v1.practice_session_id
        or getattr(job_v1, "session_id", None)
        or job_v1.presentation.artifact_id
    )
    checksum_v1 = job_v1.presentation.checksum

    # 1. On attempt=1: runs provider and writes checkpoint
    res_attempt1 = await pipeline.analyze_session(
        job_v1, job_id="job_attempt_1", attempt=1
    )
    assert res_attempt1 is not None
    assert spy_speech.call_count == 1

    clean_hash_v1 = checksum_v1.removeprefix("sha256:")
    expected_ckpt_key = f"ai/session/{session_id}/checkpoints/speech_{clean_hash_v1}.json"
    assert (tmp_path / expected_ckpt_key).is_file()

    cached_v1 = await get_stage_checkpoint(
        storage, session_id, "speech", checksum_v1
    )
    assert cached_v1 is not None

    # 2. On attempt=2 with same checksum: reuses cached checkpoint without calling provider
    res_attempt2_same = await pipeline.analyze_session(
        job_v1, job_id="job_attempt_2", attempt=2
    )
    assert res_attempt2_same is not None
    # Provider call count MUST remain 1
    assert spy_speech.call_count == 1

    # 3. On attempt=2 with changed checksum: invalidates cache and re-runs provider
    checksum_v2 = "sha256:" + "f" * 64
    job_v2 = job_v1.model_copy(deep=True)
    job_v2.presentation.checksum = checksum_v2

    res_attempt2_changed = await pipeline.analyze_session(
        job_v2, job_id="job_attempt_2_changed", attempt=2
    )
    assert res_attempt2_changed is not None
    # Provider call count MUST increment to 2
    assert spy_speech.call_count == 2

    # New checkpoint must be written for the new checksum
    clean_hash_v2 = checksum_v2.removeprefix("sha256:")
    new_ckpt_key = f"ai/session/{session_id}/checkpoints/speech_{clean_hash_v2}.json"
    assert (tmp_path / new_ckpt_key).is_file()


@pytest.mark.asyncio
async def test_stage_checkpoint_unit_invalidation_on_checksum_change(
    tmp_path: Path,
) -> None:
    """Unit test for get_stage_checkpoint and save_stage_checkpoint invalidation mechanics."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JTESTCKPT00000000000001"
    stage = "speech"
    checksum_1 = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    checksum_2 = "sha256:2222222222222222222222222222222222222222222222222222222222222222"

    class SampleData(BaseModel):
        text: str
        confidence: float

    # On attempt=1: write checkpoint
    await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum_1,
        data=SampleData(text="Founder pitch speech", confidence=0.99),
    )

    # Same checksum: cache hit
    hit = await get_stage_checkpoint(storage, session_id, stage, checksum_1)
    assert hit is not None
    assert hit["text"] == "Founder pitch speech"

    # Changed checksum: cache miss / invalidation
    miss = await get_stage_checkpoint(storage, session_id, stage, checksum_2)
    assert miss is None


# ---------------------------------------------------------------------------
# 6. Transient error handling in worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_transient_error_handling(
    analyze_session_message: QueueMessage,
) -> None:
    """Test StageTransientError produces failed, retryable update without leaking internals."""
    sensitive_leak = (
        "Groq 503 Service Unavailable: upstream server timeout; "
        "secret_token=gsk_live_SECRET999888; path=/home/user/backend/internal_key.pem; "
        "Traceback (most recent call last):\n"
        '  File "/internal/groq_client.py", line 42, in call_api\n'
    )
    failing_pipeline = FailingTransientPipeline(sensitive_leak)
    backend_client = StepCancellationBackendClient()

    terminal_update = await process_job(
        analyze_session_message, failing_pipeline, backend_client=backend_client
    )

    # 1. Update status and payload contract
    assert terminal_update.status == UpdateStatus.FAILED
    assert terminal_update.status.value == "failed"
    assert terminal_update.sequence == 2
    assert isinstance(terminal_update.payload, FailedPayload)
    assert terminal_update.payload.retryable is True
    assert terminal_update.payload.code == ErrorCode.PROVIDER_ERROR
    assert terminal_update.payload.code == "provider_error"

    # 2. Safe message without leaking stack traces or provider internals
    assert (
        terminal_update.payload.message
        == "A transient external provider error occurred."
    )
    serialized_payload = terminal_update.payload.model_dump_json()
    assert "gsk_live_SECRET999888" not in serialized_payload
    assert "internal_key.pem" not in serialized_payload
    assert "Traceback" not in serialized_payload
    assert "groq_client.py" not in serialized_payload
    assert "503 Service Unavailable" not in serialized_payload

    # 3. Backend client received the terminal update with same safe payload
    assert len(backend_client.updates) == 2
    terminal_client_update = backend_client.updates[1]
    assert terminal_client_update.status == UpdateStatus.FAILED
    assert isinstance(terminal_client_update.payload, FailedPayload)
    assert terminal_client_update.payload.retryable is True
    assert terminal_client_update.payload.code == ErrorCode.PROVIDER_ERROR
    assert "gsk_live_SECRET999888" not in terminal_client_update.payload.message
