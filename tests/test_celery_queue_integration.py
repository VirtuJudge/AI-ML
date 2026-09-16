"""Tests validating Celery queue integration, consumer task registration, and execution semantics.

Verifies:
- Production queue is 'ai_jobs' with task name 'app.worker.process_job'.
- Celery configuration parameters (acks_late, reject_on_worker_lost, prefetch, etc.).
- Delivery and acknowledgement of backend-formatted Celery tasks.
- Monotonic worker updates from started -> progress -> completed.
- Retry behavior on transient errors and non-retry on validation errors.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.celery_app import celery_app
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    EraseAIDataPayload,
    ErasureCompleted,
    FailedPayload,
    GenerateReportPayload,
    PrimaryQuestion,
    ReportCompleted,
    SessionAnalysisCompleted,
    UpdateStatus,
    WorkerUpdate,
)
from app.stages.common import StageTransientError
from app.worker import (
    _async_process_job_task,
    _run_async,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ai"
if not FIXTURES_DIR.is_dir():
    FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "Backend" / "contracts" / "fixtures" / "ai"


def _load_fixture(filename: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


class MockBackendClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, WorkerUpdate]] = []
        self.cancellation_requested = False

    async def send_update(self, job_id: str, update: WorkerUpdate) -> bool:
        self.updates.append((job_id, update))
        return True

    async def check_cancellation(self, job_id: str) -> bool:
        return self.cancellation_requested

    async def check_backend_reachability(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class LoopBoundBackendClient(MockBackendClient):
    """Model an async client whose connection pool belongs to its first event loop."""

    def __init__(self) -> None:
        super().__init__()
        self.event_loop: asyncio.AbstractEventLoop | None = None

    def _assert_event_loop(self) -> None:
        current_loop = asyncio.get_running_loop()
        if self.event_loop is None:
            self.event_loop = current_loop
        elif self.event_loop is not current_loop:
            raise RuntimeError("Event loop is closed")

    async def send_update(self, job_id: str, update: WorkerUpdate) -> bool:
        self._assert_event_loop()
        return await super().send_update(job_id, update)

    async def check_cancellation(self, job_id: str) -> bool:
        self._assert_event_loop()
        return await super().check_cancellation(job_id)


class MockPipeline:
    def __init__(self, should_fail_transient: bool = False) -> None:
        self.should_fail_transient = should_fail_transient
        self.last_analyzed_session: AnalyzeSessionPayload | None = None

    async def analyze_session(
        self,
        payload: AnalyzeSessionPayload,
        job_id: str,
        backend_client: Any,
        attempt: int = 1,
    ) -> SessionAnalysisCompleted:
        if self.should_fail_transient:
            raise StageTransientError("Groq 503 Service Unavailable")
        self.last_analyzed_session = payload
        if backend_client and hasattr(backend_client, "send_progress"):
            await backend_client.send_progress(job_id, "speech", 0.2, "Transcribing presentation speech")
        return SessionAnalysisCompleted(
            analysis_artifact=ArtifactRef(
                artifact_id="01JEXAMPLE0000000000000051",
                object_key="artifacts/session_analysis.json",
                checksum="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                schema_version=1,
            ),
            primary_questions=[
                PrimaryQuestion(
                    candidate_id="cand_1",
                    text="Question 1?",
                    reason="Reason 1",
                    rubric_dimension="market_and_business_model",
                    evidence_ids=["ev_1"],
                ),
                PrimaryQuestion(
                    candidate_id="cand_2",
                    text="Question 2?",
                    reason="Reason 2",
                    rubric_dimension="technology_and_moat",
                    evidence_ids=["ev_2"],
                ),
                PrimaryQuestion(
                    candidate_id="cand_3",
                    text="Question 3?",
                    reason="Reason 3",
                    rubric_dimension="execution_and_milestones",
                    evidence_ids=["ev_3"],
                ),
            ],
            speaker_labels=["SPEAKER_00"],
            limitations=[],
        )

    async def analyze_answer(
        self,
        payload: AnalyzeAnswerPayload,
        job_id: str,
        backend_client: Any,
        attempt: int = 1,
    ) -> AnswerAnalysisCompleted:
        return AnswerAnalysisCompleted(
            answer_id=payload.answer_id,
            transcript_artifact_id="art_trans_01",
            assessment_artifact_id="art_assess_01",
        )

    async def generate_report(
        self,
        payload: GenerateReportPayload,
        job_id: str,
        backend_client: Any,
        attempt: int = 1,
        practice_session_id: str | None = None,
    ) -> ReportCompleted:
        return ReportCompleted(
            evaluation_artifact=ArtifactRef(
                artifact_id="art_eval_01",
                object_key=f"ai/session/{practice_session_id or '01'}/evaluation.json",
                checksum="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                schema_version=1,
            ),
            report_artifact=ArtifactRef(
                artifact_id="art_rep_01",
                object_key=f"ai/session/{practice_session_id or '01'}/report.md",
                checksum="sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                schema_version=1,
            ),
        )

    async def erase_data(
        self,
        payload: EraseAIDataPayload,
        job_id: str,
        backend_client: Any,
        attempt: int = 1,
    ) -> ErasureCompleted:
        return ErasureCompleted(
            erasure_request_id=payload.erasure_request_id,
            deleted_records=10,
            deleted_objects=2,
        )


def test_celery_app_configuration_invariants():
    """Verify AI Celery app explicitly matches the backend transport contract."""
    conf = celery_app.conf
    assert conf.task_default_queue == "ai_jobs"
    assert conf.task_serializer == "json"
    assert "json" in conf.accept_content
    assert conf.result_serializer == "json"
    assert conf.enable_utc is True
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1


def test_celery_task_registration():
    """Verify backend-owned task name 'app.worker.process_job' is registered in Celery."""
    assert "app.worker.process_job" in celery_app.tasks
    task_func = celery_app.tasks["app.worker.process_job"]
    assert task_func.name == "app.worker.process_job"


def test_sync_celery_boundary_reuses_one_event_loop_across_jobs():
    """Consecutive sync Celery tasks must not invalidate cached async clients."""
    envelope = _load_fixture("job_analyze_session_valid.json")
    client = LoopBoundBackendClient()
    pipeline = MockPipeline()

    first = _run_async(
        _async_process_job_task(None, envelope, pipeline=pipeline, backend_client=client)
    )
    second = _run_async(
        _async_process_job_task(None, envelope, pipeline=pipeline, backend_client=client)
    )

    assert first is not None and first.status == UpdateStatus.COMPLETED
    assert second is not None and second.status == UpdateStatus.COMPLETED


@pytest.mark.asyncio
async def test_celery_task_consumes_backend_analyze_session_fixture():
    """Verify Celery task consumes backend-formatted analyze_session fixture and emits monotonic sequence."""
    envelope = _load_fixture("job_analyze_session_valid.json")
    client = MockBackendClient()
    pipeline = MockPipeline()

    update = await _async_process_job_task(
        task=None,
        envelope=envelope,
        pipeline=pipeline,
        backend_client=client,
    )

    assert update is not None
    assert update.status == UpdateStatus.COMPLETED
    assert len(client.updates) >= 3

    # Check monotonic sequences
    sequences = [u.sequence for _, u in client.updates]
    assert sequences[0] == 1
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)  # strictly increasing

    # Check statuses
    statuses = [u.status for _, u in client.updates]
    assert statuses[0] == UpdateStatus.STARTED
    assert UpdateStatus.PROGRESS in statuses
    assert statuses[-1] == UpdateStatus.COMPLETED


@pytest.mark.asyncio
async def test_celery_task_handles_transient_error_with_retry():
    """Verify transient error triggers Celery task.retry with bounded backoff."""
    envelope = _load_fixture("job_analyze_session_valid.json")
    client = MockBackendClient()
    pipeline = MockPipeline(should_fail_transient=True)

    mock_celery_task = MagicMock()
    mock_celery_task.request.retries = 0
    mock_celery_task.max_retries = 3
    mock_celery_task.retry.side_effect = RuntimeError("CeleryRetryRaised")

    with pytest.raises(RuntimeError, match="CeleryRetryRaised"):
        await _async_process_job_task(
            task=mock_celery_task,
            envelope=envelope,
            pipeline=pipeline,
            backend_client=client,
        )

    # Verify task.retry was called
    mock_celery_task.retry.assert_called_once()
    retry_kwargs = mock_celery_task.retry.call_args[1]
    assert "countdown" in retry_kwargs
    assert 0 <= retry_kwargs["countdown"] <= 60


@pytest.mark.asyncio
async def test_celery_task_non_retryable_validation_error():
    """Verify non-retryable invalid input acknowledges without raising task.retry."""
    invalid_envelope = {"job_id": "01JEXAMPLE_BAD", "trace_id": "trc_bad", "job_type": "invalid_type"}
    client = MockBackendClient()
    pipeline = MockPipeline()

    mock_celery_task = MagicMock()
    mock_celery_task.request.retries = 0

    res = await _async_process_job_task(
        task=mock_celery_task,
        envelope=invalid_envelope,
        pipeline=pipeline,
        backend_client=client,
    )

    # Returned None (acknowledged) and retry was NOT called
    assert res is None
    mock_celery_task.retry.assert_not_called()

    # Sent safe failed update to backend
    assert len(client.updates) == 1
    job_id, update = client.updates[0]
    assert job_id == "01JEXAMPLE_BAD"
    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.retryable is False


@pytest.mark.asyncio
async def test_celery_task_all_job_types_round_trip():
    """Verify all 4 job types complete successfully via Celery worker dispatcher."""
    client = MockBackendClient()
    pipeline = MockPipeline()

    # 1. Analyze Answer
    ans_env = _load_fixture("job_analyze_answer_valid.json")
    res_ans = await _async_process_job_task(None, ans_env, pipeline=pipeline, backend_client=client)
    assert res_ans is not None
    assert res_ans.status == UpdateStatus.COMPLETED

    # 2. Generate Report
    rep_env = _load_fixture("job_generate_report_valid.json")
    res_rep = await _async_process_job_task(None, rep_env, pipeline=pipeline, backend_client=client)
    assert res_rep is not None
    assert res_rep.status == UpdateStatus.COMPLETED

    # 3. Erase AI Data
    era_env = _load_fixture("job_erase_ai_data_valid.json")
    res_era = await _async_process_job_task(None, era_env, pipeline=pipeline, backend_client=client)
    assert res_era is not None
    assert res_era.status == UpdateStatus.COMPLETED
