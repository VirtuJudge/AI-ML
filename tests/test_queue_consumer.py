"""Unit tests for app.queue_consumer (RedisQueueConsumer)."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.backend_client import FakeBackendClient, JobNotFoundError
from app.contracts import (
    ArtifactRef,
    ErrorCode,
    FailedPayload,
    JobType,
    PrimaryQuestion,
    QueueMessage,
    SessionAnalysisCompleted,
    UpdateStatus,
)
from app.pipeline import PitchAnalysisPipeline
from app.queue_consumer import RedisQueueConsumer


def _make_sample_queue_message(job_id: str = "01JTESTJOB00000000000000001") -> QueueMessage:
    payload = {
        "presentation": {
            "artifact_id": "01JTESTART000000000000000001",
            "object_key": "presentations/test.mp4",
            "checksum": "sha256:" + "a" * 64,
            "media_type": "video/mp4",
            "duration_ms": 12000,
        },
        "supporting_documents": [],
        "rubric": {
            "rubric_id": "startup_pitch",
            "version": 1,
        },
        "requested_capabilities": [],
    }
    return QueueMessage(
        schema_version=1,
        job_id=job_id,
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id="01JTESTSESSION00000000000001",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTESTTRACE0000000000000001",
        payload=payload,
    )


@pytest.mark.asyncio
async def test_process_one_message_stores_result_and_updates_backend() -> None:
    """Verify consumer processes job, calls backend update, and writes result to Redis."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_pipeline.analyze_session.return_value = SessionAnalysisCompleted(
        analysis_artifact=ArtifactRef(
            artifact_id="01JTESTART000000000000000002",
            object_key="artifacts/session_analysis.json",
            checksum="sha256:" + "b" * 64,
            schema_version=1,
        ),
        primary_questions=[
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000001",
                text="What are your gross margins?",
                reason="Assesses unit economics.",
                rubric_dimension="market_and_business_model",
                evidence_ids=["ev_speech_001"],
            ),
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000002",
                text="How do you handle scaling bottlenecks?",
                reason="Assesses technical feasibility.",
                rubric_dimension="technology_and_moat",
                evidence_ids=["ev_speech_001"],
            ),
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000003",
                text="What are your validation milestones?",
                reason="Assesses execution milestones.",
                rubric_dimension="execution_and_milestones",
                evidence_ids=["ev_speech_001"],
            ),
        ],
        speaker_labels=["SPEAKER_00"],
        limitations=[],
    )

    mock_redis = AsyncMock()
    mock_backend = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
        result_prefix="test:result:",
    )

    msg = _make_sample_queue_message(job_id="job_abc123")
    raw_json = msg.model_dump_json()

    update = await consumer.process_one_message(raw_json)

    assert update is not None
    assert update.status == UpdateStatus.COMPLETED
    assert len(mock_backend.updates) == 2  # STARTED, then COMPLETED
    assert mock_backend.updates[0].status == UpdateStatus.STARTED
    assert mock_backend.updates[1].status == UpdateStatus.COMPLETED

    # Verify result key was written to Redis with confidential questions redacted
    mock_redis.set.assert_called()
    called_keys = [call.args[0] for call in mock_redis.set.call_args_list]
    assert "test:result:job_abc123" in called_keys
    stored_val = next(
        call.args[1]
        for call in mock_redis.set.call_args_list
        if call.args[0] == "test:result:job_abc123"
    )
    assert "What are your gross margins?" not in stored_val


@pytest.mark.asyncio
async def test_process_one_message_handles_malformed_json() -> None:
    """Verify malformed JSON does not crash consumer and returns None."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_redis = AsyncMock()
    mock_backend = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
    )

    res = await consumer.process_one_message("{invalid json payload")
    assert res is None
    assert len(mock_backend.updates) == 0
    mock_redis.lpush.assert_awaited_once()


@pytest.mark.asyncio
async def test_consumer_run_max_jobs() -> None:
    """Verify consumer run loop stops after processing max_jobs."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_pipeline.analyze_session.return_value = SessionAnalysisCompleted(
        analysis_artifact=ArtifactRef(
            artifact_id="01JTESTART000000000000000002",
            object_key="artifacts/session_analysis.json",
            checksum="sha256:" + "b" * 64,
            schema_version=1,
        ),
        primary_questions=[
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000001",
                text="What are your gross margins?",
                reason="Assesses unit economics.",
                rubric_dimension="market_and_business_model",
                evidence_ids=["ev_speech_001"],
            ),
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000002",
                text="How do you handle scaling bottlenecks?",
                reason="Assesses technical feasibility.",
                rubric_dimension="technology_and_moat",
                evidence_ids=["ev_speech_001"],
            ),
            PrimaryQuestion(
                candidate_id="01JTESTQ0000000000000000003",
                text="What are your validation milestones?",
                reason="Assesses execution milestones.",
                rubric_dimension="execution_and_milestones",
                evidence_ids=["ev_speech_001"],
            ),
        ],
        speaker_labels=[],
        limitations=[],
    )

    msg = _make_sample_queue_message(job_id="job_run_1")
    raw_json = msg.model_dump_json()

    mock_redis = AsyncMock()
    mock_redis.rpoplpush.return_value = None
    # Atomically move one ready message into the pending list.
    mock_redis.blmove.side_effect = [raw_json]

    mock_backend = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
        heartbeat_interval=100.0,  # avoid triggering during test
    )

    processed = await consumer.run(max_jobs=1)
    assert processed == 1


@pytest.mark.asyncio
async def test_heartbeat_payload() -> None:
    """Verify heartbeat writes valid alive payload to Redis."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_redis = AsyncMock()
    mock_backend = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
        heartbeat_key="test:worker:heartbeat",
        heartbeat_interval=0.01,
        heartbeat_ttl=10,
    )

    # Manually run one iteration of heartbeat set
    r = await consumer._get_redis()
    payload = {
        "status": "alive",
        "worker_id": consumer.worker_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "queues": consumer.queue_names,
    }
    await r.set(consumer.heartbeat_key, json.dumps(payload), ex=consumer.heartbeat_ttl)

    mock_redis.set.assert_called_with(
        "test:worker:heartbeat",
        pytest.approx(json.dumps(payload)),
        ex=10,
    )


@pytest.mark.asyncio
async def test_process_one_message_reports_failure_for_malformed_uuid_envelope() -> None:
    """Malformed envelopes with Backend UUID job_id and trc_ trace_id report failure."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_redis = AsyncMock()
    mock_backend = FakeBackendClient()

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
    )

    envelope = {
        "schema_version": 1,
        "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "job_type": "analyze_answr",
        "practice_session_id": "109156be-c4fb-41ea-b1b4-efe1671c5836",
        "analysis_attempt": 1,
        "created_at": "2026-09-10T12:00:00Z",
        "trace_id": "trc_4a7f2c9e8b1d4056a3c7e9f012345678",
        "payload": {},
    }

    res = await consumer.process_one_message(json.dumps(envelope))
    assert res is None
    mock_redis.lpush.assert_not_awaited()
    assert len(mock_backend.updates) == 1
    update = mock_backend.updates[0]
    assert update.status == UpdateStatus.FAILED
    assert update.trace_id == "trc_4a7f2c9e8b1d4056a3c7e9f012345678"
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.MALFORMED_PAYLOAD


@pytest.mark.asyncio
async def test_process_one_message_dead_letter_keeps_safe_ids() -> None:
    """Dead-letter records retain safe raw job_id/trace_id and drop unsafe ones."""
    mock_pipeline = AsyncMock(spec=PitchAnalysisPipeline)
    mock_redis = AsyncMock()
    mock_backend = AsyncMock()
    mock_backend.get_last_update_sequence.side_effect = JobNotFoundError("not found")

    consumer = RedisQueueConsumer(
        pipeline=mock_pipeline,
        backend_client=mock_backend,
        redis_client=mock_redis,
    )

    # 1. UUID job_id not found on backend -> dead-letter keeps UUID and trc_<hex>
    uuid_envelope = {
        "schema_version": 1,
        "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "job_type": "analyze_answr",
        "trace_id": "trc_4a7f2c9e8b1d4056a3c7e9f012345678",
        "payload": {},
    }
    await consumer.process_one_message(json.dumps(uuid_envelope))
    first_record = json.loads(mock_redis.lpush.await_args_list[0].args[1])
    assert first_record["job_id"] == "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
    assert first_record["trace_id"] == "trc_4a7f2c9e8b1d4056a3c7e9f012345678"

    # 2. Safe non-UUID/non-ULID job_id -> dead-letter keeps safe raw job_id
    safe_envelope = {
        "schema_version": 1,
        "job_id": "custom_job_123",
        "job_type": "analyze_answr",
        "trace_id": "trc_custom_1",
        "payload": {},
    }
    await consumer.process_one_message(json.dumps(safe_envelope))
    second_record = json.loads(mock_redis.lpush.await_args_list[1].args[1])
    assert second_record["job_id"] == "custom_job_123"
    assert second_record["trace_id"] == "trc_custom_1"

    # 3. Unsafe job_id -> dead-letter stores None
    unsafe_envelope = {
        "schema_version": 1,
        "job_id": "../bad\nid",
        "job_type": "analyze_answr",
        "trace_id": "https://evil.example?tok=secret",
        "payload": {},
    }
    await consumer.process_one_message(json.dumps(unsafe_envelope))
    third_record = json.loads(mock_redis.lpush.await_args_list[2].args[1])
    assert third_record["job_id"] is None
    assert third_record["trace_id"] is None
