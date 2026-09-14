"""Unit tests for stage checkpointing and transient retry mechanics."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.stages.checkpoint import get_stage_checkpoint, save_stage_checkpoint
from app.stages.common import StageTransientError, with_transient_retries
from app.storage.local import LocalDiskObjectStorage


class DummyObservationModel(BaseModel):
    """Pydantic model used to verify BaseModel checkpoint serialization."""

    speaker_label: str
    confidence: float
    metrics: dict[str, Any]


@pytest.mark.asyncio
async def test_stage_checkpoint_roundtrip_dict(tmp_path: Path) -> None:
    """Verify save_stage_checkpoint and get_stage_checkpoint roundtrip with a dict payload."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000001"
    stage = "speech"
    checksum = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    data = {
        "transcription": "Welcome to our investor pitch.",
        "word_count": 5,
        "language": "en",
    }

    ref = await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
        data=data,
        schema_version=1,
    )

    clean_hash = checksum.removeprefix("sha256:")
    expected_key = f"ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json"
    assert ref.object_key == expected_key
    assert ref.checksum.startswith("sha256:")

    retrieved = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
        expected_schema_version=1,
    )

    assert retrieved == data


@pytest.mark.asyncio
async def test_stage_checkpoint_roundtrip_basemodel(tmp_path: Path) -> None:
    """Verify save_stage_checkpoint serializes a BaseModel and get_stage_checkpoint recovers it."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000002"
    stage = "audio"
    checksum = "sha256:11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"

    model_instance = DummyObservationModel(
        speaker_label="SPEAKER_00",
        confidence=0.95,
        metrics={"wpm": 145.2, "pitch_hz": 182.4},
    )

    ref = await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
        data=model_instance,
    )

    assert ref.object_key.startswith(f"ai/session/{session_id}/checkpoints/{stage}_")

    retrieved = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
    )

    assert retrieved is not None
    assert retrieved == model_instance.model_dump(mode="json")


@pytest.mark.asyncio
async def test_stage_checkpoint_clean_hash_without_prefix(tmp_path: Path) -> None:
    """Verify input checksums without 'sha256:' prefix are handled cleanly."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000003"
    stage = "vision"
    bare_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

    ref = await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=bare_hash,
        data={"detections": 4},
    )

    assert ref.object_key == f"ai/session/{session_id}/checkpoints/{stage}_{bare_hash}.json"

    retrieved = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=bare_hash,
    )
    assert retrieved == {"detections": 4}


@pytest.mark.asyncio
async def test_get_stage_checkpoint_not_found(tmp_path: Path) -> None:
    """Verify get_stage_checkpoint returns None when the checkpoint does not exist."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    result = await get_stage_checkpoint(
        storage=storage,
        session_id="nonexistent_session",
        stage="speech",
        input_checksum="sha256:0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_stage_checkpoint_mismatched_checksum(tmp_path: Path) -> None:
    """Verify get_stage_checkpoint returns None when requested checksum does not match payload."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000004"
    stage = "documents"
    actual_checksum = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    wrong_checksum = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=actual_checksum,
        data={"pages": 10},
    )

    # Key won't match wrong_checksum (file not found)
    missing_result = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=wrong_checksum,
    )
    assert missing_result is None

    # Now create a file at actual_checksum key whose payload contains a conflicting input_checksum
    clean_actual = actual_checksum.removeprefix("sha256:")
    tampered_key = f"ai/session/{session_id}/checkpoints/{stage}_{clean_actual}.json"
    tampered_payload = {
        "stage": stage,
        "schema_version": 1,
        "producer_version": "ai-ml/0.1.0",
        "input_checksum": "sha256:mismatched_inside_payload",
        "created_at": "2026-09-14T12:00:00Z",
        "data": {"pages": 10},
    }
    await storage.upload_json(tampered_key, tampered_payload)

    mismatch_result = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=actual_checksum,
    )
    assert mismatch_result is None


@pytest.mark.asyncio
async def test_get_stage_checkpoint_mismatched_schema_version(tmp_path: Path) -> None:
    """Verify get_stage_checkpoint returns None when schema_version does not match expected."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000005"
    stage = "scoring"
    checksum = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

    await save_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
        data={"overall_score": 0.85},
        schema_version=1,
    )

    # Expecting schema_version 2 when stored is 1
    result = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
        expected_schema_version=2,
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_stage_checkpoint_corrupt_payload(tmp_path: Path) -> None:
    """Verify get_stage_checkpoint returns None when payload is corrupt or malformed."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JSESSION0000000000000006"
    stage = "evidence"
    checksum = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    clean_hash = checksum.removeprefix("sha256:")
    key = f"ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json"

    # Case 1: Corrupt JSON bytes on disk
    file_path = tmp_path / key
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("{corrupt-json-not-valid", encoding="utf-8")

    result = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
    )
    assert result is None

    # Case 2: Valid JSON but not a dict (e.g. list)
    file_path.write_text("['not', 'a', 'dict']", encoding="utf-8")
    result_non_dict = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
    )
    assert result_non_dict is None

    # Case 3: Dict payload but missing or non-dict "data" field
    file_path.write_text(
        '{"stage": "evidence", "schema_version": 1, "input_checksum": "'
        + checksum
        + '", "data": "not-a-dict"}',
        encoding="utf-8",
    )
    result_bad_data = await get_stage_checkpoint(
        storage=storage,
        session_id=session_id,
        stage=stage,
        input_checksum=checksum,
    )
    assert result_bad_data is None


@pytest.mark.asyncio
async def test_with_transient_retries_succeeds_first_attempt() -> None:
    """Verify with_transient_retries returns result immediately on first attempt."""
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        return "first_attempt_success"

    result = await with_transient_retries(op, stage_name="speech")
    assert result == "first_attempt_success"
    assert calls == 1


@pytest.mark.asyncio
async def test_with_transient_retries_retries_and_succeeds_second_attempt() -> None:
    """Verify with_transient_retries catches transient error and succeeds on second attempt."""
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StageTransientError("Temporary Groq 503 Service Unavailable")
        return "second_attempt_success"

    result = await with_transient_retries(
        op,
        stage_name="speech",
        max_attempts=3,
        base_delay=0.01,
    )
    assert result == "second_attempt_success"
    assert calls == 2


@pytest.mark.asyncio
async def test_with_transient_retries_retries_timeout_and_connection_errors() -> None:
    """Verify TimeoutError and ConnectionError are retried by default."""
    calls = 0

    async def op_timeout() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("Read timed out")
        return "timeout_recovered"

    result_timeout = await with_transient_retries(
        op_timeout,
        stage_name="speech_timeout",
        max_attempts=3,
        base_delay=0.01,
    )
    assert result_timeout == "timeout_recovered"
    assert calls == 2

    # Test ConnectionError
    conn_calls = 0

    async def op_conn() -> str:
        nonlocal conn_calls
        conn_calls += 1
        if conn_calls == 1:
            raise ConnectionError("Remote host reset connection")
        return "conn_recovered"

    result_conn = await with_transient_retries(
        op_conn,
        stage_name="speech_conn",
        max_attempts=3,
        base_delay=0.01,
    )
    assert result_conn == "conn_recovered"
    assert conn_calls == 2


@pytest.mark.asyncio
async def test_with_transient_retries_exhausts_max_attempts() -> None:
    """Verify with_transient_retries raises after exhausting max_attempts."""
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise StageTransientError("Persistent external service outage")

    with pytest.raises(StageTransientError, match="Persistent external service outage"):
        await with_transient_retries(
            op,
            stage_name="judge_panel",
            max_attempts=3,
            base_delay=0.01,
        )

    assert calls == 3


@pytest.mark.asyncio
async def test_with_transient_retries_immediately_raises_non_retryable() -> None:
    """Verify with_transient_retries immediately raises non-retryable exceptions without retry."""
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("Invalid configuration parameter")

    with pytest.raises(ValueError, match="Invalid configuration parameter"):
        await with_transient_retries(
            op,
            stage_name="audio",
            max_attempts=3,
            base_delay=0.01,
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_with_transient_retries_backoff_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify exponential backoff calculation and sleep calls."""
    recorded_delays: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        recorded_delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise StageTransientError("Transient glitch")
        return "finally_done"

    result = await with_transient_retries(
        op,
        stage_name="backoff_test",
        max_attempts=3,
        base_delay=1.0,
    )

    assert result == "finally_done"
    assert calls == 3
    assert len(recorded_delays) == 2

    # Attempt 1 retry backoff: base_delay * (2 ** 0) + jitter [0, 0.1] -> [1.0, 1.1]
    assert 1.0 <= recorded_delays[0] <= 1.1
    # Attempt 2 retry backoff: base_delay * (2 ** 1) + jitter [0, 0.1] -> [2.0, 2.1]
    assert 2.0 <= recorded_delays[1] <= 2.1
