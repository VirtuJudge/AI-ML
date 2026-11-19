"""Tests for owned per-job temp media cleanup, partial failure, restart, and erasure."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import __main__ as app_main
from app.backend_client import FakeBackendClient
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AssetInput,
    AudioAssetInput,
    EraseAIDataPayload,
    JobCancelledError,
    JobType,
    QueueMessage,
    RubricRef,
    UpdateStatus,
)
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.types import DocumentChunk
from app.stages.answers import run_answer_speech_stage
from app.stages.media import (
    JOB_TEMP_ROOT_ENV,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    MediaNormalizationError,
    cleanup_stale_worker_temp_dirs,
    get_media_temp_root,
    normalize_media,
    owned_job_temp_dir,
    split_media,
)
from app.stages.speech import run_speech_stage
from app.storage.base import compute_file_sha256
from app.storage.local import LocalDiskObjectStorage
from app.worker import _on_celery_worker_init, process_job


def _write_silent_wav(path: Path, duration_ms: int = 1000, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int((duration_ms / 1000.0) * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _list_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


@pytest.mark.asyncio
async def test_analyze_answer_cleans_owned_temp_dir_on_success_failure_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answer normalization uses an owned per-job dir cleaned on success, failure, and cancel."""
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    source_file = tmp_path / "caller_answer.webm"
    source_file.write_bytes(b"synthetic webm bytes")
    source_checksum = compute_file_sha256(source_file)

    observed_wav_paths: list[Path] = []

    async def fake_exec(*cmd: str, **_kwargs: Any) -> AsyncMock:
        out_path = Path(cmd[-1])
        observed_wav_paths.append(out_path)
        assert out_path.is_relative_to(temp_root)
        assert "vj_answer_" in out_path.parent.name
        _write_silent_wav(out_path, duration_ms=1000)
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    pipeline = FakePipeline(object_storage=storage)
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JROUND000000000000000001",
        question_id="01JQUESTION000000000000001",
        answer_id="01JANSWER00000000000000001",
        answered_by="01JUSER0000000000000000001",
        practice_session_id="01JSESSION0000000000000001",
        audio=AudioAssetInput(
            artifact_id="01JANSWERAUDIO000000000001",
            object_key=str(source_file),
            checksum=source_checksum,
            media_type="audio/webm",
            duration_ms=1000,
        ),
        remaining_follow_ups=1,
    )

    # 1. Success path
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        res = await pipeline.analyze_answer(payload, job_id="job-answer-ok")

    assert len(observed_wav_paths) == 1
    assert not observed_wav_paths[0].exists()
    assert not observed_wav_paths[0].parent.exists()
    assert _list_files(temp_root) == []
    assert source_file.is_file()

    transcript = await storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/transcript.json"
    )
    assert transcript["source_artifact_ids"] == [payload.audio.artifact_id]
    assessment = await storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/assessment.json"
    )
    assert assessment["source_artifact_ids"] == [res.transcript_artifact_id]

    # 2. Failure path (speech provider raises after normalization wrote WAV)
    class FailingSpeechProvider(FakeSpeechProvider):
        async def transcribe(self, audio_path: Path, **kwargs: Any) -> Any:
            raise RuntimeError("speech failure")

    failing_pipeline = FakePipeline(
        speech_provider=FailingSpeechProvider(),
        object_storage=storage,
    )
    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        pytest.raises(RuntimeError, match="speech failure"),
    ):
        await failing_pipeline.analyze_answer(payload, job_id="job-answer-fail")

    assert len(observed_wav_paths) == 2
    assert not observed_wav_paths[1].exists()
    assert not observed_wav_paths[1].parent.exists()
    assert _list_files(temp_root) == []
    assert source_file.is_file()

    # 3. Cancellation path (cancelled after transcription, before assessment)
    class CancelAfterFirstCheckBackend(FakeBackendClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def check_cancellation(self, job_id: str) -> bool:
            self.calls += 1
            return self.calls >= 2

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        pytest.raises(JobCancelledError),
    ):
        await pipeline.analyze_answer(
            payload,
            job_id="job-answer-cancel",
            backend_client=CancelAfterFirstCheckBackend(),
        )

    assert len(observed_wav_paths) == 3
    assert not observed_wav_paths[2].exists()
    assert not observed_wav_paths[2].parent.exists()
    assert _list_files(temp_root) == []
    assert source_file.is_file()


@pytest.mark.asyncio
async def test_analyze_session_cleans_owned_temp_dir_on_success_failure_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session split_media uses an owned per-job dir cleaned on success, failure, and cancel."""
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    source_video = tmp_path / "caller_pitch.mp4"
    source_video.write_bytes(b"synthetic mp4 bytes")
    source_checksum = compute_file_sha256(source_video)

    observed_outputs: list[tuple[Path, Path]] = []

    async def fake_split_exec(*cmd: str, **_kwargs: Any) -> AsyncMock:
        # split_media passes ... <target_audio> -an -c:v copy <target_video>
        audio_out = Path(cmd[-5])
        video_out = Path(cmd[-1])
        observed_outputs.append((audio_out, video_out))
        assert audio_out.parent == video_out.parent
        assert audio_out.is_relative_to(temp_root)
        assert "vj_session_" in audio_out.parent.name
        _write_silent_wav(audio_out, duration_ms=1000)
        video_out.write_bytes(b"synthetic vision video")
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    pipeline = FakePipeline(object_storage=storage)
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRES0000000000000000001",
            object_key=str(source_video),
            checksum=source_checksum,
            media_type="video/mp4",
            duration_ms=1000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "vision", "audio", "questions"],
        practice_session_id="01JSESS0000000000000000001",
    )

    # 1. Success path
    with patch("asyncio.create_subprocess_exec", side_effect=fake_split_exec):
        res = await pipeline.analyze_session(payload, job_id="job-sess-ok")

    assert len(res.primary_questions) == 3
    assert len(observed_outputs) == 1
    assert not observed_outputs[0][0].exists()
    assert not observed_outputs[0][1].exists()
    assert not observed_outputs[0][0].parent.exists()
    assert _list_files(temp_root) == []
    assert source_video.is_file()

    # 2. Failure path (speech provider fails after split_media)
    class FailingSpeechProvider(FakeSpeechProvider):
        async def transcribe(self, audio_path: Path, **kwargs: Any) -> Any:
            raise RuntimeError("session speech failure")

    failing_pipeline = FakePipeline(
        speech_provider=FailingSpeechProvider(),
        object_storage=storage,
    )
    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_split_exec),
        pytest.raises(RuntimeError, match="session speech failure"),
    ):
        await failing_pipeline.analyze_session(payload, job_id="job-sess-fail")

    assert len(observed_outputs) == 2
    assert not observed_outputs[1][0].exists()
    assert not observed_outputs[1][1].exists()
    assert _list_files(temp_root) == []
    assert source_video.is_file()

    # 3. Cancellation path via worker process_job
    class CancelAtVisionBackend(FakeBackendClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def check_cancellation(self, job_id: str) -> bool:
            self.calls += 1
            return self.calls >= 3

    msg = QueueMessage(
        schema_version=1,
        job_id="01JJOBSESSIONCANCEL0000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=payload.practice_session_id,
        analysis_attempt=1,
        created_at="2026-10-15T12:00:00Z",
        trace_id="01JTRACE000000000000000001",
        payload=payload.model_dump(mode="json"),
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_split_exec):
        update = await process_job(msg, pipeline, backend_client=CancelAtVisionBackend())

    assert update.status == UpdateStatus.CANCELLED
    assert len(observed_outputs) == 3
    assert not observed_outputs[2][0].exists()
    assert not observed_outputs[2][1].exists()
    assert _list_files(temp_root) == []
    assert source_video.is_file()


@pytest.mark.asyncio
async def test_partial_normalization_and_split_failures_leave_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial ffmpeg/header failures in normalize_media and split_media clean partial files."""
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    source_media = tmp_path / "source.mp4"
    source_media.write_bytes(b"source media bytes")

    # 1. normalize_media writes partial file then ffmpeg exits non-zero
    async def partial_norm_fail(*cmd: str, **_kwargs: Any) -> AsyncMock:
        Path(cmd[-1]).write_bytes(b"partial corrupt wav")
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"truncated stream"))
        return proc

    with (
        patch("asyncio.create_subprocess_exec", side_effect=partial_norm_fail),
        pytest.raises(FFmpegExecutionError),
    ):
        await normalize_media(source_media)
    assert _list_files(temp_root) == []
    assert source_media.is_file()

    # 2. normalize_media writes corrupt WAV and exits 0 -> get_wav_metadata fails
    async def corrupt_wav_zero_exit(*cmd: str, **_kwargs: Any) -> AsyncMock:
        Path(cmd[-1]).write_bytes(b"not a valid wav header")
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    explicit_out = temp_root / "explicit_partial.wav"
    with (
        patch("asyncio.create_subprocess_exec", side_effect=corrupt_wav_zero_exit),
        pytest.raises(MediaNormalizationError),
    ):
        await normalize_media(source_media, output_path=explicit_out)
    assert not explicit_out.exists()
    assert _list_files(temp_root) == []

    # 3. normalize_media FFmpegNotFoundError cleans temp file
    with (
        patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no ffmpeg")),
        pytest.raises(FFmpegNotFoundError),
    ):
        await normalize_media(source_media)
    assert _list_files(temp_root) == []

    # 4. Caller-owned source file is never deleted even if passed as output_path
    with (
        patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no ffmpeg")),
        pytest.raises(FFmpegNotFoundError),
    ):
        await normalize_media(source_media, output_path=source_media)
    assert source_media.is_file()

    # 5. split_media partial failure cleans up BOTH audio and video temp outputs
    async def partial_split_fail(*cmd: str, **_kwargs: Any) -> AsyncMock:
        Path(cmd[-5]).write_bytes(b"partial audio")
        Path(cmd[-1]).write_bytes(b"partial video")
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"mux error"))
        return proc

    with (
        patch("asyncio.create_subprocess_exec", side_effect=partial_split_fail),
        pytest.raises(FFmpegExecutionError),
    ):
        await split_media(source_media)
    assert _list_files(temp_root) == []
    assert source_media.is_file()

    # 6. split_media audio-only fallback where normalize_media fails cleans BOTH audio and video
    async def audio_only_then_norm_fail(*cmd: str, **_kwargs: Any) -> AsyncMock:
        proc = AsyncMock()
        proc.returncode = 1
        if "-c:v" in cmd:
            Path(cmd[-5]).write_bytes(b"empty audio stub")
            Path(cmd[-1]).write_bytes(b"empty video stub")
            proc.communicate = AsyncMock(
                return_value=(b"", b"Output file does not contain any stream")
            )
        else:
            Path(cmd[-1]).write_bytes(b"partial fallback wav")
            proc.communicate = AsyncMock(return_value=(b"", b"fallback decode error"))
        return proc

    with (
        patch("asyncio.create_subprocess_exec", side_effect=audio_only_then_norm_fail),
        pytest.raises(FFmpegExecutionError),
    ):
        await split_media(source_media)
    assert _list_files(temp_root) == []
    assert source_media.is_file()


@pytest.mark.asyncio
async def test_direct_stage_calls_clean_temp_wav_without_deleting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct stage calls clean temp WAVs when output_wav_path=None."""
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    source_audio = tmp_path / "direct_source.mp4"
    source_audio.write_bytes(b"source bytes")

    async def fake_exec(*cmd: str, **_kwargs: Any) -> AsyncMock:
        _write_silent_wav(Path(cmd[-1]), duration_ms=1000)
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        ans_res = await run_answer_speech_stage(
            source_audio,
            FakeSpeechProvider(),
            skip_normalization=False,
            strict_timestamps=False,
        )
        assert ans_res.transcript.full_text != ""
        assert _list_files(temp_root) == []
        assert source_audio.is_file()

        speech_res = await run_speech_stage(
            source_audio,
            FakeSpeechProvider(),
            FakeDiarizationProvider(),
            skip_normalization=False,
            strict_timestamps=False,
        )
        assert speech_res.transcription.full_text != ""
        assert _list_files(temp_root) == []
        assert source_audio.is_file()


def test_worker_restart_cleanup_removes_orphaned_temp_and_scratch_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker startup cleanup removes stale job temp dirs and scratch dirs from prior crashes."""
    monkeypatch.chdir(tmp_path)
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    # Simulate orphaned files from a crashed worker
    with owned_job_temp_dir(
        job_kind="session",
        session_id="01JCRASHEDSESS000000000001",
        asset_version_id="01JCRASHEDART0000000000001",
    ) as leaked_dir:
        (leaked_dir / "presentation_normalized.wav").write_bytes(b"orphaned wav")
        # Copy path before context exit won't apply; simulate crash by creating sibling dir
        crashed_dir = leaked_dir.parent / "vj_answer_crashed_123"
        crashed_dir.mkdir(parents=True)
        (crashed_dir / "answer_normalized.wav").write_bytes(b"crashed wav")

    stale_media = tmp_path / ".storage" / "temp_media" / "01JCRASHEDART0000000000001"
    stale_media.mkdir(parents=True)
    (stale_media / "raw.mp4").write_bytes(b"stale downloaded media")

    stale_reports = tmp_path / ".storage" / "temp_reports"
    stale_reports.mkdir(parents=True)
    (stale_reports / "report.md").write_text("stale report", encoding="utf-8")

    caller_source = tmp_path / "caller_owned_source.mp4"
    caller_source.write_bytes(b"caller source bytes")

    removed = cleanup_stale_worker_temp_dirs()
    assert removed >= 3
    assert _list_files(temp_root) == []
    assert _list_files(tmp_path / ".storage" / "temp_media") == []
    assert _list_files(tmp_path / ".storage" / "temp_reports") == []
    assert caller_source.is_file()

    # Verify Celery worker_init hook triggers cleanup
    orphan = get_media_temp_root() / "vj_orphan"
    orphan.mkdir(parents=True)
    (orphan / "leftover.wav").write_bytes(b"leftover")
    _on_celery_worker_init()
    assert not orphan.exists()

    # Verify app.__main__.main() runs cleanup before starting Celery worker
    orphan2 = get_media_temp_root() / "vj_orphan2"
    orphan2.mkdir(parents=True)
    (orphan2 / "leftover2.wav").write_bytes(b"leftover2")

    mock_worker = MagicMock()
    with (
        patch("app.__main__.build_pipeline", new_callable=AsyncMock, return_value=MagicMock()),
        patch("app.celery_app.celery_app.Worker", return_value=mock_worker),
    ):
        app_main.main()

    assert not orphan2.exists()
    mock_worker.start.assert_called_once()


@pytest.mark.asyncio
async def test_explicit_erasure_removes_session_answer_asset_and_orphaned_job_temp_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #65 repro: answer normalization + explicit erasure leaves zero temp waveforms."""
    monkeypatch.chdir(tmp_path)
    temp_root = tmp_path / "sys_tmp"
    temp_root.mkdir()
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    document_store = FakeDocumentStore()
    pipeline = FakePipeline(object_storage=storage, document_store=document_store)

    source_audio = tmp_path / "answer_input.webm"
    source_audio.write_bytes(b"synthetic webm answer")
    uploaded = await storage.upload_file("answers/team/ans.webm", source_audio)

    session_id = "01JSESSISSUE65000000000001"
    answer_id = "01JANSISSUE650000000000001"
    asset_version_id = "01JARTISSUE650000000000001"

    async def fake_exec(*cmd: str, **_kwargs: Any) -> AsyncMock:
        _write_silent_wav(Path(cmd[-1]), duration_ms=1000)
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    answer_job = AnalyzeAnswerPayload(
        qa_round_id="01JROUNDISSUE6500000000001",
        question_id="01JQUESTISSUE6500000000001",
        answer_id=answer_id,
        answered_by="01JUSERISSUE65000000000001",
        practice_session_id=session_id,
        audio=AudioAssetInput(
            artifact_id=asset_version_id,
            object_key="answers/team/ans.webm",
            checksum=uploaded.checksum,
            media_type="audio/webm",
            duration_ms=1000,
        ),
        remaining_follow_ups=1,
    )

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await pipeline.analyze_answer(answer_job, job_id="job-issue65-repro")

    # Normalized waveform must not survive answer processing
    assert _list_files(temp_root) == []

    # Also simulate a crashed prior job dir + unrelated session dir + document chunks
    crashed_target = get_media_temp_root() / (f"vj_answer_{owned_job_temp_dir.__name__}_")
    # Create via owned_job_temp_dir tag convention by simulating crash before cleanup
    root = get_media_temp_root()
    root.mkdir(parents=True, exist_ok=True)
    with patch("app.stages.media.shutil.rmtree") as mock_rmtree:
        # Bypass cleanup inside context manager to simulate SIGKILL crash
        mock_rmtree.return_value = None
        with owned_job_temp_dir(
            job_kind="session",
            session_id=session_id,
            asset_version_id=asset_version_id,
        ) as crashed_job_dir:
            (crashed_job_dir / "presentation_normalized.wav").write_bytes(b"crashed wav")
        with owned_job_temp_dir(
            job_kind="session",
            session_id="01JUNRELATEDSESS0000000001",
            asset_version_id="01JUNRELATEDART00000000001",
        ) as unrelated_job_dir:
            (unrelated_job_dir / "presentation_normalized.wav").write_bytes(b"unrelated wav")
    assert not crashed_target.exists()

    await document_store.store_chunks(
        session_id,
        [
            DocumentChunk(
                chunk_id="c1",
                asset_version_id=asset_version_id,
                page_or_slide=1,
                text="synthetic chunk",
            )
        ],
    )

    erasure_result = await pipeline.erase_data(
        EraseAIDataPayload(
            erasure_request_id="01JERASEISSUE6500000000001",
            scope="practice_session",
            scope_id=session_id,
            practice_session_ids=[session_id],
            answer_ids=[answer_id],
            asset_version_ids=[asset_version_id],
        )
    )

    assert erasure_result.deleted_objects >= 2
    assert erasure_result.deleted_records == 1
    assert not crashed_job_dir.exists()
    assert not (tmp_path / ".storage" / "temp_media" / asset_version_id).exists()
    assert await storage.list_objects(f"ai/session/{session_id}/") == []
    # Unrelated session's dir and caller-owned source file remain untouched
    assert unrelated_job_dir.exists()
    assert source_audio.is_file()


def test_startup_cleanup_only_removes_worker_owned_job_dirs(tmp_path: Path) -> None:
    """A shared AI_MEDIA_TEMP_DIR keeps entries the worker did not create."""
    from app.stages.media import cleanup_stale_worker_temp_dirs

    root = tmp_path / "shared_tmp"
    owned = root / "vj_session_s_abc_tmp123"
    foreign_dir = root / "someone_else"
    foreign_file = root / "notes.txt"
    owned.mkdir(parents=True)
    (owned / "presentation_normalized.wav").write_bytes(b"x")
    foreign_dir.mkdir()
    foreign_file.write_text("keep", encoding="utf-8")

    removed = cleanup_stale_worker_temp_dirs(base_dir=root, scratch_dirs=())

    assert removed == 1
    assert not owned.exists()
    assert foreign_dir.is_dir()
    assert foreign_file.read_text(encoding="utf-8") == "keep"
