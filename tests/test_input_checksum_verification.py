"""Tests for input SHA-256 checksum verification across all pipeline ingestion paths."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.backend_client import FakeBackendClient
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AssetInput,
    AudioAssetInput,
    ChecksumMismatchError,
    ErrorCode,
    FailedPayload,
    GenerateReportPayload,
    JobType,
    QueueMessage,
    RubricRef,
    UpdateStatus,
)
from app.pipeline import FakePipeline
from app.providers.fake_documents import FakeDocumentProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeSpeechProvider
from app.providers.types import DocumentChunk, TranscriptionResult
from app.stages.media import MediaNormalizationResult, MediaSplitResult
from app.stages.report_loader import (
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
    load_report_evidence,
)
from app.storage.base import compute_file_sha256
from app.storage.local import LocalDiskObjectStorage
from app.worker import _async_process_job_task, process_job

WRONG_SHA256 = "sha256:" + "f" * 64


class SpySpeechProvider(FakeSpeechProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Path] = []

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self.calls.append(audio_path)
        return await super().transcribe(audio_path)


class SpyDocumentProvider(FakeDocumentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Path] = []

    async def extract_and_embed(
        self, doc_path: Path, *, asset_version_id: str = ""
    ) -> list[DocumentChunk]:
        self.calls.append(doc_path)
        return await super().extract_and_embed(doc_path, asset_version_id=asset_version_id)


class SpyJudgeProvider(FakeJudgeModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.question_calls = 0
        self.answer_calls = 0
        self.report_calls = 0

    async def generate_questions(self, *args: Any, **kwargs: Any) -> Any:
        self.question_calls += 1
        return await super().generate_questions(*args, **kwargs)

    async def assess_answer(self, *args: Any, **kwargs: Any) -> Any:
        self.answer_calls += 1
        return await super().assess_answer(*args, **kwargs)

    async def generate_report_feedback(self, *args: Any, **kwargs: Any) -> Any:
        self.report_calls += 1
        return await super().generate_report_feedback(*args, **kwargs)


def _mock_split_result(audio_path: Path) -> MediaSplitResult:
    return MediaSplitResult(
        audio_path=audio_path,
        video_path=None,
        duration_ms=15000,
        sample_rate=16000,
        channels=1,
    )


def _mock_norm_result(output_path: Path) -> MediaNormalizationResult:
    return MediaNormalizationResult(
        output_path=output_path,
        duration_ms=15000,
        sample_rate=16000,
        channels=1,
    )


@pytest.mark.asyncio
async def test_analyze_session_local_matching_checksum_succeeds(tmp_path: Path) -> None:
    """Local presentation file with matching SHA-256 proceeds through normalization and stages."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    local_media = tmp_path / "pitch.mp4"
    local_media.write_bytes(b"valid presentation bytes")
    checksum = compute_file_sha256(local_media).upper().replace("SHA256:", "sha256:")

    spy_speech = SpySpeechProvider()
    pipeline = FakePipeline(speech_provider=spy_speech, object_storage=storage)
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRESLOCALMATCH000000001",
            object_key=str(local_media),
            checksum=checksum,
            media_type="video/mp4",
            duration_ms=15000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
        practice_session_id="01JSESSLOCALMATCH000000001",
    )

    with patch("app.pipeline.split_media", return_value=_mock_split_result(local_media)) as split:
        result = await pipeline.analyze_session(payload)

    assert split.call_count == 1
    assert len(spy_speech.calls) == 1
    assert len(result.primary_questions) == 3


@pytest.mark.asyncio
async def test_analyze_session_local_mismatch_fails_before_normalization_and_providers(
    tmp_path: Path,
) -> None:
    """A01 reproduction: local bytes differing from declared checksum fail before demux/models."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    local_media = tmp_path / "pitch.mp4"
    local_media.write_bytes(b"corrupted presentation bytes")
    assert compute_file_sha256(local_media) != WRONG_SHA256

    spy_speech = SpySpeechProvider()
    spy_judge = SpyJudgeProvider()
    pipeline = FakePipeline(
        speech_provider=spy_speech,
        judge_provider=spy_judge,
        object_storage=storage,
    )
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRESLOCALBAD00000000001",
            object_key=str(local_media),
            checksum=WRONG_SHA256,
            media_type="video/mp4",
            duration_ms=15000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
        practice_session_id="01JSESSLOCALBAD00000000001",
    )

    with (
        patch("app.pipeline.split_media") as mock_split,
        patch("app.pipeline.get_stage_checkpoint") as mock_ckpt,
        pytest.raises(ChecksumMismatchError),
    ):
        await pipeline.analyze_session(payload, attempt=2)

    mock_split.assert_not_called()
    mock_ckpt.assert_not_called()
    assert len(spy_speech.calls) == 0
    assert spy_judge.question_calls == 0
    assert await storage.list_objects("ai/session/01JSESSLOCALBAD00000000001/") == []


@pytest.mark.asyncio
async def test_analyze_session_downloaded_mismatch_and_changed_object_behind_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed object behind an existing key fails before checkpoint reuse or normalization."""
    monkeypatch.chdir(tmp_path)
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    source = tmp_path / "v1.mp4"
    source.write_bytes(b"original presentation video v1")
    uploaded_v1 = await storage.upload_file("uploads/team/session/pitch.mp4", source)

    spy_speech = SpySpeechProvider()
    pipeline = FakePipeline(speech_provider=spy_speech, object_storage=storage)
    session_id = "01JSESSCHANGEDOBJ000000001"
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRESCHANGED000000000001",
            object_key="uploads/team/session/pitch.mp4",
            checksum=uploaded_v1.checksum,
            media_type="video/mp4",
            duration_ms=15000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
        practice_session_id=session_id,
    )

    # Attempt 1 succeeds and writes checkpoints for uploaded_v1.checksum
    with patch("app.pipeline.split_media", return_value=_mock_split_result(source)):
        await pipeline.analyze_session(payload, attempt=1)
    assert len(spy_speech.calls) == 1

    # Replace the stored bytes behind the same object key
    replacement = tmp_path / "v2.mp4"
    replacement.write_bytes(b"tampered presentation video v2")
    await storage.upload_file("uploads/team/session/pitch.mp4", replacement)

    # Attempt 2 with original manifest checksum must fail BEFORE checkpoint reuse or split_media
    with (
        patch("app.pipeline.split_media") as mock_split,
        patch("app.pipeline.get_stage_checkpoint") as mock_ckpt,
        pytest.raises(ChecksumMismatchError),
    ):
        await pipeline.analyze_session(payload, attempt=2)

    mock_split.assert_not_called()
    mock_ckpt.assert_not_called()
    assert len(spy_speech.calls) == 1


@pytest.mark.asyncio
async def test_analyze_session_supporting_document_mismatch_fails_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local and downloaded supporting-document mismatches fail before parsing or model calls."""
    monkeypatch.chdir(tmp_path)
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")

    local_doc = tmp_path / "local_deck.pdf"
    local_doc.write_bytes(b"%PDF-1.4 local content")

    spy_speech = SpySpeechProvider()
    spy_docs = SpyDocumentProvider()
    pipeline = FakePipeline(
        speech_provider=spy_speech,
        document_provider=spy_docs,
        object_storage=storage,
    )

    local_mismatch_payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRESDOCMISMATCH00000001",
            object_key="uploads/missing_synthetic.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01JDOCLOCALBAD000000000001",
                object_key=str(local_doc),
                checksum=WRONG_SHA256,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "documents", "questions"],
        practice_session_id="01JSESSDOCLOCALBAD00000001",
    )

    with pytest.raises(ChecksumMismatchError):
        await pipeline.analyze_session(local_mismatch_payload)

    assert len(spy_speech.calls) == 0
    assert len(spy_docs.calls) == 0

    # Now test changed object behind an existing remote document key
    doc_v1 = tmp_path / "doc_v1.pdf"
    doc_v1.write_bytes(b"%PDF-1.4 original deck")
    doc_v1_ref = await storage.upload_file("uploads/team/deck.pdf", doc_v1)

    doc_v2 = tmp_path / "doc_v2.pdf"
    doc_v2.write_bytes(b"%PDF-1.4 replaced deck")
    await storage.upload_file("uploads/team/deck.pdf", doc_v2)

    remote_mismatch_payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JPRESDOCMISMATCH00000002",
            object_key="uploads/missing_synthetic.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01JDOCREMOTEBAD00000000001",
                object_key="uploads/team/deck.pdf",
                checksum=doc_v1_ref.checksum,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "documents", "questions"],
        practice_session_id="01JSESSDOCREMOTEBAD0000001",
    )

    with pytest.raises(ChecksumMismatchError):
        await pipeline.analyze_session(remote_mismatch_payload)

    assert len(spy_speech.calls) == 0
    assert len(spy_docs.calls) == 0


@pytest.mark.asyncio
async def test_analyze_answer_checksum_verification_local_downloaded_and_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answer audio verifies SHA-256 for local, downloaded, and changed-object inputs."""
    monkeypatch.chdir(tmp_path)
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")

    local_audio = tmp_path / "answer_local.wav"
    local_audio.write_bytes(b"RIFF answer audio bytes")
    local_checksum = compute_file_sha256(local_audio)

    spy_speech = SpySpeechProvider()
    spy_judge = SpyJudgeProvider()
    pipeline = FakePipeline(
        speech_provider=spy_speech,
        judge_provider=spy_judge,
        object_storage=storage,
    )

    def _answer_job(object_key: str, checksum: str, answer_id: str) -> AnalyzeAnswerPayload:
        return AnalyzeAnswerPayload(
            qa_round_id="01JQAROUND0000000000000001",
            question_id="01JQUESTION000000000000001",
            answer_id=answer_id,
            answered_by="01JUSER0000000000000000001",
            practice_session_id="01JSESSANSWER0000000000001",
            audio=AudioAssetInput(
                artifact_id="01JANSWERAUDIO000000000001",
                object_key=object_key,
                checksum=checksum,
                media_type="audio/wav",
                duration_ms=15000,
            ),
            remaining_follow_ups=1,
        )

    # 1. Local matching succeeds
    norm_out = tmp_path / "norm.wav"
    with patch("app.stages.answers.normalize_media", return_value=_mock_norm_result(norm_out)):
        res_ok = await pipeline.analyze_answer(
            _answer_job(str(local_audio), local_checksum, "01JANS_LOCAL_OK")
        )
    assert res_ok.answer_id == "01JANS_LOCAL_OK"
    assert len(spy_speech.calls) == 1

    # 2. Local mismatch fails before normalization or transcription
    with (
        patch("app.stages.answers.normalize_media") as mock_norm,
        pytest.raises(ChecksumMismatchError),
    ):
        await pipeline.analyze_answer(
            _answer_job(str(local_audio), WRONG_SHA256, "01JANS_LOCAL_BAD")
        )
    mock_norm.assert_not_called()
    assert len(spy_speech.calls) == 1
    assert (
        await storage.list_objects(
            "ai/session/01JSESSANSWER0000000000001/answers/01JANS_LOCAL_BAD/"
        )
        == []
    )

    # 3. Downloaded matching succeeds, then changed object behind same key fails
    uploaded_ref = await storage.upload_file("answers/team/ans.wav", local_audio)
    with patch("app.stages.answers.normalize_media", return_value=_mock_norm_result(norm_out)):
        res_dl = await pipeline.analyze_answer(
            _answer_job("answers/team/ans.wav", uploaded_ref.checksum, "01JANS_REMOTE_OK")
        )
    assert res_dl.answer_id == "01JANS_REMOTE_OK"
    assert len(spy_speech.calls) == 2

    tampered_audio = tmp_path / "tampered.wav"
    tampered_audio.write_bytes(b"RIFF tampered answer audio")
    await storage.upload_file("answers/team/ans.wav", tampered_audio)

    with (
        patch("app.stages.answers.normalize_media") as mock_norm,
        pytest.raises(ChecksumMismatchError),
    ):
        await pipeline.analyze_answer(
            _answer_job("answers/team/ans.wav", uploaded_ref.checksum, "01JANS_REMOTE_BAD")
        )
    mock_norm.assert_not_called()
    assert len(spy_speech.calls) == 2


@pytest.mark.asyncio
async def test_report_loader_and_generate_report_verify_checksums_before_parsing(
    tmp_path: Path,
) -> None:
    """Checksummed report inputs fail before JSON parsing or LLM calls on mismatch/replacement."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    analysis_data = build_synthetic_analysis_data("01JSESSREPORT0000000000001")
    qa_data = build_synthetic_qa_data("01JSESSREPORT0000000000001")

    analysis_ref = await storage.upload_json(
        "ai/session/01JSESSREPORT0000000000001/analysis.json",
        analysis_data,
        artifact_id="01JANALYSISREF000000000001",
    )
    qa_ref = await storage.upload_json(
        "ai/session/01JSESSREPORT0000000000001/qa.json",
        qa_data,
        artifact_id="01JQAREF000000000000000001",
    )

    # 1. Matching checksums succeed
    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=storage)
    assert bundle.session_id == "01JSESSREPORT0000000000001"

    # 2. Corrupt non-JSON bytes with mismatched checksum must fail BEFORE json.loads (no fallback)
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_bytes(b"{not valid json at all")
    await storage.upload_file("ai/session/01JSESSREPORT0000000000001/analysis.json", corrupt_file)

    spy_judge = SpyJudgeProvider()
    pipeline = FakePipeline(judge_provider=spy_judge, object_storage=storage)
    report_job = GenerateReportPayload(
        report_id="01JREPORT00000000000000001",
        analysis_artifact=analysis_ref,
        qa_artifact=qa_ref,
    )

    with pytest.raises(ChecksumMismatchError):
        await pipeline.generate_report(report_job, practice_session_id="01JSESSREPORT0000000000001")

    assert spy_judge.report_calls == 0

    # 3. Restore analysis.json, but overwrite qa.json behind existing key with different valid JSON
    await storage.upload_json(
        "ai/session/01JSESSREPORT0000000000001/analysis.json",
        analysis_data,
        artifact_id="01JANALYSISREF000000000001",
    )
    modified_qa = dict(qa_data)
    modified_qa["tampered"] = True
    await storage.upload_json(
        "ai/session/01JSESSREPORT0000000000001/qa.json",
        modified_qa,
        artifact_id="01JQAREF000000000000000001",
    )

    with pytest.raises(ChecksumMismatchError):
        await pipeline.generate_report(report_job, practice_session_id="01JSESSREPORT0000000000001")

    assert spy_judge.report_calls == 0


@pytest.mark.asyncio
async def test_worker_returns_safe_non_retryable_integrity_failure_without_leaks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Checksum mismatch returns non-retryable FAILED update with no content or object key."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    secret_content = b"SECRET_PROPRIETARY_PITCH_AUDIO_BYTES_DO_NOT_LEAK"
    secret_key = "uploads/secret-team/private-folder/confidential-pitch.mp4"

    src = tmp_path / "secret.mp4"
    src.write_bytes(secret_content)
    await storage.upload_file(secret_key, src)

    pipeline = FakePipeline(object_storage=storage)
    backend = FakeBackendClient()
    envelope = {
        "schema_version": 1,
        "job_id": "01JJOBINTEGRITY00000000001",
        "job_type": JobType.ANALYZE_SESSION.value,
        "practice_session_id": "01JSESSINTEGRITY0000000001",
        "analysis_attempt": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "trace_id": "01JTRACEINTEGRITY000000001",
        "payload": {
            "presentation": {
                "artifact_id": "01JARTINTEGRITY00000000001",
                "object_key": secret_key,
                "checksum": WRONG_SHA256,
                "media_type": "video/mp4",
            },
            "supporting_documents": [],
            "rubric": {"rubric_id": "startup_pitch", "version": 1},
            "requested_capabilities": ["speech", "questions"],
        },
    }

    caplog.set_level(logging.DEBUG)
    mock_task = MagicMock()
    mock_task.request.retries = 0
    mock_task.max_retries = 3

    update = await _async_process_job_task(
        mock_task,
        envelope,
        pipeline=pipeline,
        backend_client=backend,
    )

    mock_task.retry.assert_not_called()
    assert update is not None
    assert update.status == UpdateStatus.FAILED
    assert isinstance(update.payload, FailedPayload)
    assert update.payload.code == ErrorCode.CHECKSUM_MISMATCH
    assert update.payload.retryable is False
    assert update.payload.stage == "ingestion"

    serialized = update.model_dump_json()
    assert secret_key not in serialized
    assert "confidential-pitch.mp4" not in serialized
    assert "SECRET_PROPRIETARY" not in serialized
    assert secret_key not in caplog.text
    assert "SECRET_PROPRIETARY" not in caplog.text

    # Also verify direct process_job handling for report jobs
    report_msg = QueueMessage(
        schema_version=1,
        job_id="01JJOBREPORTINT00000000001",
        job_type=JobType.GENERATE_REPORT,
        practice_session_id="01JSESSREPORTINT0000000001",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTRACEREPORTINT000000001",
        payload={
            "report_id": "01JREPORTINT00000000000001",
            "analysis_artifact": {
                "artifact_id": "01JARTSECRETANALYSIS000001",
                "object_key": "ai/session/secret-path/analysis.json",
                "checksum": WRONG_SHA256,
                "schema_version": 1,
            },
            "qa_artifact": {
                "artifact_id": "01JARTSECRETQA000000000001",
                "object_key": "ai/session/secret-path/qa.json",
                "checksum": WRONG_SHA256,
                "schema_version": 1,
            },
        },
    )
    await storage.upload_json(
        "ai/session/secret-path/analysis.json",
        {"secret_transcript": "DO_NOT_LEAK_THIS_JSON"},
    )
    report_update = await process_job(report_msg, pipeline)
    assert report_update.status == UpdateStatus.FAILED
    assert isinstance(report_update.payload, FailedPayload)
    assert report_update.payload.code == ErrorCode.CHECKSUM_MISMATCH
    assert report_update.payload.retryable is False
    assert report_update.payload.stage == "ingestion"
    assert "secret-path" not in report_update.payload.message
    assert "DO_NOT_LEAK" not in report_update.payload.message
