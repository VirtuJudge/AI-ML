"""Unit tests for FakePipeline implementation."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    EraseAIDataPayload,
    GenerateReportPayload,
    RubricRef,
    SpeakerMapping,
)
from app.pipeline import FakePipeline, combine_timed_evidence
from app.providers.fake_speech import FakeSpeechProvider
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    SpeakerSegment,
    TranscriptionResult,
    VisualObservation,
)
from app.stages.media import MediaSplitResult
from app.storage.base import ObjectNotFoundError
from app.storage.local import LocalDiskObjectStorage


@pytest.mark.asyncio
async def test_analyze_session_completes(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_session returns 3 primary questions and validates properly."""
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
    )

    result = await fake_pipeline.analyze_session(payload)
    assert len(result.primary_questions) == 3
    assert result.analysis_artifact.artifact_id is not None
    assert result.analysis_artifact.checksum.startswith("sha256:")
    assert len(result.speaker_labels) == 2


@pytest.mark.asyncio
async def test_analyze_session_with_supporting_documents(
    fake_pipeline: FakePipeline, tmp_path: Path
) -> None:
    """Test that analyze_session processes supporting documents and generates questions."""
    doc_path = tmp_path / "pitch_deck.pdf"
    doc_path.write_bytes(b"%PDF dummy")

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01JTESTDOC0000000000000001",
                object_key=str(doc_path),
                checksum="sha256:" + "b" * 64,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "documents", "questions"],
    )

    result = await fake_pipeline.analyze_session(payload)
    assert len(result.primary_questions) == 3
    assert result.analysis_artifact.artifact_id is not None
    assert len(result.limitations) == 0


@pytest.mark.asyncio
async def test_analyze_session_downloads_supporting_documents_from_object_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote Supporting Documents are downloaded before document extraction."""
    monkeypatch.chdir(tmp_path)
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "object-storage")
    source = tmp_path / "source-deck.pdf"
    source.write_bytes(b"%PDF synthetic supporting document")
    await storage.upload_file("uploads/team/project/deck.pdf", source)

    pipeline = FakePipeline(object_storage=storage)
    document_id = "01JTESTDOCREMOTE00000000001"
    session_id = "01JTESTSESSIONREMOTE0000001"
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTESTPRESENTATIONREMOTE01",
            object_key="uploads/team/project/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id=document_id,
                object_key="uploads/team/project/deck.pdf",
                checksum="sha256:" + "b" * 64,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "documents", "questions"],
        practice_session_id=session_id,
    )

    result = await pipeline.analyze_session(payload)
    stored_chunks = await pipeline.document_store.retrieve_top_k(
        session_id,
        [],
        asset_version_ids=[document_id],
    )

    assert len(stored_chunks) == 2
    assert not any(item.code == "document_file_missing" for item in result.limitations)


@pytest.mark.asyncio
async def test_analyze_session_result_is_stable(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_session returns stable, identical results for repeated calls."""
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )

    result1 = await fake_pipeline.analyze_session(payload)
    result2 = await fake_pipeline.analyze_session(payload)
    assert result1 == result2


@pytest.mark.asyncio
async def test_analyze_answer_completes(fake_pipeline: FakePipeline) -> None:
    """Test that analyze_answer returns real artifact IDs and follow-up when remaining > 0."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        question_text="What supports the acquisition-cost claim?",
        rubric_dimension="market_and_business_model",
        question_evidence_ids=["ev_speech_001"],
        remaining_follow_ups=1,
    )

    result = await fake_pipeline.analyze_answer(payload)
    assert result.answer_id == payload.answer_id
    assert result.transcript_artifact_id != "01JEXAMPLE000000000000002A"
    assert len(result.transcript_artifact_id) > 0
    assert result.assessment_artifact_id != "01JEXAMPLE000000000000002B"
    assert len(result.assessment_artifact_id) > 0
    assert result.follow_up is not None
    assert isinstance(result.follow_up.rubric_dimension, str)
    assert len(result.follow_up.rubric_dimension) > 0
    assert isinstance(result.follow_up.evidence_ids, list)
    assert result.follow_up.evidence_ids == ["ev_speech_001"]


@pytest.mark.asyncio
async def test_analyze_answer_no_followup_when_zero_remaining(
    fake_pipeline: FakePipeline,
) -> None:
    """Test that analyze_answer returns follow_up=None when remaining_follow_ups is 0."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=0,
    )

    result = await fake_pipeline.analyze_answer(payload)
    assert result.transcript_artifact_id != "01JEXAMPLE000000000000002A"
    assert len(result.transcript_artifact_id) > 0
    assert result.assessment_artifact_id != "01JEXAMPLE000000000000002B"
    assert len(result.assessment_artifact_id) > 0
    assert result.follow_up is None


@pytest.mark.asyncio
async def test_analyze_answer_skipped_produces_no_invented_evidence() -> None:
    """Verify empty transcript produces valid artifact IDs and no invented evidence."""

    class EmptySpeechProvider(FakeSpeechProvider):
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            return TranscriptionResult(full_text="", segments=[])

    pipeline = FakePipeline(speech_provider=EmptySpeechProvider())
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000099",
        answered_by="01JTEST0000000000000000013",
        practice_session_id="01JTEST0000000000000000009",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        question_text="What supports the acquisition-cost claim?",
        rubric_dimension="market_and_business_model",
        question_evidence_ids=["ev_speech_001"],
        remaining_follow_ups=2,
    )

    result = await pipeline.analyze_answer(payload)
    assert result.transcript_artifact_id != "01JEXAMPLE000000000000002A"
    assert len(result.transcript_artifact_id) > 0
    assert result.assessment_artifact_id != "01JEXAMPLE000000000000002B"
    assert len(result.assessment_artifact_id) > 0
    assert result.follow_up is None

    # Verify assessment artifact uploaded to object storage has no invented evidence
    assessment_data = await pipeline.object_storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/assessment.json"
    )
    assert assessment_data["assessment"]["evidence_ids"] == []
    assert assessment_data["assessment"]["text"] == ""
    assert assessment_data["follow_up"] is None


@pytest.mark.asyncio
async def test_analyze_answer_idempotent(fake_pipeline: FakePipeline) -> None:
    """Call analyze_answer twice with the same payload and verify results are identical."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=1,
    )

    result1 = await fake_pipeline.analyze_answer(payload)
    result2 = await fake_pipeline.analyze_answer(payload)

    assert result1.answer_id == result2.answer_id
    assert len(result1.transcript_artifact_id) == 26
    assert len(result2.transcript_artifact_id) == 26
    assert len(result1.assessment_artifact_id) == 26
    assert len(result2.assessment_artifact_id) == 26
    assert result1.follow_up == result2.follow_up


@pytest.mark.asyncio
async def test_analyze_answer_follow_up_has_required_fields(
    fake_pipeline: FakePipeline,
) -> None:
    """Verify follow_up has text, reason, dimension, and evidence_ids with quota > 0."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        question_text="What supports the acquisition-cost claim?",
        rubric_dimension="market_and_business_model",
        question_evidence_ids=["ev_speech_001"],
        remaining_follow_ups=2,
    )

    result = await fake_pipeline.analyze_answer(payload)
    assert result.follow_up is not None
    assert isinstance(result.follow_up.text, str) and len(result.follow_up.text.strip()) > 0
    assert isinstance(result.follow_up.reason, str) and len(result.follow_up.reason.strip()) > 0
    assert (
        isinstance(result.follow_up.rubric_dimension, str)
        and len(result.follow_up.rubric_dimension.strip()) > 0
    )
    assert isinstance(result.follow_up.evidence_ids, list)
    assert result.follow_up.evidence_ids == ["ev_speech_001"]


@pytest.mark.asyncio
async def test_analyze_answer_privacy_no_transcript_or_audio_in_logs(
    fake_pipeline: FakePipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify private answer audio and transcript text do not appear in normal logs."""
    sensitive_transcript = "CONFIDENTIAL_PATENT_SECRET_REVENUE_METRIC_99"

    class SensitiveSpeechProvider(FakeSpeechProvider):
        async def transcribe(self, audio_path: Path, **kwargs: Any) -> TranscriptionResult:
            return TranscriptionResult(
                segments=[],
                full_text=sensitive_transcript,
            )

    pipeline = FakePipeline(speech_provider=SensitiveSpeechProvider())
    caplog.set_level(logging.INFO)

    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTEST0000000000000000014",
            object_key="audio/answer_private.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=1,
    )

    result = await pipeline.analyze_answer(payload)
    assert result.answer_id == payload.answer_id
    assert sensitive_transcript not in caplog.text


@pytest.mark.asyncio
async def test_generate_report_completes(fake_pipeline: FakePipeline) -> None:
    """Test report generation returns mapped user IDs for feedback."""
    speaker_mappings = [
        SpeakerMapping(
            speaker_label="SPEAKER_00",
            user_id="01JTEST0000000000000000031",
            display_name="Founder Alice",
        ),
        SpeakerMapping(
            speaker_label="SPEAKER_01",
            user_id="01JTEST0000000000000000032",
            display_name="CTO Bob",
        ),
    ]
    payload = GenerateReportPayload(
        report_id="01JTEST0000000000000000030",
        analysis_artifact=ArtifactRef(
            artifact_id="01JTEST0000000000000000033",
            object_key="artifacts/analysis.json",
            checksum="sha256:" + "c" * 64,
            schema_version=1,
        ),
        qa_artifact=ArtifactRef(
            artifact_id="01JTEST0000000000000000034",
            object_key="artifacts/qa.json",
            checksum="sha256:" + "d" * 64,
            schema_version=1,
        ),
        speaker_mappings=speaker_mappings,
    )

    result = await fake_pipeline.generate_report(payload)
    assert len(result.member_feedback_user_ids) == 2
    assert "01JTEST0000000000000000031" in result.member_feedback_user_ids
    assert "01JTEST0000000000000000032" in result.member_feedback_user_ids

    # Verify evaluation artifact uploaded to storage
    assert result.evaluation_artifact.schema_version == 1
    assert result.evaluation_artifact.object_key.endswith("evaluation.json")
    eval_data = await fake_pipeline.object_storage.read_json(result.evaluation_artifact.object_key)
    assert "overall_score" in eval_data
    assert len(eval_data["components"]) == 6
    assert len(eval_data["member_feedback"]) == 2

    # Verify report markdown artifact uploaded to storage
    assert result.report_artifact.schema_version == 1
    assert result.report_artifact.object_key.endswith("report.md")


@pytest.mark.asyncio
async def test_erase_data_completes(fake_pipeline: FakePipeline) -> None:
    """Test data erasure returns expected record and object counts."""
    payload = EraseAIDataPayload(
        erasure_request_id="01JTEST0000000000000000040",
        scope="practice_session",
        scope_id="01JTEST0000000000000000041",
    )

    result = await fake_pipeline.erase_data(payload)
    assert result.erasure_request_id == "01JTEST0000000000000000040"
    assert result.deleted_records == 14
    assert result.deleted_objects == 6


def test_combine_timed_evidence_audio_attribution() -> None:
    """Verify audio observations without labels are attributed to active speaker turn."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    audio_obs = [
        AudioObservation(
            start_ms=0, end_ms=4000, metric="speaking_rate_wpm", value=140.0, unit="wpm"
        ),
        AudioObservation(
            start_ms=6000, end_ms=9000, metric="speaking_rate_wpm", value=130.0, unit="wpm"
        ),
    ]
    _, updated_audio, _, _ = combine_timed_evidence(diarization, [], audio_obs)
    assert len(updated_audio) == 2
    assert updated_audio[0].speaker_label == "SPEAKER_00"
    assert updated_audio[1].speaker_label == "SPEAKER_01"


def test_combine_timed_evidence_deduplicates_fragmented_tracks() -> None:
    """Verify visual tracks across edit cuts (PERSON_00 and PERSON_02) both map to SPEAKER_00."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=10000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=10000, end_ms=20000, speaker_label="SPEAKER_01"),
            SpeakerSegment(start_ms=20000, end_ms=30000, speaker_label="SPEAKER_00"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    visual_obs = [
        # PERSON_00 speaks during 0-10s
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="mouth_aspect_ratio",
            value=0.25,
            unit="ratio",
            speaker_label="PERSON_00",
        ),
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="gaze_direction",
            value=1.0,
            unit="index",
            speaker_label="PERSON_00",
        ),
        # PERSON_01 speaks during 10-20s
        VisualObservation(
            start_ms=11000,
            end_ms=12000,
            metric="mouth_aspect_ratio",
            value=0.30,
            unit="ratio",
            speaker_label="PERSON_01",
        ),
        # Presenter A returns in new track PERSON_02 during SPEAKER_00's turn (20-30s)
        VisualObservation(
            start_ms=21000,
            end_ms=22000,
            metric="mouth_aspect_ratio",
            value=0.28,
            unit="ratio",
            speaker_label="PERSON_02",
        ),
        VisualObservation(
            start_ms=21000,
            end_ms=22000,
            metric="gaze_direction",
            value=1.0,
            unit="index",
            speaker_label="PERSON_02",
        ),
    ]

    v_res, _, mapping, lims = combine_timed_evidence(diarization, visual_obs, [])

    assert mapping["PERSON_00"] == "SPEAKER_00"
    assert mapping["PERSON_01"] == "SPEAKER_01"
    assert mapping["PERSON_02"] == "SPEAKER_00"
    assert len(lims) == 0

    # All PERSON_00 and PERSON_02 observations must be relabeled as SPEAKER_00
    for obs in v_res:
        if obs.start_ms < 10000 or obs.start_ms >= 20000:
            assert obs.speaker_label == "SPEAKER_00"
        else:
            assert obs.speaker_label == "SPEAKER_01"


def test_combine_timed_evidence_co_presenters_mar_disambiguation() -> None:
    """Verify co-presenters on stage simultaneously are differentiated by mouth_aspect_ratio."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    visual_obs = [
        # During 0-5s: PERSON_00 has high MAR (speaking), PERSON_01 has low MAR (listening)
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="mouth_aspect_ratio",
            value=0.26,
            unit="ratio",
            speaker_label="PERSON_00",
        ),
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="mouth_aspect_ratio",
            value=0.04,
            unit="ratio",
            speaker_label="PERSON_01",
        ),
        # During 5-10s: PERSON_01 has high MAR (speaking), PERSON_00 has low MAR (listening)
        VisualObservation(
            start_ms=6000,
            end_ms=7000,
            metric="mouth_aspect_ratio",
            value=0.05,
            unit="ratio",
            speaker_label="PERSON_00",
        ),
        VisualObservation(
            start_ms=6000,
            end_ms=7000,
            metric="mouth_aspect_ratio",
            value=0.29,
            unit="ratio",
            speaker_label="PERSON_01",
        ),
    ]

    _, _, mapping, _ = combine_timed_evidence(diarization, visual_obs, [])
    assert mapping["PERSON_00"] == "SPEAKER_00"
    assert mapping["PERSON_01"] == "SPEAKER_01"


def test_combine_timed_evidence_ambiguous_track_limitation() -> None:
    """Verify an ambiguous visual track that never spoke produces a limitation."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=5000, end_ms=10000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )
    # PERSON_SILENT appears during 15-20s when no speech occurs
    visual_obs = [
        VisualObservation(
            start_ms=15000,
            end_ms=16000,
            metric="gaze_direction",
            value=1.0,
            unit="index",
            speaker_label="PERSON_SILENT",
        ),
    ]
    _, _, mapping, lims = combine_timed_evidence(diarization, visual_obs, [])
    assert "PERSON_SILENT" not in mapping
    assert any(lim.code == "ambiguous_visual_speaker_mapping" for lim in lims)


@pytest.mark.asyncio
async def test_analyze_session_aggregates_limitations() -> None:
    """Verify limitations from all stages are preserved and aggregated."""

    class VisionWithLimitationProvider:
        async def analyze_video(self, video_path: Path, **kwargs: Any) -> list[VisualObservation]:
            return []

    pipeline = FakePipeline(vision_provider=VisionWithLimitationProvider())
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )
    result = await pipeline.analyze_session(payload)
    limitation_codes = [lim.code for lim in result.limitations]
    assert "no_visual_observations" in limitation_codes


@pytest.mark.asyncio
async def test_analyze_session_raises_on_download_error_with_real_provider() -> None:
    """Verify download failure is not swallowed when running with a non-fake provider."""
    mock_storage = AsyncMock()
    mock_storage.download_file.side_effect = ObjectNotFoundError(
        "Object 'uploads/pres.mp4' not found"
    )

    class RealDummySpeechProvider:
        async def transcribe(self, audio_path: Path, **kwargs: Any) -> Any:
            raise NotImplementedError

    pipeline = FakePipeline(
        speech_provider=RealDummySpeechProvider(),
        object_storage=mock_storage,
    )
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000001",
            object_key="uploads/pres.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )

    with pytest.raises(ObjectNotFoundError, match=r"Object 'uploads/pres\.mp4' not found"):
        await pipeline.analyze_session(payload)


@pytest.mark.asyncio
async def test_analyze_session_downloads_media_when_available_in_storage(tmp_path: Path) -> None:
    """Verify presentation media is downloaded to temp directory and used by pipeline."""
    import wave

    storage_dir = tmp_path / "storage"
    storage = LocalDiskObjectStorage(base_dir=storage_dir)

    # Upload test presentation WAV to storage
    pres_file = tmp_path / "presentation.wav"
    with wave.open(str(pres_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * (16000 * 15))

    await storage.upload_file("uploads/presentation.wav", pres_file)

    pipeline = FakePipeline(object_storage=storage)
    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTEST0000000000000000099",
            object_key="uploads/presentation.wav",
            checksum="sha256:" + "a" * 64,
            media_type="audio/wav",
            duration_ms=15000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "vision", "audio", "questions"],
    )

    with patch(
        "app.pipeline.split_media",
        return_value=MediaSplitResult(
            audio_path=pres_file,
            video_path=None,
            duration_ms=15000,
            sample_rate=16000,
            channels=1,
        ),
    ):
        res = await pipeline.analyze_session(payload)
    assert len(res.primary_questions) == 3

    # Temp media should have been downloaded
    downloaded = Path(".storage/temp_media/01JTEST0000000000000000099/presentation.wav")
    assert downloaded.is_file()
    assert downloaded.stat().st_size == pres_file.stat().st_size


@pytest.mark.asyncio
async def test_analyze_answer_skipped_with_none_audio(fake_pipeline: FakePipeline) -> None:
    """Verify analyze_answer handles skipped answer with audio=None without error."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000098",
        answered_by="01JTEST0000000000000000013",
        practice_session_id="01JTEST0000000000000000009",
        audio=None,
        remaining_follow_ups=1,
    )
    result = await fake_pipeline.analyze_answer(payload)
    assert result.answer_id == payload.answer_id
    assert result.transcript_artifact_id
    assert result.assessment_artifact_id
    assert result.follow_up is None

    # Verify DerivedArtifact fields in stored artifacts
    transcript_data = await fake_pipeline.object_storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/transcript.json"
    )
    assert transcript_data["kind"] == "transcript"
    assert transcript_data["producer_version"] == "ai-ml/0.1.0"
    assert transcript_data["source_artifact_ids"] == []
    assert transcript_data["created_at"] != "2026-09-02T12:00:00Z"

    assessment_data = await fake_pipeline.object_storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/assessment.json"
    )
    assert assessment_data["kind"] == "answer_assessment"
    assert assessment_data["producer_version"] == "ai-ml/0.1.0"
    assert assessment_data["source_artifact_ids"] == [result.transcript_artifact_id]
    assert assessment_data["created_at"] != "2026-09-02T12:00:00Z"


@pytest.mark.asyncio
async def test_analyze_answer_derived_artifact_fields(fake_pipeline: FakePipeline) -> None:
    """Verify stored transcript and assessment artifacts conform to DerivedArtifact contract."""
    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000077",
        answered_by="01JTEST0000000000000000013",
        practice_session_id="01JTEST0000000000000000009",
        audio=AudioAssetInput(
            artifact_id="01JTESTAUDIO00000000000001",
            object_key="audio/answer.wav",
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=1,
    )
    result = await fake_pipeline.analyze_answer(payload)

    transcript_data = await fake_pipeline.object_storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/transcript.json"
    )
    assert transcript_data["kind"] == "transcript"
    assert transcript_data["producer_version"] == "ai-ml/0.1.0"
    assert transcript_data["source_artifact_ids"] == ["01JTESTAUDIO00000000000001"]
    assert transcript_data["created_at"] != "2026-09-02T12:00:00Z"

    assessment_data = await fake_pipeline.object_storage.read_json(
        f"ai/session/{payload.practice_session_id}/answers/{payload.answer_id}/assessment.json"
    )
    assert assessment_data["kind"] == "answer_assessment"
    assert assessment_data["producer_version"] == "ai-ml/0.1.0"
    assert assessment_data["source_artifact_ids"] == [result.transcript_artifact_id]
    assert assessment_data["created_at"] != "2026-09-02T12:00:00Z"
    assert assessment_data["assessment"]["evidence_ids"] == []


@pytest.mark.asyncio
async def test_analyze_answer_download_error_logs_artifact_id_not_object_key(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Verify download error logs contain artifact_id instead of raw object_key."""
    secret_key = "secret/storage/path/private_key_do_not_log.wav"
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "empty_storage")
    pipeline = FakePipeline(object_storage=storage)

    payload = AnalyzeAnswerPayload(
        qa_round_id="01JTEST0000000000000000010",
        question_id="01JTEST0000000000000000011",
        answer_id="01JTEST0000000000000000012",
        answered_by="01JTEST0000000000000000013",
        audio=AudioAssetInput(
            artifact_id="01JTESTSAFEARTIFACT00000001",
            object_key=secret_key,
            checksum="sha256:" + "b" * 64,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=1,
    )

    caplog.set_level(logging.WARNING)
    await pipeline.analyze_answer(payload)

    assert secret_key not in caplog.text
    assert "01JTESTSAFEARTIFACT00000001" in caplog.text


@pytest.mark.asyncio
async def test_analyze_session_embeds_by_speaker_in_analysis_artifact(
    tmp_path: Path,
) -> None:
    """Verify analysis.json includes by_speaker with speaking intervals and windowed metrics."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTESTSESSIONSPK000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions", "vision", "audio"],
    )

    result = await pipeline.analyze_session(payload)
    stored = await storage.read_json(result.analysis_artifact.object_key)

    assert "by_speaker" in stored
    by_speaker = stored["by_speaker"]

    # Verify both fake speakers have isolated entries
    assert "SPEAKER_00" in by_speaker
    assert "SPEAKER_01" in by_speaker

    # Verify SPEAKER_00 profile
    spk0 = by_speaker["SPEAKER_00"]
    assert spk0["speaking_time_ms"] == 9200
    assert len(spk0["intervals"]) == 1
    assert spk0["intervals"][0]["formatted"] == "00:00 - 00:09"
    assert spk0["intervals"][0]["start_ms"] == 0
    assert spk0["intervals"][0]["end_ms"] == 9200

    # Verify SPEAKER_00 windows contain aggregated metrics
    assert len(spk0["windows"]) >= 1
    w0 = spk0["windows"][0]
    assert w0["start_ms"] == 0
    assert w0["end_ms"] == 10000
    assert "acoustic" in w0
    assert "speaking_rate_wpm" in w0["acoustic"]
    assert "pitch_mean_hz" in w0["acoustic"]

    # Verify SPEAKER_01 profile
    spk1 = by_speaker["SPEAKER_01"]
    assert spk1["speaking_time_ms"] == 4700
    assert len(spk1["intervals"]) == 1
    assert spk1["intervals"][0]["formatted"] == "00:09 - 00:14"
    assert spk1["intervals"][0]["start_ms"] == 9300
    assert spk1["intervals"][0]["end_ms"] == 14000
