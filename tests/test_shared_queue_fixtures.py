"""Tests validating AI-ML contract models against shared Backend golden fixtures.

Ensures cross-repository queue message compatibility between Backend producers and AI consumers.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    CancelledPayload,
    EraseAIDataPayload,
    ErasureCompleted,
    FailedPayload,
    GenerateReportPayload,
    JobType,
    PrimaryQuestion,
    ProgressPayload,
    QueueMessage,
    ReportCompleted,
    RubricRef,
    SessionAnalysisCompleted,
    StartedPayload,
    UpdateStatus,
    WorkerUpdate,
)

FIXTURES_DIR_BACKEND = Path(__file__).resolve().parent.parent.parent / "Backend" / "contracts" / "fixtures" / "ai"
FIXTURES_DIR_LOCAL = Path(__file__).resolve().parent / "fixtures" / "ai"


def _get_fixture_dir() -> Path:
    if FIXTURES_DIR_BACKEND.is_dir():
        return FIXTURES_DIR_BACKEND
    if FIXTURES_DIR_LOCAL.is_dir():
        return FIXTURES_DIR_LOCAL
    raise FileNotFoundError("Could not find fixtures in Backend or local directory")


def _load_fixture(filename: str) -> dict:
    fdir = _get_fixture_dir()
    path = fdir / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_analyze_session_valid_fixture_consumption():
    """Verify backend-generated analyze_session job matches QueueMessage and AnalyzeSessionPayload."""
    data = _load_fixture("job_analyze_session_valid.json")
    msg = QueueMessage.model_validate(data)

    assert msg.schema_version == 1
    assert msg.job_id == "01JEXAMPLE0000000000000001"
    assert msg.job_type == JobType.ANALYZE_SESSION
    assert msg.practice_session_id == "01JEXAMPLE0000000000000002"
    assert msg.analysis_attempt == 1
    assert isinstance(msg.created_at, datetime)
    assert msg.trace_id == "trc_test_001"

    payload = AnalyzeSessionPayload.model_validate(msg.payload)
    assert isinstance(payload.presentation, AssetInput)
    assert payload.presentation.artifact_id == "01JEXAMPLE0000000000000003"
    assert payload.presentation.object_key == "raw/presentation.mp4"
    assert payload.presentation.checksum.startswith("sha256:")
    assert payload.presentation.media_type == "video/mp4"
    assert payload.presentation.duration_ms == 180000

    assert len(payload.supporting_documents) == 1
    doc = payload.supporting_documents[0]
    assert doc.artifact_id == "01JEXAMPLE0000000000000004"
    assert doc.object_key == "raw/deck.pdf"
    assert doc.checksum.startswith("sha256:")
    assert doc.media_type == "application/pdf"

    assert isinstance(payload.rubric, RubricRef)
    assert payload.rubric.rubric_id == "rubric_seed_default"
    assert payload.rubric.version == 1
    assert payload.requested_capabilities == ["pitch_structure", "audio_analysis"]


def test_analyze_answer_valid_fixture_consumption():
    """Verify backend-generated analyze_answer job matches QueueMessage and AnalyzeAnswerPayload."""
    data = _load_fixture("job_analyze_answer_valid.json")
    msg = QueueMessage.model_validate(data)

    assert msg.job_type == JobType.ANALYZE_ANSWER
    payload = AnalyzeAnswerPayload.model_validate(msg.payload)
    assert payload.qa_round_id == "qa_round_001"
    assert payload.question_id == "q_001"
    assert payload.answer_id == "ans_001"
    assert payload.answered_by == "user_founder_01"
    assert isinstance(payload.audio, AudioAssetInput)
    assert payload.audio.duration_ms == 45000
    assert payload.remaining_follow_ups == 2


def test_generate_report_valid_fixture_consumption():
    """Verify backend-generated generate_report job matches QueueMessage and GenerateReportPayload."""
    data = _load_fixture("job_generate_report_valid.json")
    msg = QueueMessage.model_validate(data)

    assert msg.job_type == JobType.GENERATE_REPORT
    payload = GenerateReportPayload.model_validate(msg.payload)
    assert payload.report_id == "rep_001"
    assert payload.rubric.rubric_id == "startup_pitch"
    assert isinstance(payload.analysis_artifact, ArtifactRef)
    assert payload.analysis_artifact.schema_version == 1
    assert isinstance(payload.qa_artifact, ArtifactRef)
    assert payload.qa_artifact.schema_version == 1
    assert len(payload.speaker_mappings) == 1
    assert payload.speaker_mappings[0].speaker_label == "SPEAKER_00"
    assert payload.speaker_mappings[0].user_id == "user_founder_01"
    assert payload.speaker_mappings[0].display_name == "Jane Founder"


def test_erase_ai_data_valid_fixture_consumption():
    """Verify backend-generated erase_ai_data job matches QueueMessage and EraseAIDataPayload."""
    data = _load_fixture("job_erase_ai_data_valid.json")
    msg = QueueMessage.model_validate(data)

    assert msg.job_type == JobType.ERASE_AI_DATA
    payload = EraseAIDataPayload.model_validate(msg.payload)
    assert payload.erasure_request_id == "era_001"
    assert payload.scope == "practice_session"
    assert payload.scope_id == "01JEXAMPLE0000000000000002"


def test_invalid_job_fixtures():
    """Verify negative job fixtures fail validation cleanly."""
    # missing rubric
    data_missing = _load_fixture("job_missing_required.json")
    msg = QueueMessage.model_validate(data_missing)
    with pytest.raises(ValidationError):
        AnalyzeSessionPayload.model_validate(msg.payload)

    # unknown job type
    data_unknown = _load_fixture("job_unknown_enum.json")
    with pytest.raises(ValidationError):
        QueueMessage.model_validate(data_unknown)

    # invalid audio missing duration_ms
    data_audio = _load_fixture("job_audio_missing_duration.json")
    msg_audio = QueueMessage.model_validate(data_audio)
    with pytest.raises(ValidationError):
        AnalyzeAnswerPayload.model_validate(msg_audio.payload)

    # invalid checksum
    data_checksum = _load_fixture("job_invalid_checksum.json")
    msg_chk = QueueMessage.model_validate(data_checksum)
    with pytest.raises(ValidationError):
        AnalyzeSessionPayload.model_validate(msg_chk.payload)


def test_worker_update_valid_fixtures():
    """Verify worker update valid fixtures deserialize and match models."""
    # started
    data_started = _load_fixture("worker_update_started_valid.json")
    up_started = WorkerUpdate.model_validate(data_started)
    assert up_started.status == UpdateStatus.STARTED
    started_pl = StartedPayload.model_validate(up_started.payload)
    assert started_pl.pipeline_version == "1.0.0"

    # progress
    data_progress = _load_fixture("worker_update_progress_valid.json")
    up_progress = WorkerUpdate.model_validate(data_progress)
    assert up_progress.status == UpdateStatus.PROGRESS
    progress_pl = ProgressPayload.model_validate(up_progress.payload)
    assert 0.0 <= progress_pl.progress <= 1.0

    # completed analyze_session
    data_comp_session = _load_fixture("worker_update_completed_analyze_session.json")
    up_session = WorkerUpdate.model_validate(data_comp_session)
    assert up_session.status == UpdateStatus.COMPLETED
    session_pl = SessionAnalysisCompleted.model_validate(up_session.payload)
    assert len(session_pl.primary_questions) == 3
    assert all(len(q.evidence_ids) > 0 for q in session_pl.primary_questions)

    # completed analyze_answer
    data_comp_ans = _load_fixture("worker_update_completed_analyze_answer.json")
    up_ans = WorkerUpdate.model_validate(data_comp_ans)
    assert up_ans.status == UpdateStatus.COMPLETED
    ans_pl = AnswerAnalysisCompleted.model_validate(up_ans.payload)
    assert ans_pl.answer_id == "ans_001"

    # completed generate_report
    data_comp_rep = _load_fixture("worker_update_completed_generate_report.json")
    up_rep = WorkerUpdate.model_validate(data_comp_rep)
    assert up_rep.status == UpdateStatus.COMPLETED
    rep_pl = ReportCompleted.model_validate(up_rep.payload)
    assert rep_pl.evaluation_artifact.schema_version == 1

    # completed erase_ai_data
    data_comp_erase = _load_fixture("worker_update_completed_erase_ai_data.json")
    up_erase = WorkerUpdate.model_validate(data_comp_erase)
    assert up_erase.status == UpdateStatus.COMPLETED
    erase_pl = ErasureCompleted.model_validate(up_erase.payload)
    assert erase_pl.deleted_records >= 0

    # cancelled
    data_cancelled = _load_fixture("worker_update_cancelled_valid.json")
    up_can = WorkerUpdate.model_validate(data_cancelled)
    assert up_can.status == UpdateStatus.CANCELLED
    CancelledPayload.model_validate(up_can.payload)

    # failed
    data_failed = _load_fixture("worker_update_failed_valid.json")
    up_fail = WorkerUpdate.model_validate(data_failed)
    assert up_fail.status == UpdateStatus.FAILED
    FailedPayload.model_validate(up_fail.payload)


def test_worker_update_negative_fixtures():
    """Verify negative worker update fixtures fail validation."""
    # missing required
    data_missing = _load_fixture("worker_update_missing_required.json")
    up_missing = WorkerUpdate.model_validate(data_missing)
    with pytest.raises(ValidationError):
        StartedPayload.model_validate(up_missing.payload)

    # unknown status enum
    data_unknown = _load_fixture("worker_update_unknown_enum.json")
    with pytest.raises(ValidationError):
        WorkerUpdate.model_validate(data_unknown)

    # progress out of range
    data_prog_oor = _load_fixture("worker_update_progress_out_of_range.json")
    up_prog = WorkerUpdate.model_validate(data_prog_oor)
    with pytest.raises(ValidationError):
        ProgressPayload.model_validate(up_prog.payload)

    # empty evidence in primary questions
    data_empty_ev = _load_fixture("worker_update_empty_evidence.json")
    up_empty_ev = WorkerUpdate.model_validate(data_empty_ev)
    with pytest.raises(ValidationError):
        SessionAnalysisCompleted.model_validate(up_empty_ev.payload)

    # primary question count under (fewer than 3)
    data_under = _load_fixture("worker_update_question_count_under.json")
    up_under = WorkerUpdate.model_validate(data_under)
    with pytest.raises(ValidationError):
        SessionAnalysisCompleted.model_validate(up_under.payload)

    # primary question count over (more than 3)
    data_over = _load_fixture("worker_update_question_count_over.json")
    up_over = WorkerUpdate.model_validate(data_over)
    with pytest.raises(ValidationError):
        SessionAnalysisCompleted.model_validate(up_over.payload)

    # negative deleted records
    data_neg = _load_fixture("worker_update_negative_deleted_records.json")
    up_neg = WorkerUpdate.model_validate(data_neg)
    with pytest.raises(ValidationError):
        ErasureCompleted.model_validate(up_neg.payload)
