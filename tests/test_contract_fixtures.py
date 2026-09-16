"""Contract fixture tests for VirtuJudge AI-ML."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    EraseAIDataPayload,
    ErasureCompleted,
    FailedPayload,
    GenerateReportPayload,
    ProgressPayload,
    QueueMessage,
    ReportCompleted,
    SessionAnalysisCompleted,
    StartedPayload,
)
from tests.conftest import load_json_fixture_text


class TestQueueMessageFixtures:
    """Validate queue message JSON fixtures deserialization."""

    def test_analyze_session_fixture(self) -> None:
        raw = load_json_fixture_text("queue_messages/analyze_session.json")
        msg = QueueMessage.model_validate_json(raw)
        assert msg.job_type == "analyze_session"
        payload = AnalyzeSessionPayload.model_validate(msg.payload)
        assert payload.presentation.media_type == "video/mp4"
        assert payload.presentation.checksum.startswith("sha256:")
        assert payload.rubric.rubric_id == "startup_pitch"

    def test_analyze_answer_fixture(self) -> None:
        raw = load_json_fixture_text("queue_messages/analyze_answer.json")
        msg = QueueMessage.model_validate_json(raw)
        assert msg.job_type == "analyze_answer"
        payload = AnalyzeAnswerPayload.model_validate(msg.payload)
        assert payload.audio.duration_ms == 84000
        assert payload.remaining_follow_ups == 2

    def test_generate_report_fixture(self) -> None:
        raw = load_json_fixture_text("queue_messages/generate_report.json")
        msg = QueueMessage.model_validate_json(raw)
        assert msg.job_type == "generate_report"
        payload = GenerateReportPayload.model_validate(msg.payload)
        assert payload.rubric.rubric_id == "startup_pitch"
        assert len(payload.speaker_mappings) == 1
        assert payload.speaker_mappings[0].speaker_label == "SPEAKER_00"

    def test_erase_ai_data_fixture(self) -> None:
        raw = load_json_fixture_text("queue_messages/erase_ai_data.json")
        msg = QueueMessage.model_validate_json(raw)
        assert msg.job_type == "erase_ai_data"
        payload = EraseAIDataPayload.model_validate(msg.payload)
        assert payload.scope == "practice_session"


class TestCompletedPayloadFixtures:
    """Validate completed payload JSON fixtures deserialization."""

    def test_analyze_session_completed(self) -> None:
        raw = load_json_fixture_text("completed_payloads/analyze_session_completed.json")
        completed = SessionAnalysisCompleted.model_validate_json(raw)
        assert len(completed.primary_questions) == 3
        assert len(completed.speaker_labels) == 2
        assert completed.analysis_artifact.schema_version == 1

    def test_analyze_answer_completed_with_followup(self) -> None:
        raw = load_json_fixture_text("completed_payloads/analyze_answer_completed.json")
        completed = AnswerAnalysisCompleted.model_validate_json(raw)
        assert completed.follow_up is not None
        assert completed.follow_up.rubric_dimension == "business_reasoning"

    def test_analyze_answer_completed_no_followup(self) -> None:
        raw = load_json_fixture_text("completed_payloads/analyze_answer_completed_no_followup.json")
        completed = AnswerAnalysisCompleted.model_validate_json(raw)
        assert completed.follow_up is None

    def test_generate_report_completed(self) -> None:
        raw = load_json_fixture_text("completed_payloads/generate_report_completed.json")
        completed = ReportCompleted.model_validate_json(raw)
        assert completed.evaluation_artifact.schema_version == 1
        assert completed.report_artifact.schema_version == 1
        assert len(completed.member_feedback_user_ids) == 1

    def test_erase_ai_data_completed(self) -> None:
        raw = load_json_fixture_text("completed_payloads/erase_ai_data_completed.json")
        completed = ErasureCompleted.model_validate_json(raw)
        assert completed.deleted_records == 14
        assert completed.deleted_objects == 6


class TestUpdatePayloadFixtures:
    """Validate update payload JSON fixtures deserialization."""

    def test_started_fixture(self) -> None:
        raw = load_json_fixture_text("update_payloads/started.json")
        payload = StartedPayload.model_validate_json(raw)
        assert payload.pipeline_version == "0.1.0"

    def test_progress_fixture(self) -> None:
        raw = load_json_fixture_text("update_payloads/progress.json")
        payload = ProgressPayload.model_validate_json(raw)
        assert payload.stage == "speech"
        assert payload.progress == 0.45

    def test_failed_fixture(self) -> None:
        raw = load_json_fixture_text("update_payloads/failed.json")
        payload = FailedPayload.model_validate_json(raw)
        assert payload.stage == "speech"
        assert payload.code == "provider_timeout"
        assert payload.retryable is True


class TestInvalidFixtures:
    """Validate that invalid fixtures fail with expected ValidationError."""

    def test_bad_checksum_rejected(self) -> None:
        raw = load_json_fixture_text("invalid/bad_checksum.json")
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeSessionPayload.model_validate_json(raw)
        assert "checksum" in str(exc_info.value)

    def test_missing_required_field_rejected(self) -> None:
        raw = load_json_fixture_text("invalid/missing_required_field.json")
        with pytest.raises(ValidationError) as exc_info:
            QueueMessage.model_validate_json(raw)
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "job_id" in missing_fields
        assert "trace_id" in missing_fields

    def test_unknown_job_type_rejected(self) -> None:
        raw = load_json_fixture_text("invalid/unknown_job_type.json")
        with pytest.raises(ValidationError) as exc_info:
            QueueMessage.model_validate_json(raw)
        assert "job_type" in str(exc_info.value)

    def test_wrong_question_count_rejected(self) -> None:
        raw = load_json_fixture_text("invalid/wrong_question_count.json")
        with pytest.raises(ValidationError) as exc_info:
            SessionAnalysisCompleted.model_validate_json(raw)
        assert "primary_questions must contain exactly 3 items" in str(exc_info.value)


def test_fixtures_dir_fixture(fixtures_dir: Path) -> None:
    """Ensure fixtures_dir points to existing directory containing json files."""
    assert fixtures_dir.is_dir()
    assert (fixtures_dir / "queue_messages" / "analyze_session.json").is_file()
