"""Unit tests for contract models and validations."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.contracts import (
    ArtifactRef,
    JobType,
    Limitation,
    PrimaryQuestion,
    ProgressPayload,
    QueueMessage,
    SessionAnalysisCompleted,
)


def test_queue_message_round_trip(analyze_session_message: QueueMessage) -> None:
    """Test serialization and deserialization of a QueueMessage."""
    json_data = analyze_session_message.model_dump_json()
    deserialized = QueueMessage.model_validate_json(json_data)
    assert deserialized == analyze_session_message

    dict_data = analyze_session_message.model_dump()
    from_dict = QueueMessage.model_validate(dict_data)
    assert from_dict == analyze_session_message


def test_analyze_session_completed_requires_three_questions() -> None:
    """Test that SessionAnalysisCompleted requires exactly 3 primary questions."""
    artifact = ArtifactRef(
        artifact_id="01JTEST0000000000000000099",
        object_key="artifacts/test.json",
        checksum="sha256:" + "a" * 64,
        schema_version=1,
    )
    questions = [
        PrimaryQuestion(
            candidate_id=f"01JTEST00000000000000000{i:02d}",
            text=f"Question {i}",
            reason=f"Reason {i}",
            rubric_dimension="market",
            evidence_ids=[f"ev_{i}"],
        )
        for i in range(1, 3)
    ]

    # Exactly 2 questions must raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        SessionAnalysisCompleted(
            analysis_artifact=artifact,
            primary_questions=questions,
            speaker_labels=["SPEAKER_00"],
            limitations=[],
        )
    assert "primary_questions" in str(exc_info.value)

    # 4 questions must also raise ValidationError
    extra_question = PrimaryQuestion(
        candidate_id="01JTEST0000000000000000093",
        text="Question 3",
        reason="Reason 3",
        rubric_dimension="market",
        evidence_ids=["ev_3"],
    )
    fourth_question = PrimaryQuestion(
        candidate_id="01JTEST0000000000000000094",
        text="Question 4",
        reason="Reason 4",
        rubric_dimension="market",
        evidence_ids=["ev_4"],
    )
    with pytest.raises(ValidationError):
        SessionAnalysisCompleted(
            analysis_artifact=artifact,
            primary_questions=[*questions, extra_question, fourth_question],
        )

    # Exactly 3 questions must succeed
    valid = SessionAnalysisCompleted(
        analysis_artifact=artifact,
        primary_questions=[*questions, extra_question],
        speaker_labels=["SPEAKER_00"],
        limitations=[],
    )
    assert len(valid.primary_questions) == 3


def test_artifact_ref_rejects_bad_checksum() -> None:
    """Test that creating ArtifactRef with 'md5:abc' raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ArtifactRef(
            artifact_id="01JTEST0000000000000000099",
            object_key="artifacts/test.json",
            checksum="md5:abc",
        )
    assert "checksum" in str(exc_info.value)

    # Valid sha256 prefix + 64 hex characters succeeds
    valid_ref = ArtifactRef(
        artifact_id="01JTEST0000000000000000099",
        object_key="artifacts/test.json",
        checksum="sha256:" + "0" * 64,
    )
    assert valid_ref.checksum.startswith("sha256:")


def test_normalized_score_range() -> None:
    """Test that ProgressPayload rejects progress outside 0.0-1.0."""
    with pytest.raises(ValidationError):
        ProgressPayload(stage="transcription", progress=1.5, message="Processing")

    with pytest.raises(ValidationError):
        ProgressPayload(stage="transcription", progress=-0.1, message="Processing")

    valid_payload = ProgressPayload(stage="transcription", progress=0.75, message="Processing")
    assert valid_payload.progress == 0.75


def test_all_job_types_valid() -> None:
    """Test that QueueMessage accepts all JobType enum members."""
    for job_type in JobType:
        msg = QueueMessage(
            schema_version=1,
            job_id="01JTEST0000000000000000001",
            job_type=job_type,
            practice_session_id="01JTEST0000000000000000002",
            analysis_attempt=1,
            created_at=datetime.now(UTC),
            trace_id="01JTEST0000000000000000003",
            payload={},
        )
        assert msg.job_type == job_type


def test_session_analysis_completed_serialization() -> None:
    """Test serialization and deserialization round-trip for SessionAnalysisCompleted."""
    completed = SessionAnalysisCompleted(
        analysis_artifact=ArtifactRef(
            artifact_id="01JTEST0000000000000000001",
            object_key="artifacts/session_analysis.json",
            checksum="sha256:" + "f" * 64,
            schema_version=1,
        ),
        primary_questions=[
            PrimaryQuestion(
                candidate_id="01JTEST0000000000000000002",
                text="How do you handle customer churn?",
                reason="Assess retention strategy",
                rubric_dimension="retention",
                evidence_ids=["ev_1"],
            ),
            PrimaryQuestion(
                candidate_id="01JTEST0000000000000000003",
                text="What are the unit economics?",
                reason="Assess financial sustainability",
                rubric_dimension="financials",
                evidence_ids=["ev_2"],
            ),
            PrimaryQuestion(
                candidate_id="01JTEST0000000000000000004",
                text="What is your unfair advantage?",
                reason="Assess moat",
                rubric_dimension="differentiation",
                evidence_ids=["ev_3"],
            ),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        limitations=[
            Limitation(
                code="LOW_AUDIO_SNR",
                scope="audio",
                message="Background noise detected during pitch",
                affected_dimensions=["delivery"],
            )
        ],
    )

    serialized = completed.model_dump_json()
    deserialized = SessionAnalysisCompleted.model_validate_json(serialized)
    assert deserialized == completed
