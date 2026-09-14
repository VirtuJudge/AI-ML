"""Unit tests for contract models and validations."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.contracts import (
    ArtifactRef,
    Evaluation,
    FeedbackSection,
    Finding,
    JobType,
    Limitation,
    MemberFeedback,
    PrimaryQuestion,
    ProgressPayload,
    QueueMessage,
    RubricRef,
    ScoreComponent,
    SessionAnalysisCompleted,
    SpeakingInterval,
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


def test_analyze_answer_payload_optional_audio() -> None:
    """Verify AnalyzeAnswerPayload accepts None, omitted audio, and null for skipped answers."""
    from app.contracts import AnalyzeAnswerPayload, AudioAssetInput

    # 1. audio omitted
    p1 = AnalyzeAnswerPayload.model_validate(
        {
            "qa_round_id": "qa_01",
            "question_id": "q_01",
            "answer_id": "ans_01",
            "answered_by": "user_01",
            "remaining_follow_ups": 2,
        }
    )
    assert p1.audio is None

    # 2. audio: None
    p2 = AnalyzeAnswerPayload.model_validate(
        {
            "qa_round_id": "qa_01",
            "question_id": "q_01",
            "answer_id": "ans_01",
            "answered_by": "user_01",
            "audio": None,
            "remaining_follow_ups": 0,
        }
    )
    assert p2.audio is None

    # 3. audio as AudioAssetInput dict
    p3 = AnalyzeAnswerPayload.model_validate(
        {
            "qa_round_id": "qa_01",
            "question_id": "q_01",
            "answer_id": "ans_01",
            "answered_by": "user_01",
            "audio": {
                "artifact_id": "art_01",
                "object_key": "audio.wav",
                "checksum": "sha256:" + "a" * 64,
                "media_type": "audio/wav",
                "duration_ms": 5000,
            },
            "remaining_follow_ups": 1,
        }
    )
    assert isinstance(p3.audio, AudioAssetInput)
    assert p3.audio.duration_ms == 5000


def test_score_component_model() -> None:
    """Verify ScoreComponent validates scored and not_evaluated states."""
    scored = ScoreComponent(
        dimension="pitch_content_and_evidence",
        status="scored",
        normalized_score=0.85,
        display_score=85,
        label="strong",
        configured_weight=0.25,
        effective_weight=0.25,
        evidence_ids=["ev_speech_001"],
        rationale="Clear evidence-grounded problem statement.",
    )
    assert scored.status == "scored"
    assert scored.normalized_score == 0.85
    assert scored.display_score == 85
    assert scored.label == "strong"

    unscored = ScoreComponent(
        dimension="delivery_and_body_language",
        status="not_evaluated",
        configured_weight=0.15,
        limitation_code="no_video_stream",
        rationale="Video was unavailable.",
    )
    assert unscored.status == "not_evaluated"
    assert unscored.normalized_score is None
    assert unscored.display_score is None
    assert unscored.limitation_code == "no_video_stream"


def test_finding_model() -> None:
    """Verify Finding accepts all valid kinds and metadata."""
    finding = Finding(
        id="find_001",
        kind="strength",
        title="Compelling Value Proposition",
        detail="The founder articulated the value proposition within the first 60 seconds.",
        recommendation="Maintain this strong opener in future pitches.",
        evidence_ids=["ev_speech_001", "ev_doc_slide_01"],
        rubric_dimension="pitch_content_and_evidence",
        speaker_labels=["SPEAKER_00"],
    )
    assert finding.kind == "strength"
    assert finding.speaker_labels == ["SPEAKER_00"]
    assert len(finding.evidence_ids) == 2


def test_speaking_interval_model() -> None:
    """Verify SpeakingInterval model constraints."""
    interval = SpeakingInterval(start_ms=0, end_ms=135000, formatted="00:00 - 02:15")
    assert interval.start_ms == 0
    assert interval.end_ms == 135000
    assert interval.formatted == "00:00 - 02:15"

    with pytest.raises(ValidationError):
        SpeakingInterval(start_ms=-10, end_ms=5000, formatted="invalid")


def test_member_feedback_model() -> None:
    """Verify MemberFeedback model with speaking intervals and delivery components."""
    member = MemberFeedback(
        user_id="user_founder_1",
        display_name="Jane Founder",
        speaker_labels=["SPEAKER_00"],
        speaking_intervals=[
            SpeakingInterval(start_ms=0, end_ms=135000, formatted="00:00 - 02:15"),
        ],
        speaking_time_ms=135000,
        summary="Clear and confident delivery with excellent pacing.",
        strengths=[
            Finding(
                id="f_m1",
                kind="strength",
                title="Pacing",
                detail="Maintained 138 WPM throughout.",
                evidence_ids=["ev_speech_001"],
            )
        ],
        improvements=[],
        delivery_components=[
            ScoreComponent(
                dimension="delivery_and_body_language",
                status="scored",
                normalized_score=0.88,
                display_score=88,
                label="strong",
                configured_weight=0.15,
            )
        ],
    )
    assert member.user_id == "user_founder_1"
    assert len(member.speaking_intervals) == 1
    assert member.speaking_intervals[0].formatted == "00:00 - 02:15"
    assert member.speaking_time_ms == 135000


def test_evaluation_round_trip() -> None:
    """Verify complete serialization and deserialization of Evaluation artifact."""
    evaluation = Evaluation(
        id="01JEVALUATION0000000000001",
        analysis_attempt_id="01JANALYSIS00000000000001",
        qa_round_id="01JQAROUND0000000000000001",
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        overall_score=0.78,
        components=[
            ScoreComponent(
                dimension="pitch_content_and_evidence",
                status="scored",
                normalized_score=0.80,
                display_score=80,
                label="strong",
                configured_weight=0.25,
                effective_weight=0.25,
                evidence_ids=["ev_speech_001"],
            ),
            ScoreComponent(
                dimension="qa_quality",
                status="scored",
                normalized_score=0.75,
                display_score=75,
                label="good",
                configured_weight=0.20,
                effective_weight=0.20,
                evidence_ids=["ev_qa_001"],
            ),
        ],
        findings=[
            Finding(
                id="f_team_1",
                kind="strength",
                title="Market Sizing",
                detail="Clear bottom-up TAM breakdown.",
                evidence_ids=["ev_doc_slide_02"],
            )
        ],
        team_feedback=FeedbackSection(
            summary="Strong overall pitch with rigorous defensibility and solid team coordination.",
            strengths=[
                Finding(
                    id="f_team_1",
                    kind="strength",
                    title="Market Sizing",
                    detail="Clear bottom-up TAM breakdown.",
                    evidence_ids=["ev_doc_slide_02"],
                )
            ],
            improvements=[],
            score_components=[],
            limitations=[],
        ),
        member_feedback=[
            MemberFeedback(
                user_id="user_founder_1",
                display_name="Jane Founder",
                speaker_labels=["SPEAKER_00"],
                speaking_intervals=[
                    SpeakingInterval(start_ms=0, end_ms=135000, formatted="00:00 - 02:15")
                ],
                speaking_time_ms=135000,
                summary="Dynamic delivery with natural transitions.",
                strengths=[],
                improvements=[],
                delivery_components=[],
            )
        ],
        limitations=[],
        reproducibility={"generator": "VirtuJudge Judge Model v1", "temperature": 0.0},
    )

    json_str = evaluation.model_dump_json()
    reconstituted = Evaluation.model_validate_json(json_str)

    assert reconstituted.id == evaluation.id
    assert reconstituted.schema_version == 1
    assert reconstituted.overall_score == 0.78
    assert len(reconstituted.components) == 2
    assert reconstituted.team_feedback.summary.startswith("Strong overall pitch")
    assert len(reconstituted.member_feedback) == 1
    assert reconstituted.member_feedback[0].speaking_intervals[0].formatted == "00:00 - 02:15"


def test_speaker_mapping_optional_display_name() -> None:
    """Verify SpeakerMapping accepts optional display_name conforming to data contracts."""
    from app.contracts import SpeakerMapping

    # With display_name
    m1 = SpeakerMapping(speaker_label="SPEAKER_00", user_id="u1", display_name="Jane")
    assert m1.display_name == "Jane"

    # Without display_name (defaults to None)
    m2 = SpeakerMapping(speaker_label="SPEAKER_01", user_id="u2")
    assert m2.display_name is None


