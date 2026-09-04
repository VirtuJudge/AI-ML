import pytest

import app.providers.types as provider_types
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    EraseAIDataPayload,
    ErasureCompleted,
    GenerateReportPayload,
    ReportCompleted,
    RubricRef,
    SessionAnalysisCompleted,
    SpeakerMapping,
)
from app.pipeline import FakePipeline

FAKE_SHA256 = "sha256:" + "a" * 64


@pytest.mark.asyncio
async def test_complete_product_journey(journey_pipeline: FakePipeline) -> None:
    """Execute the end-to-end flow: session -> Q&A answers -> followups -> report -> erasure."""
    # 1. Analyze Session
    session_job = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01J_PRES_000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum=FAKE_SHA256,
            media_type="video/mp4",
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01J_DOC_000000000000000001",
                object_key="uploads/deck.pdf",
                checksum=FAKE_SHA256,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=[
            "speech",
            "diarization",
            "vision",
            "audio",
            "documents",
            "questions",
        ],
    )
    session_result = await journey_pipeline.analyze_session(session_job)
    assert isinstance(session_result, SessionAnalysisCompleted)
    assert len(session_result.primary_questions) == 3
    assert len(session_result.speaker_labels) == 2
    assert session_result.analysis_artifact.schema_version == 1

    # 2. Analyze Answers for each primary question
    qa_answers: list[AnswerAnalysisCompleted] = []
    for idx, question in enumerate(session_result.primary_questions):
        answer_job = AnalyzeAnswerPayload(
            qa_round_id=f"01J_ROUND_{idx}",
            question_id=question.candidate_id,
            answer_id=f"01J_ANS_{idx}",
            answered_by="user_01",
            audio=AudioAssetInput(
                artifact_id=f"01J_AUDIO_{idx}",
                object_key=f"answers/ans_{idx}.wav",
                checksum=FAKE_SHA256,
                media_type="audio/wav",
                duration_ms=45000,
            ),
            remaining_follow_ups=1,
        )
        answer_result = await journey_pipeline.analyze_answer(answer_job)
        assert isinstance(answer_result, AnswerAnalysisCompleted)
        assert answer_result.follow_up is not None
        qa_answers.append(answer_result)

    # 3. Analyze Answer for follow-up (when remaining_follow_ups = 0)
    follow_up_job = AnalyzeAnswerPayload(
        qa_round_id="01J_ROUND_FOLLOWUP",
        question_id="01J_Q_FOLLOWUP",
        answer_id="01J_ANS_FOLLOWUP",
        answered_by="user_01",
        audio=AudioAssetInput(
            artifact_id="01J_AUDIO_FOLLOWUP",
            object_key="answers/ans_followup.wav",
            checksum=FAKE_SHA256,
            media_type="audio/wav",
            duration_ms=25000,
        ),
        remaining_follow_ups=0,
    )
    follow_up_result = await journey_pipeline.analyze_answer(follow_up_job)
    assert isinstance(follow_up_result, AnswerAnalysisCompleted)
    assert follow_up_result.follow_up is None

    # 4. Generate Report (feeding analysis_artifact from session into report generation)
    report_job = GenerateReportPayload(
        report_id="01J_REPORT_000000000000001",
        analysis_artifact=session_result.analysis_artifact,
        qa_artifact=ArtifactRef(
            artifact_id="01J_QA_ARTIFACT_00000001",
            object_key="artifacts/qa_aggregate.json",
            checksum=FAKE_SHA256,
            schema_version=1,
        ),
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="user_alice_01",
                display_name="Alice Founder",
            ),
            SpeakerMapping(
                speaker_label="SPEAKER_01",
                user_id="user_bob_02",
                display_name="Bob CTO",
            ),
        ],
    )
    report_result = await journey_pipeline.generate_report(report_job)
    assert isinstance(report_result, ReportCompleted)
    assert report_result.evaluation_artifact.schema_version == 1
    assert report_result.report_artifact.schema_version == 1
    assert "user_alice_01" in report_result.member_feedback_user_ids
    assert "user_bob_02" in report_result.member_feedback_user_ids

    # 5. Erase AI Data
    erasure_job = EraseAIDataPayload(
        erasure_request_id="01J_ERASURE_000000000001",
        scope="practice_session",
        scope_id="01J_SESSION_0000000000001",
    )
    erasure_result = await journey_pipeline.erase_data(erasure_job)
    assert isinstance(erasure_result, ErasureCompleted)
    assert erasure_result.deleted_records == 14
    assert erasure_result.deleted_objects == 6


@pytest.mark.asyncio
async def test_journey_results_validate_against_fixtures(journey_pipeline: FakePipeline) -> None:
    """Verify journey results round-trip through JSON and validate strictly against models."""
    session_job = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01J_PRES_000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum=FAKE_SHA256,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )
    session_result = await journey_pipeline.analyze_session(session_job)
    session_json = session_result.model_dump_json()
    validated_session = SessionAnalysisCompleted.model_validate_json(session_json)
    assert validated_session == session_result

    answer_job = AnalyzeAnswerPayload(
        qa_round_id="01J_ROUND_0",
        question_id="01J_Q_0",
        answer_id="01J_ANS_0",
        answered_by="user_01",
        audio=AudioAssetInput(
            artifact_id="01J_AUDIO_0",
            object_key="answers/ans_0.wav",
            checksum=FAKE_SHA256,
            media_type="audio/wav",
            duration_ms=30000,
        ),
        remaining_follow_ups=1,
    )
    answer_result = await journey_pipeline.analyze_answer(answer_job)
    answer_json = answer_result.model_dump_json()
    validated_answer = AnswerAnalysisCompleted.model_validate_json(answer_json)
    assert validated_answer == answer_result

    report_job = GenerateReportPayload(
        report_id="01J_REPORT_0001",
        analysis_artifact=session_result.analysis_artifact,
        qa_artifact=ArtifactRef(
            artifact_id="01J_QA_0001",
            object_key="artifacts/qa.json",
            checksum=FAKE_SHA256,
            schema_version=1,
        ),
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="user_01",
                display_name="Founder",
            )
        ],
    )
    report_result = await journey_pipeline.generate_report(report_job)
    report_json = report_result.model_dump_json()
    validated_report = ReportCompleted.model_validate_json(report_json)
    assert validated_report == report_result


@pytest.mark.asyncio
async def test_provider_types_not_in_boundary_results(journey_pipeline: FakePipeline) -> None:
    """Ensure no raw provider instances appear inside pipeline boundary return values."""
    provider_type_classes = tuple(
        cls
        for name, cls in vars(provider_types).items()
        if isinstance(cls, type)
        and issubclass(cls, provider_types.BaseModel)
        and cls is not provider_types.BaseModel
    )

    session_job = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01J_PRES_000000000000000001",
            object_key="uploads/presentation.mp4",
            checksum=FAKE_SHA256,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech"],
    )
    session_result = await journey_pipeline.analyze_session(session_job)

    # Traverse fields and ensure none is an instance of a provider type
    for field_name, value in session_result:
        assert not isinstance(value, provider_type_classes), f"{field_name} is a provider type"
        if isinstance(value, list):
            for item in value:
                assert not isinstance(item, provider_type_classes)
