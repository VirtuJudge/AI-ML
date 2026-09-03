"""Pipeline protocols and test implementations for pitch analysis."""

from typing import Protocol

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    EraseAIDataPayload,
    ErasureCompleted,
    FollowUpQuestion,
    GenerateReportPayload,
    PrimaryQuestion,
    ReportCompleted,
    SessionAnalysisCompleted,
)

FAKE_SESSION_ANALYSIS_ARTIFACT_ID = "01JEXAMPLE000000000000000A"
FAKE_QUESTION_1_ID = "01JEXAMPLE000000000000001A"
FAKE_QUESTION_2_ID = "01JEXAMPLE000000000000001B"
FAKE_QUESTION_3_ID = "01JEXAMPLE000000000000001C"
FAKE_TRANSCRIPT_ARTIFACT_ID = "01JEXAMPLE000000000000002A"
FAKE_ASSESSMENT_ARTIFACT_ID = "01JEXAMPLE000000000000002B"
FAKE_EVALUATION_ARTIFACT_ID = "01JEXAMPLE000000000000003A"
FAKE_REPORT_ARTIFACT_ID = "01JEXAMPLE000000000000003B"

FAKE_SHA256_A = "sha256:" + "a" * 64
FAKE_SHA256_B = "sha256:" + "b" * 64
FAKE_SHA256_C = "sha256:" + "c" * 64


class PitchAnalysisPipeline(Protocol):
    """Protocol defining the interface for pitch analysis pipeline operations."""

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted: ...

    async def analyze_answer(self, job: AnalyzeAnswerPayload) -> AnswerAnalysisCompleted: ...

    async def generate_report(self, job: GenerateReportPayload) -> ReportCompleted: ...

    async def erase_data(self, job: EraseAIDataPayload) -> ErasureCompleted: ...


class FakePipeline:
    """Deterministic fake pipeline implementation for tests and local development."""

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted:
        return SessionAnalysisCompleted(
            analysis_artifact=ArtifactRef(
                artifact_id=FAKE_SESSION_ANALYSIS_ARTIFACT_ID,
                object_key="artifacts/session_analysis.json",
                checksum=FAKE_SHA256_A,
                schema_version=1,
            ),
            primary_questions=[
                PrimaryQuestion(
                    candidate_id=FAKE_QUESTION_1_ID,
                    text="What is your projected customer acquisition cost at scale, "
                    "and what channels drive that estimate?",
                    reason="Validates financial feasibility and go-to-market model assumptions.",
                    rubric_dimension="market_and_business_model",
                    evidence_ids=["evidence_slide_04", "evidence_speech_01"],
                ),
                PrimaryQuestion(
                    candidate_id=FAKE_QUESTION_2_ID,
                    text="How does your core proprietary technology maintain its defensive moat "
                    "against incumbent fast-followers?",
                    reason="Assesses technical defensibility and differentiation.",
                    rubric_dimension="technology_and_moat",
                    evidence_ids=["evidence_slide_07"],
                ),
                PrimaryQuestion(
                    candidate_id=FAKE_QUESTION_3_ID,
                    text="What specific milestones must be achieved during the initial pilot phase "
                    "to secure renewal commitments?",
                    reason="Evaluates execution roadmap and early customer validation.",
                    rubric_dimension="execution_and_milestones",
                    evidence_ids=["evidence_slide_09", "evidence_speech_02"],
                ),
            ],
            speaker_labels=["SPEAKER_00", "SPEAKER_01"],
            limitations=[],
        )

    async def analyze_answer(self, job: AnalyzeAnswerPayload) -> AnswerAnalysisCompleted:
        follow_up: FollowUpQuestion | None = None
        if job.remaining_follow_ups > 0:
            follow_up = FollowUpQuestion(
                text="Could you break down the pilot retention metrics across "
                "your key enterprise segments?",
                reason="Clarifies customer retention resilience mentioned in the previous answer.",
                rubric_dimension="customer_retention",
                evidence_ids=["evidence_answer_01"],
            )

        return AnswerAnalysisCompleted(
            answer_id=job.answer_id,
            transcript_artifact_id=FAKE_TRANSCRIPT_ARTIFACT_ID,
            assessment_artifact_id=FAKE_ASSESSMENT_ARTIFACT_ID,
            follow_up=follow_up,
        )

    async def generate_report(self, job: GenerateReportPayload) -> ReportCompleted:
        return ReportCompleted(
            evaluation_artifact=ArtifactRef(
                artifact_id=FAKE_EVALUATION_ARTIFACT_ID,
                object_key="artifacts/evaluation.json",
                checksum=FAKE_SHA256_B,
                schema_version=1,
            ),
            report_artifact=ArtifactRef(
                artifact_id=FAKE_REPORT_ARTIFACT_ID,
                object_key="artifacts/report.json",
                checksum=FAKE_SHA256_C,
                schema_version=1,
            ),
            member_feedback_user_ids=[mapping.user_id for mapping in job.speaker_mappings],
            limitations=[],
        )

    async def erase_data(self, job: EraseAIDataPayload) -> ErasureCompleted:
        return ErasureCompleted(
            erasure_request_id=job.erasure_request_id,
            deleted_records=14,
            deleted_objects=6,
        )


__all__ = ["FakePipeline", "PitchAnalysisPipeline"]
