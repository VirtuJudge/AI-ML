from pathlib import Path
from typing import Any, Protocol

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    EraseAIDataPayload,
    ErasureCompleted,
    FollowUpQuestion,
    GenerateReportPayload,
    ReportCompleted,
    SessionAnalysisCompleted,
)
from app.providers.base import (
    AudioMetricsProvider,
    DiarizationProvider,
    DocumentProvider,
    JudgeModelProvider,
    SpeechProvider,
    VisionProvider,
)
from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.fake_documents import FakeDocumentProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.fake_vision import FakeVisionProvider

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
    """Deterministic fake pipeline implementation composed from swappable stage adapters."""

    def __init__(
        self,
        speech_provider: SpeechProvider | None = None,
        diarization_provider: DiarizationProvider | None = None,
        vision_provider: VisionProvider | None = None,
        audio_provider: AudioMetricsProvider | None = None,
        document_provider: DocumentProvider | None = None,
        judge_provider: JudgeModelProvider | None = None,
    ) -> None:
        self.speech_provider = speech_provider or FakeSpeechProvider()
        self.diarization_provider = diarization_provider or FakeDiarizationProvider()
        self.vision_provider = vision_provider or FakeVisionProvider()
        self.audio_provider = audio_provider or FakeAudioMetricsProvider()
        self.document_provider = document_provider or FakeDocumentProvider()
        self.judge_provider = judge_provider or FakeJudgeModelProvider()
        self.last_vision_observations: list[Any] = []
        self.last_audio_observations: list[Any] = []

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted:
        audio_path = Path(job.presentation.object_key)
        transcript = await self.speech_provider.transcribe(audio_path)
        diarization = await self.diarization_provider.diarize(audio_path)
        self.last_vision_observations = await self.vision_provider.analyze_video(audio_path)
        self.last_audio_observations = await self.audio_provider.extract_metrics(audio_path)

        doc_chunks = []
        for doc in job.supporting_documents:
            chunks = await self.document_provider.extract_and_embed(Path(doc.object_key))
            doc_chunks.extend(chunks)

        questions = await self.judge_provider.generate_questions(
            transcript=transcript.full_text,
            rubric_id=job.rubric.rubric_id,
            document_chunks=doc_chunks if doc_chunks else None,
        )

        return SessionAnalysisCompleted(
            analysis_artifact=ArtifactRef(
                artifact_id=FAKE_SESSION_ANALYSIS_ARTIFACT_ID,
                object_key="artifacts/session_analysis.json",
                checksum=FAKE_SHA256_A,
                schema_version=1,
            ),
            primary_questions=questions,
            speaker_labels=diarization.speaker_labels,
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
