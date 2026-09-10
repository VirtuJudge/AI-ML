import asyncio
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
    Limitation,
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
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    VisualObservation,
)
from app.stages.audio import run_audio_stage
from app.stages.media import split_media
from app.stages.speech import run_speech_stage
from app.stages.vision import run_vision_stage

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


def combine_timed_evidence(
    diarization: DiarizationResult,
    visual_observations: list[VisualObservation],
    audio_observations: list[AudioObservation],
) -> tuple[list[VisualObservation], list[AudioObservation], dict[str, str], list[Limitation]]:
    """Cross-correlate visual and acoustic observations with diarized speaker turns.

    1. Attributes sliding-window audio observations to active speaker turns.
    2. Correlates visual person tracks (PERSON_XX) to canonical speaker labels (SPEAKER_XX)
       using mouth_aspect_ratio (lip activity) and temporal presence during speaker turns.
    3. Deduplicates fragmented person tracks (e.g. PERSON_00 and PERSON_02 -> SPEAKER_00).
    4. Re-labels visual observations with canonical speaker labels.
    """
    valid_speakers = [
        s for s in diarization.speaker_labels
        if s not in ("SPEAKER_UNKNOWN", "UNKNOWN", "")
    ]

    # --- 1. Audio Attribution ---
    updated_audio: list[AudioObservation] = []
    for a_obs in audio_observations:
        if a_obs.speaker_label:
            updated_audio.append(a_obs)
            continue

        overlap_by_speaker: dict[str, int] = {}
        for seg in diarization.segments:
            if seg.speaker_label not in valid_speakers:
                continue
            overlap = max(0, min(a_obs.end_ms, seg.end_ms) - max(a_obs.start_ms, seg.start_ms))
            if overlap > 0:
                overlap_by_speaker[seg.speaker_label] = (
                    overlap_by_speaker.get(seg.speaker_label, 0) + overlap
                )

        best_spk = None
        if overlap_by_speaker:
            best_spk = max(overlap_by_speaker.items(), key=lambda item: item[1])[0]

        if best_spk:
            updated_audio.append(a_obs.model_copy(update={"speaker_label": best_spk}))
        else:
            updated_audio.append(a_obs)

    # --- 2. Visual Person -> Speaker Correlation ---
    person_ids = {
        v.speaker_label
        for v in visual_observations
        if v.speaker_label and v.speaker_label.startswith("PERSON_")
    }

    mar_activity: dict[str, dict[str, float]] = {pid: {} for pid in person_ids}
    presence_time: dict[str, dict[str, int]] = {pid: {} for pid in person_ids}

    for v_obs in visual_observations:
        pid = v_obs.speaker_label
        if not pid or pid not in person_ids:
            continue

        for seg in diarization.segments:
            if seg.speaker_label not in valid_speakers:
                continue
            overlap = max(0, min(v_obs.end_ms, seg.end_ms) - max(v_obs.start_ms, seg.start_ms))
            if overlap <= 0:
                continue

            presence_time[pid][seg.speaker_label] = (
                presence_time[pid].get(seg.speaker_label, 0) + overlap
            )

            if v_obs.metric == "mouth_aspect_ratio":
                activity = max(0.0, v_obs.value) * overlap
                mar_activity[pid][seg.speaker_label] = (
                    mar_activity[pid].get(seg.speaker_label, 0.0) + activity
                )

    person_to_speaker: dict[str, str] = {}
    limitations: list[Limitation] = []

    for pid in sorted(person_ids):
        best_score = -1.0
        best_speaker = None
        for spk in valid_speakers:
            mar_score = mar_activity[pid].get(spk, 0.0)
            pres_score = float(presence_time[pid].get(spk, 0))
            combined_score = (mar_score * 10.0) + pres_score
            if combined_score > best_score and combined_score > 0:
                best_score = combined_score
                best_speaker = spk

        if best_speaker:
            person_to_speaker[pid] = best_speaker
        elif len(valid_speakers) == 1:
            person_to_speaker[pid] = valid_speakers[0]
        else:
            limitations.append(
                Limitation(
                    code="ambiguous_visual_speaker_mapping",
                    scope="vision",
                    message=f"Visual track {pid} could not be correlated with an active speaker.",
                    affected_dimensions=["individual_feedback", "visual_delivery"],
                )
            )

    # --- 3. Relabel Visual Observations ---
    updated_visual: list[VisualObservation] = []
    for v_obs in visual_observations:
        if v_obs.speaker_label in person_to_speaker:
            updated_visual.append(
                v_obs.model_copy(update={"speaker_label": person_to_speaker[v_obs.speaker_label]})
            )
        elif not v_obs.speaker_label and valid_speakers:
            best_spk = None
            best_overlap = 0
            for seg in diarization.segments:
                if seg.speaker_label not in valid_speakers:
                    continue
                overlap = max(0, min(v_obs.end_ms, seg.end_ms) - max(v_obs.start_ms, seg.start_ms))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = seg.speaker_label
            if best_spk:
                updated_visual.append(v_obs.model_copy(update={"speaker_label": best_spk}))
            else:
                updated_visual.append(v_obs)
        else:
            updated_visual.append(v_obs)

    return updated_visual, updated_audio, person_to_speaker, limitations


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

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted:
        input_file = Path(job.presentation.object_key)
        skip_file_check = not input_file.is_file()

        if not skip_file_check:
            split_res = await split_media(input_file)
            audio_path: Path | None = split_res.audio_path
            video_path: Path | None = split_res.video_path
            duration_ms = split_res.duration_ms or job.presentation.duration_ms or 15000
        else:
            audio_path = input_file
            video_path = input_file
            duration_ms = job.presentation.duration_ms or 15000

        speech_res, vision_res, audio_res = await asyncio.gather(
            run_speech_stage(
                audio_path,
                self.speech_provider,
                self.diarization_provider,
                media_duration_ms=duration_ms,
                skip_normalization=True,
            ),
            run_vision_stage(
                video_path,
                self.vision_provider,
                media_duration_ms=duration_ms,
                source_artifact_id=job.presentation.artifact_id,
                skip_file_check=skip_file_check,
            ),
            run_audio_stage(
                audio_path,
                self.audio_provider,
                media_duration_ms=duration_ms,
                source_artifact_id=job.presentation.artifact_id,
                skip_file_check=skip_file_check,
            ),
        )

        _, _, _, corr_limitations = combine_timed_evidence(
            speech_res.diarization,
            vision_res.observations,
            audio_res.observations,
        )

        all_limitations = (
            speech_res.limitations
            + vision_res.limitations
            + audio_res.limitations
            + corr_limitations
        )

        doc_chunks = []
        for doc in job.supporting_documents:
            chunks = await self.document_provider.extract_and_embed(Path(doc.object_key))
            doc_chunks.extend(chunks)

        questions = await self.judge_provider.generate_questions(
            transcript=speech_res.transcription.full_text,
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
            speaker_labels=speech_res.speaker_labels,
            limitations=all_limitations,
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


__all__ = ["FakePipeline", "PitchAnalysisPipeline", "combine_timed_evidence"]
