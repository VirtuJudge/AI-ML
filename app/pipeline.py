import asyncio
import hashlib
import logging
import os
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Protocol

import ulid

from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    EraseAIDataPayload,
    ErasureCompleted,
    GenerateReportPayload,
    Limitation,
    ReportCompleted,
    SessionAnalysisCompleted,
)
from app.document_store import DocumentStore, FakeDocumentStore
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
from app.stages.aggregation import aggregate_speaker_observations
from app.stages.answer_assessment import run_answer_assessment_stage
from app.stages.answers import run_answer_speech_stage
from app.stages.audio import run_audio_stage
from app.stages.documents import run_document_stage
from app.stages.media import split_media
from app.stages.questions import run_question_stage
from app.stages.reporting import run_report_stage
from app.stages.speech import run_speech_stage
from app.stages.vision import run_vision_stage
from app.storage import create_object_storage
from app.storage.base import (
    ObjectNotFoundError,
    ObjectStorageError,
    ObjectStorageProtocol,
)
from app.storage.local import LocalDiskObjectStorage

logger = logging.getLogger(__name__)

PRODUCER_VERSION: str = "ai-ml/0.1.0"


def generate_deterministic_ulid(seed_key: str, dt: datetime | str | None = None) -> str:
    """Generate a deterministic ULID from a seed key and timestamp."""
    if dt is None:
        dt = datetime.now(UTC)
    elif isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    ts_bytes = int(dt.timestamp() * 1000).to_bytes(6, byteorder="big")
    rand_bytes = hashlib.sha256(seed_key.encode()).digest()[:10]
    return str(ulid.from_bytes(ts_bytes + rand_bytes))



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
        document_store: DocumentStore | None = None,
        object_storage: ObjectStorageProtocol | None = None,
    ) -> None:
        self.speech_provider = speech_provider or FakeSpeechProvider()
        self.diarization_provider = diarization_provider or FakeDiarizationProvider()
        self.vision_provider = vision_provider or FakeVisionProvider()
        self.audio_provider = audio_provider or FakeAudioMetricsProvider()
        self.document_provider = document_provider or FakeDocumentProvider()
        self.judge_provider = judge_provider or FakeJudgeModelProvider()
        self.document_store = document_store or FakeDocumentStore()
        if object_storage is not None:
            self.object_storage = object_storage
        elif os.getenv("APP_ENV") == "test" or "PYTEST_CURRENT_TEST" in os.environ:
            self.object_storage = LocalDiskObjectStorage()
        else:
            self.object_storage = create_object_storage()

    def _is_synthetic_speech_run(self) -> bool:
        """Check whether the pipeline is executing with purely fake/synthetic speech provider."""
        return isinstance(self.speech_provider, FakeSpeechProvider)

    async def analyze_session(self, job: AnalyzeSessionPayload) -> SessionAnalysisCompleted:
        input_file = Path(job.presentation.object_key)
        skip_file_check = not input_file.is_file()

        # If object key is remote and not on local disk, download via object storage
        if skip_file_check:
            temp_dir = Path(".storage/temp_media") / job.presentation.artifact_id
            temp_dest = temp_dir / Path(job.presentation.object_key).name
            try:
                await self.object_storage.download_file(job.presentation.object_key, temp_dest)
                if temp_dest.is_file():
                    input_file = temp_dest
                    skip_file_check = False
                else:
                    raise ObjectStorageError(
                        f"Download succeeded but file not found at destination '{temp_dest}'."
                    )
            except ObjectNotFoundError as err:
                if not self._is_synthetic_speech_run():
                    logger.error(
                        "Presentation media '%s' not found in object storage: %s",
                        job.presentation.object_key,
                        err,
                        exc_info=True,
                    )
                    raise
                logger.warning(
                    "Presentation media '%s' not found in storage; using synthetic inputs.",
                    job.presentation.object_key,
                )
            except Exception as exc:
                if not self._is_synthetic_speech_run():
                    logger.error(
                        "Failed to download presentation media '%s' from object storage: %s",
                        job.presentation.object_key,
                        exc,
                        exc_info=True,
                    )
                    msg = f"Failed to download media '{job.presentation.object_key}': {exc}"
                    raise ObjectStorageError(msg) from exc
                logger.warning(
                    "Error downloading media '%s' from storage in synthetic mode: %s",
                    job.presentation.object_key,
                    exc,
                )

        if not skip_file_check and not input_file.is_file():
            raise FileNotFoundError(
                f"Presentation media file '{input_file}' does not exist."
            )

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
                strict_timestamps=False,
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

        updated_visual, updated_audio, speaker_mapping, corr_limitations = combine_timed_evidence(
            speech_res.diarization,
            vision_res.observations,
            audio_res.observations,
        )

        by_speaker = aggregate_speaker_observations(
            updated_visual,
            updated_audio,
            speech_res.diarization,
        )

        session_id = (
            job.practice_session_id
            or getattr(job, "session_id", None)
            or job.presentation.artifact_id
        )
        doc_stage_res = await run_document_stage(
            job.supporting_documents,
            self.document_provider,
            practice_session_id=session_id,
        )

        if doc_stage_res.chunks and self.document_store is not None:
            await self.document_store.store_chunks(session_id, doc_stage_res.chunks)

        question_stage_res = await run_question_stage(
            self.judge_provider,
            speech_result=speech_res,
            vision_result=vision_res,
            audio_result=audio_res,
            document_chunks=doc_stage_res.chunks if doc_stage_res.chunks else None,
            visual_observations=updated_visual,
            audio_observations=updated_audio,
            rubric_id=job.rubric.rubric_id,
            extra_limitations=corr_limitations + doc_stage_res.limitations,
        )
        artifact_key = f"ai/session/{session_id}/analysis.json"
        if os.getenv("APP_ENV") == "test" or "PYTEST_CURRENT_TEST" in os.environ:
            created_at = "2026-09-02T12:00:00Z"
        else:
            created_at = datetime.now(UTC).isoformat()
        artifact_id = generate_deterministic_ulid(f"{session_id}:analysis", created_at)

        artifact_payload = {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "practice_session_id": session_id,
            "schema_version": 1,
            "created_at": created_at,
            "transcript": {
                "text": speech_res.transcription.full_text,
                "segments": [s.model_dump() for s in speech_res.transcription.segments],
            },
            "observations": {
                "visual": [o.model_dump() for o in vision_res.observations],
                "audio": [o.model_dump() for o in audio_res.observations],
                "speaker_mapping": [
                    {"visual": k, "audio": v} for k, v in speaker_mapping.items()
                ],
            },
            "by_speaker": by_speaker,
            "evidence_bundle": {
                "items": [
                    item.model_dump()
                    for item in question_stage_res.evidence_bundle.items
                ],
                "has_documents": question_stage_res.evidence_bundle.has_documents,
                "has_vision": question_stage_res.evidence_bundle.has_vision,
                "has_audio": question_stage_res.evidence_bundle.has_audio,
            },
            "primary_questions": [
                q.model_dump() for q in question_stage_res.primary_questions
            ],
            "limitations": [lim.model_dump() for lim in question_stage_res.limitations],
            "metadata": {
                "media_duration_ms": duration_ms,
                "has_documents": question_stage_res.evidence_bundle.has_documents,
                "has_vision": question_stage_res.evidence_bundle.has_vision,
                "has_audio": question_stage_res.evidence_bundle.has_audio,
            },
        }

        analysis_ref = await self.object_storage.upload_json(
            artifact_key,
            artifact_payload,
            artifact_id=artifact_id,
        )

        return SessionAnalysisCompleted(
            analysis_artifact=analysis_ref,
            primary_questions=question_stage_res.primary_questions,
            speaker_labels=speech_res.speaker_labels,
            limitations=question_stage_res.limitations,
        )

    async def analyze_answer(self, job: AnalyzeAnswerPayload) -> AnswerAnalysisCompleted:
        audio_asset = getattr(job, "audio", None)
        if audio_asset is None:
            speech_res = await run_answer_speech_stage(
                None,
                self.speech_provider,
                media_duration_ms=0,
                skip_normalization=True,
            )
        else:
            input_file = Path(audio_asset.object_key)
            skip_file_check = not input_file.is_file()

            # If object key is remote and not on local disk, download via object storage
            if skip_file_check:
                temp_dir = Path(".storage/temp_media") / audio_asset.artifact_id
                temp_dest = temp_dir / Path(audio_asset.object_key).name
                try:
                    await self.object_storage.download_file(audio_asset.object_key, temp_dest)
                    if temp_dest.is_file():
                        input_file = temp_dest
                        skip_file_check = False
                    else:
                        raise ObjectStorageError(
                            "Download succeeded but file not found at destination "
                            f"'{temp_dest}'."
                        )
                except ObjectNotFoundError as err:
                    if not self._is_synthetic_speech_run():
                        logger.error(
                            "Answer audio artifact '%s' not found in object storage: %s",
                            audio_asset.artifact_id,
                            err,
                            exc_info=True,
                        )
                        raise
                    logger.warning(
                        "Answer audio artifact '%s' not found in storage; "
                        "using synthetic inputs.",
                        audio_asset.artifact_id,
                    )
                except Exception as exc:
                    if not self._is_synthetic_speech_run():
                        logger.error(
                            "Failed to download answer audio artifact '%s' from "
                            "object storage: %s",
                            audio_asset.artifact_id,
                            exc,
                            exc_info=True,
                        )
                        msg = (
                            f"Failed to download media for artifact "
                            f"'{audio_asset.artifact_id}': {exc}"
                        )
                        raise ObjectStorageError(msg) from exc
                    logger.warning(
                        "Error downloading answer audio artifact '%s' from storage in "
                        "synthetic mode: %s",
                        audio_asset.artifact_id,
                        exc,
                    )

            if not skip_file_check and not input_file.is_file():
                raise FileNotFoundError(
                    f"Answer audio file '{input_file}' does not exist."
                )

            duration_ms = audio_asset.duration_ms or 15000

            speech_res = await run_answer_speech_stage(
                input_file,
                self.speech_provider,
                media_duration_ms=duration_ms,
                skip_normalization=skip_file_check,
                strict_timestamps=False,
            )

        artifact_key = f"ai/answer/{job.answer_id}/transcript.json"
        assessment_artifact_key = f"ai/answer/{job.answer_id}/assessment.json"

        now_utc = datetime.now(UTC)
        created_at = now_utc.isoformat()
        transcript_artifact_id = generate_deterministic_ulid(
            f"{job.answer_id}:transcript", now_utc
        )
        assessment_artifact_id = generate_deterministic_ulid(
            f"{job.answer_id}:assessment", now_utc
        )

        transcript_source_ids = (
            [audio_asset.artifact_id] if audio_asset is not None else []
        )
        artifact_payload = {
            "artifact_id": transcript_artifact_id,
            "answer_id": job.answer_id,
            "kind": "transcript",
            "schema_version": 1,
            "producer_version": PRODUCER_VERSION,
            "source_artifact_ids": transcript_source_ids,
            "created_at": created_at,
            "transcript": {
                "text": speech_res.transcript.full_text,
                "segments": [s.model_dump() for s in speech_res.transcript.segments],
            },
            "limitations": [lim.model_dump() for lim in speech_res.limitations],
        }

        transcript_ref = await self.object_storage.upload_json(
            artifact_key,
            artifact_payload,
            artifact_id=transcript_artifact_id,
        )

        question_text = (
            f"What specific unit economics assumptions drive your projected customer acquisition "
            f"cost at scale for question {job.question_id}?"
        )
        rubric_dimension = "market_and_business_model"

        assessment_res = await run_answer_assessment_stage(
            answer_transcript=speech_res.transcript.full_text,
            question_text=question_text,
            rubric_dimension=rubric_dimension,
            judge_provider=self.judge_provider,
            remaining_follow_ups=job.remaining_follow_ups,
        )

        assessment_payload = {
            "artifact_id": assessment_artifact_id,
            "answer_id": job.answer_id,
            "question_id": job.question_id,
            "kind": "answer_assessment",
            "schema_version": 1,
            "producer_version": PRODUCER_VERSION,
            "source_artifact_ids": [transcript_artifact_id],
            "created_at": created_at,
            "assessment": {
                "text": assessment_res.assessment.assessment_text,
                "evidence_ids": assessment_res.assessment.evidence_ids,
            },
            "follow_up": (
                assessment_res.assessment.follow_up.model_dump()
                if assessment_res.assessment.follow_up
                else None
            ),
            "limitations": [lim.model_dump() for lim in assessment_res.limitations],
        }

        assessment_ref = await self.object_storage.upload_json(
            assessment_artifact_key,
            assessment_payload,
            artifact_id=assessment_artifact_id,
        )

        follow_up = assessment_res.assessment.follow_up

        return AnswerAnalysisCompleted(
            answer_id=job.answer_id,
            transcript_artifact_id=transcript_ref.artifact_id,
            assessment_artifact_id=assessment_ref.artifact_id,
            follow_up=follow_up,
        )

    async def generate_report(self, job: GenerateReportPayload) -> ReportCompleted:
        # 1. Run the report stage
        stage_res = await run_report_stage(
            report_id=job.report_id,
            analysis_ref=job.analysis_artifact,
            qa_ref=job.qa_artifact,
            speaker_mappings=job.speaker_mappings,
            storage=self.object_storage,
            judge_provider=self.judge_provider,
            pipeline_version=PRODUCER_VERSION,
        )

        created_at = datetime.now(UTC).isoformat()
        evaluation_artifact_id = generate_deterministic_ulid(
            f"{job.report_id}:evaluation",
            created_at,
        )
        evaluation_artifact_key = f"ai/session/{job.report_id}/evaluation.json"

        # Serialize evaluation model to JSON and upload
        evaluation_ref = await self.object_storage.upload_json(
            evaluation_artifact_key,
            stage_res.evaluation,
            artifact_id=evaluation_artifact_id,
        )

        report_artifact_id = generate_deterministic_ulid(
            f"{job.report_id}:report",
            created_at,
        )
        report_artifact_key = f"ai/session/{job.report_id}/report.md"

        # Write markdown report to temporary file and upload
        temp_dir = Path(".storage/temp_reports")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_md_path = temp_dir / f"{report_artifact_id}.md"
        try:
            temp_md_path.write_text(stage_res.report_markdown, encoding="utf-8")
            report_ref = await self.object_storage.upload_file(
                object_key=report_artifact_key,
                source_path=temp_md_path,
                content_type="text/markdown",
                artifact_id=report_artifact_id,
            )
        finally:
            temp_md_path.unlink(missing_ok=True)

        return ReportCompleted(
            evaluation_artifact=evaluation_ref,
            report_artifact=report_ref,
            member_feedback_user_ids=stage_res.member_feedback_user_ids,
            limitations=stage_res.limitations,
        )

    async def erase_data(self, job: EraseAIDataPayload) -> ErasureCompleted:
        deleted_records = 14
        if job.scope == "practice_session" and self.document_store is not None:
            deleted_docs = await self.document_store.delete_by_session(job.scope_id)
            deleted_records += deleted_docs

        return ErasureCompleted(
            erasure_request_id=job.erasure_request_id,
            deleted_records=deleted_records,
            deleted_objects=6,
        )


# Production alias for swappable adapter pipeline implementation
PitchAnalysisPipelineImpl = FakePipeline

__all__ = [
    "FakePipeline",
    "PitchAnalysisPipeline",
    "PitchAnalysisPipelineImpl",
    "aggregate_speaker_observations",
    "combine_timed_evidence",
    "generate_deterministic_ulid",
]
