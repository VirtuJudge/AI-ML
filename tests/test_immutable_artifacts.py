"""Tests for immutable artifact object keys, interleaved attempts, and idempotent retries (#64)."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.backend_client import FakeBackendClient, JobConflictError
from app.contracts import (
    AnalyzeAnswerPayload,
    AnalyzeSessionPayload,
    AnswerAnalysisCompleted,
    ArtifactRef,
    AssetInput,
    AudioAssetInput,
    GenerateReportPayload,
    JobType,
    QueueMessage,
    ReportCompleted,
    RubricRef,
    SessionAnalysisCompleted,
    SpeakerMapping,
    UpdateStatus,
)
from app.pipeline import FakePipeline
from app.providers.base import AnswerAssessment
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeSpeechProvider
from app.providers.types import TranscriptionResult, TranscriptionSegment
from app.stages.common import StageTransientError
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job

FAKE_SHA256 = "sha256:" + "a" * 64


class _ConfigurableSpeechProvider(FakeSpeechProvider):
    """Deterministic speech provider returning configurable transcript text and counting calls."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.calls = 0

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=5000,
                    text=self.text,
                    confidence=0.98,
                )
            ],
            full_text=self.text,
            language="en",
        )


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _session_payload(session_id: str) -> AnalyzeSessionPayload:
    return AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JIMMUTABLEPRES0000000001",
            object_key="uploads/presentation.mp4",
            checksum=FAKE_SHA256,
            media_type="video/mp4",
            duration_ms=15000,
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
        practice_session_id=session_id,
    )


@pytest.mark.asyncio
async def test_recomputed_session_analysis_preserves_earlier_checksummed_bytes(
    tmp_path: Path,
) -> None:
    """A03: Recomputing a session after checkpoint removal preserves earlier ArtifactRef bytes."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JIMMUTABLESESS0000000001"
    payload = _session_payload(session_id)

    pipeline_v1 = FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("Attempt one original pitch transcript."),
        object_storage=storage,
    )
    res1 = await pipeline_v1.analyze_session(payload, job_id="job-sess-1", attempt=1)
    ref1 = res1.analysis_artifact
    bytes1 = (storage.base_dir / ref1.object_key).read_bytes()
    assert _sha256_bytes(bytes1) == ref1.checksum
    assert "attempts/1/jobs/job-sess-1/" in ref1.object_key
    assert ref1.artifact_id in ref1.object_key

    # Force recomputation by removing checkpoints and changing speech output.
    await storage.delete_prefix(f"ai/session/{session_id}/checkpoints/")

    pipeline_v2 = FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("Attempt two revised pitch transcript."),
        object_storage=storage,
    )
    res2 = await pipeline_v2.analyze_session(payload, job_id="job-sess-2", attempt=2)
    ref2 = res2.analysis_artifact
    bytes2 = (storage.base_dir / ref2.object_key).read_bytes()
    assert _sha256_bytes(bytes2) == ref2.checksum

    assert ref1.object_key != ref2.object_key
    assert ref1.artifact_id != ref2.artifact_id
    assert ref1.checksum != ref2.checksum

    # Earlier ArtifactRef still resolves to its exact original checksummed bytes.
    assert (storage.base_dir / ref1.object_key).read_bytes() == bytes1
    assert _sha256_bytes((storage.base_dir / ref1.object_key).read_bytes()) == ref1.checksum
    stored1 = await storage.read_json(ref1.object_key)
    stored2 = await storage.read_json(ref2.object_key)
    assert stored1["transcript"]["text"] == "Attempt one original pitch transcript."
    assert stored2["transcript"]["text"] == "Attempt two revised pitch transcript."

    # Direct calls without job_id also never overwrite each other.
    res_direct = await pipeline_v2.analyze_session(payload)
    assert res_direct.analysis_artifact.object_key not in {ref1.object_key, ref2.object_key}
    assert (storage.base_dir / ref1.object_key).read_bytes() == bytes1
    assert (storage.base_dir / ref2.object_key).read_bytes() == bytes2


@pytest.mark.asyncio
async def test_recomputed_answer_and_report_artifacts_preserve_earlier_bytes(
    tmp_path: Path,
) -> None:
    """Answer and report artifacts across attempts keep distinct keys and exact original bytes."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JIMMUTABLESESS0000000002"
    answer_id = "01JIMMUTABLEANS00000000001"

    class _ScoreJudge(FakeJudgeModelProvider):
        def __init__(self, score: float, summary: str) -> None:
            super().__init__()
            self._score = score
            self._summary = summary

        async def assess_answer(self, *args: Any, **kwargs: Any) -> AnswerAssessment:
            return AnswerAssessment(
                assessment_text=self._summary,
                score=self._score,
                evidence_ids=["ev_speech_001"],
                follow_up=None,
            )

    p1 = FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("Answer attempt 1"),
        judge_provider=_ScoreJudge(0.42, "Attempt 1 answer assessment"),
        object_storage=storage,
    )
    p2 = FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("Answer attempt 2"),
        judge_provider=_ScoreJudge(0.91, "Attempt 2 answer assessment"),
        object_storage=storage,
    )

    ans_payload = AnalyzeAnswerPayload(
        qa_round_id="01JROUND000000000000000001",
        question_id="q_01",
        answer_id=answer_id,
        answered_by="user_01",
        practice_session_id=session_id,
        audio=AudioAssetInput(
            artifact_id="01JAUDIO000000000000000001",
            object_key="answers/a.wav",
            checksum=FAKE_SHA256,
            media_type="audio/wav",
            duration_ms=20000,
        ),
        question_text="What is your moat?",
        rubric_dimension="technical_feasibility",
        question_evidence_ids=["ev_speech_001"],
        remaining_follow_ups=0,
    )

    ans1 = await p1.analyze_answer(ans_payload, job_id="job-ans-1", attempt=1)
    ans2 = await p2.analyze_answer(ans_payload, job_id="job-ans-2", attempt=2)

    assert ans1.transcript_artifact_id != ans2.transcript_artifact_id
    assert ans1.assessment_artifact_id != ans2.assessment_artifact_id

    ans1_transcript_key = (
        f"ai/session/{session_id}/answers/{answer_id}/attempts/1/jobs/job-ans-1/"
        f"{ans1.transcript_artifact_id}/transcript.json"
    )
    ans1_assessment_key = (
        f"ai/session/{session_id}/answers/{answer_id}/attempts/1/jobs/job-ans-1/"
        f"{ans1.assessment_artifact_id}/assessment.json"
    )
    ans2_assessment_key = (
        f"ai/session/{session_id}/answers/{answer_id}/attempts/2/jobs/job-ans-2/"
        f"{ans2.assessment_artifact_id}/assessment.json"
    )
    assert (await storage.read_json(ans1_transcript_key))["transcript"][
        "text"
    ] == "Answer attempt 1"
    assert (await storage.read_json(ans1_assessment_key))["assessment"]["score"] == 0.42
    assert (await storage.read_json(ans2_assessment_key))["assessment"]["score"] == 0.91

    # Generate report for attempt 1 and attempt 2, verifying report_loader selects the matching
    # assessment artifact by assessment_artifact_id and earlier report/evaluation bytes stay intact.
    sess1 = await p1.analyze_session(_session_payload(session_id), job_id="job-s-1", attempt=1)
    sess2 = await p2.analyze_session(_session_payload(session_id), job_id="job-s-2", attempt=2)

    async def _upload_qa(attempt: int, ans_res: AnswerAnalysisCompleted) -> ArtifactRef:
        return await storage.upload_json(
            f"ai/session/{session_id}/attempts/{attempt}/qa.json",
            {
                "artifact_id": f"01JQA0000000000000000000{attempt}",
                "practice_session_id": session_id,
                "qa_round_id": f"01JQAROUND000000000000000{attempt}",
                "analysis_attempt": attempt,
                "questions": [
                    {
                        "id": "q_01",
                        "text": "What is your moat?",
                        "rubric_dimension": "technical_feasibility",
                    }
                ],
                "answers": [
                    {
                        "id": answer_id,
                        "question_id": "q_01",
                        "status": "submitted",
                        "assessment_artifact_id": ans_res.assessment_artifact_id,
                        "transcript": f"Answer attempt {attempt}",
                    }
                ],
                "assessments": [],
            },
        )

    qa1_ref = await _upload_qa(1, ans1)
    qa2_ref = await _upload_qa(2, ans2)
    mappings = [SpeakerMapping(speaker_label="SPEAKER_00", user_id="user_01", display_name="Alice")]

    rep1 = await p1.generate_report(
        GenerateReportPayload(
            report_id="01JREPORT00000000000000001",
            analysis_artifact=sess1.analysis_artifact,
            qa_artifact=qa1_ref,
            speaker_mappings=mappings,
        ),
        job_id="job-rep-1",
        attempt=1,
        practice_session_id=session_id,
    )
    eval1_bytes = (storage.base_dir / rep1.evaluation_artifact.object_key).read_bytes()
    rep1_bytes = (storage.base_dir / rep1.report_artifact.object_key).read_bytes()

    rep2 = await p2.generate_report(
        GenerateReportPayload(
            report_id="01JREPORT00000000000000001",
            analysis_artifact=sess2.analysis_artifact,
            qa_artifact=qa2_ref,
            speaker_mappings=mappings,
        ),
        job_id="job-rep-2",
        attempt=2,
        practice_session_id=session_id,
    )

    assert rep1.evaluation_artifact.object_key != rep2.evaluation_artifact.object_key
    assert rep1.report_artifact.object_key != rep2.report_artifact.object_key
    assert (storage.base_dir / rep1.evaluation_artifact.object_key).read_bytes() == eval1_bytes
    assert (storage.base_dir / rep1.report_artifact.object_key).read_bytes() == rep1_bytes

    eval1_json = await storage.read_json(rep1.evaluation_artifact.object_key)
    eval2_json = await storage.read_json(rep2.evaluation_artifact.object_key)
    qa1_comp = next(c for c in eval1_json["components"] if c["dimension"] == "qa_quality")
    qa2_comp = next(c for c in eval2_json["components"] if c["dimension"] == "qa_quality")
    assert qa1_comp["normalized_score"] == 0.42
    assert qa2_comp["normalized_score"] == 0.91


@pytest.mark.asyncio
async def test_interleaved_attempts_stale_rejected_and_cancelled_do_not_clobber_accepted(
    tmp_path: Path,
) -> None:
    """Interleaved stale/cancelled attempt 1 cannot alter attempt 2's accepted object."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "01JINTERLEAVED000000000001"
    payload = _session_payload(session_id)

    gate_entered = asyncio.Event()
    release_attempt_1 = asyncio.Event()

    class _GatedSpeechProvider(_ConfigurableSpeechProvider):
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            gate_entered.set()
            await release_attempt_1.wait()
            return await super().transcribe(audio_path)

    stale_pipeline = FakePipeline(
        speech_provider=_GatedSpeechProvider("Stale attempt 1 transcript"),
        object_storage=storage,
    )
    accepted_pipeline = FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("Accepted attempt 2 transcript"),
        object_storage=storage,
    )

    msg1 = QueueMessage(
        schema_version=1,
        job_id="01JJOBSTALE000000000000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=session_id,
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="trc_stale_1",
        payload=payload.model_dump(mode="json"),
    )
    msg2 = QueueMessage(
        schema_version=1,
        job_id="01JJOBACCEPTED00000000000002",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=session_id,
        analysis_attempt=2,
        created_at=datetime.now(UTC),
        trace_id="trc_accepted_2",
        payload=payload.model_dump(mode="json"),
    )

    class _RejectCompletionBackendClient(FakeBackendClient):
        async def send_update(self, job_id: str, update: Any) -> dict[str, Any] | None:
            if update.status == UpdateStatus.COMPLETED:
                raise JobConflictError(f"Stale attempt rejected for job {job_id}")
            return await super().send_update(job_id, update)

    stale_backend = _RejectCompletionBackendClient()
    accepted_backend = FakeBackendClient()

    stale_task = asyncio.create_task(process_job(msg1, stale_pipeline, stale_backend))
    await gate_entered.wait()

    # Attempt 2 runs and completes while attempt 1 is still in flight.
    accepted_update = await process_job(msg2, accepted_pipeline, accepted_backend)
    assert accepted_update.status == UpdateStatus.COMPLETED
    assert isinstance(accepted_update.payload, SessionAnalysisCompleted)
    accepted_ref = accepted_update.payload.analysis_artifact
    accepted_bytes = (storage.base_dir / accepted_ref.object_key).read_bytes()
    assert _sha256_bytes(accepted_bytes) == accepted_ref.checksum

    # Let stale attempt 1 finish uploading after attempt 2, then get rejected at callback.
    release_attempt_1.set()
    with pytest.raises(JobConflictError):
        await stale_task

    # Accepted artifact must remain byte-for-byte identical with matching checksum.
    assert (storage.base_dir / accepted_ref.object_key).read_bytes() == accepted_bytes
    assert _sha256_bytes((storage.base_dir / accepted_ref.object_key).read_bytes()) == (
        accepted_ref.checksum
    )

    # A cancelled stale attempt also leaves no output for itself and never touches accepted output.
    cancel_msg = msg1.model_copy(
        update={"job_id": "01JJOBCANCELLED0000000000003", "trace_id": "trc_cancel_3"}
    )
    cancelled_update = await process_job(
        cancel_msg,
        accepted_pipeline,
        FakeBackendClient(cancel_requested=True),
    )
    assert cancelled_update.status == UpdateStatus.CANCELLED
    assert (
        await storage.list_objects(f"ai/session/{session_id}/attempts/1/jobs/{cancel_msg.job_id}/")
        == []
    )
    assert (storage.base_dir / accepted_ref.object_key).read_bytes() == accepted_bytes


@pytest.mark.asyncio
async def test_duplicate_delivery_returns_identical_refs_without_recomputation_or_overwrite(
    tmp_path: Path,
) -> None:
    """Duplicate delivery of the same job_id returns identical ArtifactRefs without rerunning."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    spy_speech = _ConfigurableSpeechProvider("Idempotent speech")
    pipeline = FakePipeline(speech_provider=spy_speech, object_storage=storage)
    backend = FakeBackendClient()
    session_id = "01JDUPDELIVERY000000000001"

    msg = QueueMessage(
        schema_version=1,
        job_id="01JDUPSESSJOB0000000000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=session_id,
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="trc_dup_1",
        payload=_session_payload(session_id).model_dump(mode="json"),
    )

    first = await process_job(msg, pipeline, backend)
    assert first.status == UpdateStatus.COMPLETED
    assert isinstance(first.payload, SessionAnalysisCompleted)
    assert spy_speech.calls == 1
    first_bytes = (storage.base_dir / first.payload.analysis_artifact.object_key).read_bytes()

    # Second delivery with the same job_id returns the exact same ArtifactRef cleanly.
    second = await process_job(msg, pipeline, backend)
    assert second.status == UpdateStatus.COMPLETED
    assert isinstance(second.payload, SessionAnalysisCompleted)
    assert spy_speech.calls == 1
    assert second.payload == first.payload
    assert (
        storage.base_dir / first.payload.analysis_artifact.object_key
    ).read_bytes() == first_bytes

    # Concurrent duplicate delivery of the same job converges on one ArtifactRef.
    concurrent_msg = msg.model_copy(update={"job_id": "01JDUPCONCURRENT0000000001"})
    r1, r2 = await asyncio.gather(
        pipeline.analyze_session(
            _session_payload(session_id),
            job_id=concurrent_msg.job_id,
            attempt=1,
        ),
        pipeline.analyze_session(
            _session_payload(session_id),
            job_id=concurrent_msg.job_id,
            attempt=1,
        ),
    )
    assert r1.analysis_artifact == r2.analysis_artifact
    assert (
        _sha256_bytes((storage.base_dir / r1.analysis_artifact.object_key).read_bytes())
        == r1.analysis_artifact.checksum
    )

    # Duplicate delivery for generate_report also returns identical ArtifactRefs.
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/attempts/1/qa.json",
        {
            "artifact_id": "01JDUPQA000000000000000001",
            "practice_session_id": session_id,
            "qa_round_id": "01JDUPROUND000000000000001",
            "analysis_attempt": 1,
            "questions": [{"id": "q1", "text": "Why?", "rubric_dimension": "business_reasoning"}],
            "answers": [{"question_id": "q1", "status": "submitted", "transcript": "Because."}],
            "assessments": [{"question_id": "q1", "score": 0.8, "evidence_ids": ["ev_speech_001"]}],
        },
    )
    rep_msg = QueueMessage(
        schema_version=1,
        job_id="01JDUPREPJOB00000000000001",
        job_type=JobType.GENERATE_REPORT,
        practice_session_id=session_id,
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="trc_dup_rep_1",
        payload=GenerateReportPayload(
            report_id="01JDUPREPORT00000000000001",
            analysis_artifact=r1.analysis_artifact,
            qa_artifact=qa_ref,
            speaker_mappings=[
                SpeakerMapping(speaker_label="SPEAKER_00", user_id="user_01", display_name="Alice")
            ],
        ).model_dump(mode="json"),
    )
    rep_first = await process_job(rep_msg, pipeline, backend)
    rep_second = await process_job(rep_msg, pipeline, backend)
    assert isinstance(rep_first.payload, ReportCompleted)
    assert isinstance(rep_second.payload, ReportCompleted)
    assert rep_second.payload == rep_first.payload


@pytest.mark.asyncio
async def test_retry_after_partial_failure_same_job_id_succeeds_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retry of a partially failed analyze_answer job succeeds without overwriting partials."""

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    storage = LocalDiskObjectStorage(base_dir=tmp_path)

    class _FlakyAssessmentJudge(FakeJudgeModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def assess_answer(self, *args: Any, **kwargs: Any) -> AnswerAssessment:
            self.calls += 1
            if self.calls <= 3:
                raise StageTransientError("Transient judge outage", stage="assessment")
            return await super().assess_answer(*args, **kwargs)

    judge = _FlakyAssessmentJudge()
    pipeline = FakePipeline(judge_provider=judge, object_storage=storage)
    session_id = "01JRETRYPARTIAL00000000001"
    answer_id = "01JRETRYANS000000000000001"
    job_id = "01JRETRYJOB000000000000001"

    ans_payload = AnalyzeAnswerPayload(
        qa_round_id="01JROUND000000000000000001",
        question_id="q_01",
        answer_id=answer_id,
        answered_by="user_01",
        practice_session_id=session_id,
        audio=AudioAssetInput(
            artifact_id="01JAUDIO000000000000000001",
            object_key="answers/a.wav",
            checksum=FAKE_SHA256,
            media_type="audio/wav",
            duration_ms=20000,
        ),
        remaining_follow_ups=0,
    )

    with pytest.raises(StageTransientError):
        await pipeline.analyze_answer(ans_payload, job_id=job_id, attempt=1)

    partial_keys = await storage.list_objects(
        f"ai/session/{session_id}/answers/{answer_id}/attempts/1/jobs/{job_id}/"
    )
    assert len(partial_keys) == 1
    assert partial_keys[0].endswith("/transcript.json")
    partial_bytes = (storage.base_dir / partial_keys[0]).read_bytes()

    # Retry of same job_id completes with fresh immutable artifacts; partial write stays intact.
    res = await pipeline.analyze_answer(ans_payload, job_id=job_id, attempt=1)
    assert isinstance(res, AnswerAnalysisCompleted)
    assert (storage.base_dir / partial_keys[0]).read_bytes() == partial_bytes

    # Duplicate delivery after completion returns the same completed ids without re-judging.
    res_dup = await pipeline.analyze_answer(ans_payload, job_id=job_id, attempt=1)
    assert res_dup == res
    assert judge.calls == 4


async def test_completed_job_redelivery_is_acknowledged_without_republishing(
    tmp_path: Path,
) -> None:
    """Backend reports the job completed: no provider calls and no further updates."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    spy_speech = _ConfigurableSpeechProvider("Already done")
    pipeline = FakePipeline(speech_provider=spy_speech, object_storage=storage)
    backend = FakeBackendClient()
    session_id = "01JCOMPLETEDREDELIVERY0001"
    msg = QueueMessage(
        schema_version=1,
        job_id="01JCOMPLETEDJOB00000000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id=session_id,
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="trc_completed_1",
        payload=_session_payload(session_id).model_dump(mode="json"),
    )

    first = await process_job(msg, pipeline, backend)
    assert first.status == UpdateStatus.COMPLETED
    updates_after_first = len(backend.updates)

    backend.job_status = "completed"
    second = await process_job(msg, pipeline, backend)
    assert second.status == UpdateStatus.COMPLETED
    assert second.payload == first.payload
    assert spy_speech.calls == 1
    assert len(backend.updates) == updates_after_first  # nothing re-published


async def test_output_ids_are_deterministic_per_job_and_attempt(tmp_path: Path) -> None:
    """The same job on fresh storage (no manifest) produces the same output IDs."""
    session_id = "01JDETERMINISTICIDS0000001"
    job_id = "01JDETERMINISTICJOB0000001"
    results = []
    for run in ("a", "b"):
        storage = LocalDiskObjectStorage(base_dir=tmp_path / run)
        pipeline = FakePipeline(
            speech_provider=_ConfigurableSpeechProvider("same"), object_storage=storage
        )
        results.append(
            await pipeline.analyze_session(_session_payload(session_id), job_id=job_id, attempt=1)
        )
    assert results[0].analysis_artifact.artifact_id == results[1].analysis_artifact.artifact_id
    assert [q.candidate_id for q in results[0].primary_questions] == [
        q.candidate_id for q in results[1].primary_questions
    ]

    other_attempt = await FakePipeline(
        speech_provider=_ConfigurableSpeechProvider("same"),
        object_storage=LocalDiskObjectStorage(base_dir=tmp_path / "c"),
    ).analyze_session(_session_payload(session_id), job_id=job_id, attempt=2)
    assert other_attempt.analysis_artifact.artifact_id != results[0].analysis_artifact.artifact_id


def test_fallback_question_ids_are_stable_for_the_same_content() -> None:
    from app.stages.evidence import EvidenceBundle
    from app.stages.judge_panel import JUDGE_PANEL, create_fallback_question

    bundle = EvidenceBundle(items=[])
    first = create_fallback_question(JUDGE_PANEL[0], bundle)
    again = create_fallback_question(JUDGE_PANEL[0], bundle)
    other = create_fallback_question(JUDGE_PANEL[1], bundle)
    assert first.candidate_id == again.candidate_id
    assert first.candidate_id != other.candidate_id
