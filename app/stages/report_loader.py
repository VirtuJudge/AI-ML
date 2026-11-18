"""Evidence and Q&A artifact loader for report generation (AI-07).

Fetches and unpacks session analysis artifacts (`analysis.json`) and Q&A round
artifacts (`qa.json`) from object storage. Provides deterministic synthetic defaults
when running in test environments with mock artifact references.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import ArtifactRef, Limitation
from app.stages.scoring import extract_assessment_score
from app.storage.base import (
    ObjectNotFoundError,
    ObjectStorageError,
    ObjectStorageProtocol,
    verify_file_checksum,
)

logger = logging.getLogger(__name__)


class ReportEvidenceBundle(BaseModel):
    """Consolidated evidence and Q&A context ready for rubric scoring and report generation."""

    session_id: str
    practice_session_id: str = ""
    transcript_full_text: str = ""
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)
    document_chunks: list[dict[str, Any]] = Field(default_factory=list)
    by_speaker: dict[str, Any] = Field(default_factory=dict)
    diarization_segments: list[dict[str, Any]] = Field(default_factory=list)
    speaker_labels: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    questions: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[dict[str, Any]] = Field(default_factory=list)
    assessments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_synthetic_analysis_data(
    session_id: str = "01JTESTSESSION0000000000001",
) -> dict[str, Any]:
    """Generate contract-compliant synthetic analysis artifact payload for testing."""
    return {
        "artifact_id": f"{session_id}:analysis",
        "session_id": session_id,
        "practice_session_id": session_id,
        "schema_version": 1,
        "created_at": "2026-09-02T12:00:00Z",
        "transcript": {
            "text": (
                "Welcome to our pitch for VirtuJudge, the automated AI evaluation platform. "
                "We solve founder preparation bottlenecks by providing instant, rubric-grounded feedback. "
                "Our unit economics and defensibility are proven by early customer trials."
            ),
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 4500,
                    "text": "Welcome to our pitch for VirtuJudge, the automated AI evaluation platform.",
                    "confidence": 0.98,
                },
                {
                    "start_ms": 4600,
                    "end_ms": 9200,
                    "text": "We solve founder preparation bottlenecks by providing instant, rubric-grounded feedback.",
                    "confidence": 0.95,
                },
                {
                    "start_ms": 9300,
                    "end_ms": 14000,
                    "text": "Our unit economics and defensibility are proven by early customer trials.",
                    "confidence": 0.97,
                },
            ],
        },
        "by_speaker": {
            "SPEAKER_00": {
                "speaking_time_ms": 9200,
                "intervals": [
                    {
                        "start_ms": 0,
                        "end_ms": 9200,
                        "formatted": "00:00 - 00:09",
                    }
                ],
                "windows": [
                    {
                        "start_ms": 0,
                        "end_ms": 10000,
                        "acoustic": {
                            "speaking_rate_wpm": 138.0,
                            "pitch_mean_hz": 145.2,
                            "pitch_std_hz": 12.3,
                            "pause_duration_ms": 250,
                            "pause_count": 1,
                            "filler_count": 0,
                        },
                        "visual": {
                            "gaze_direction": 1.0,
                            "head_pitch_degrees": -2.5,
                            "posture_openness": 0.85,
                            "shoulder_symmetry_ratio": 0.96,
                            "upper_body_movement_px": 3.2,
                        },
                    }
                ],
            },
            "SPEAKER_01": {
                "speaking_time_ms": 4700,
                "intervals": [
                    {
                        "start_ms": 9300,
                        "end_ms": 14000,
                        "formatted": "00:09 - 00:14",
                    }
                ],
                "windows": [
                    {
                        "start_ms": 10000,
                        "end_ms": 20000,
                        "acoustic": {
                            "speaking_rate_wpm": 135.0,
                            "pitch_mean_hz": 148.5,
                            "pitch_std_hz": 11.0,
                            "pause_duration_ms": 200,
                            "pause_count": 1,
                            "filler_count": 0,
                        },
                        "visual": {
                            "gaze_direction": 1.0,
                            "head_pitch_degrees": -1.2,
                            "posture_openness": 0.82,
                            "shoulder_symmetry_ratio": 0.95,
                            "upper_body_movement_px": 2.8,
                        },
                    }
                ],
            },
        },
        "evidence_bundle": {
            "items": [
                {
                    "evidence_id": "ev_speech_001",
                    "source": "speech",
                    "speaker_label": "SPEAKER_00",
                    "start_ms": 0,
                    "end_ms": 9200,
                    "title": "Problem Statement",
                    "text": "We solve founder preparation bottlenecks by providing instant, rubric-grounded feedback.",
                },
                {
                    "evidence_id": "ev_doc_slide_01",
                    "source": "documents",
                    "page_or_slide": 1,
                    "title": "Title Slide",
                    "text": "VirtuJudge: AI-Powered Startup Pitch Evaluation",
                },
            ],
            "has_documents": True,
            "has_vision": True,
            "has_audio": True,
        },
        "limitations": [],
        "metadata": {
            "media_duration_ms": 15000,
            "has_documents": True,
            "has_vision": True,
            "has_audio": True,
        },
    }


def build_synthetic_qa_data(session_id: str = "01JTESTSESSION0000000000001") -> dict[str, Any]:
    """Generate contract-compliant synthetic Q&A artifact payload for testing."""
    return {
        "artifact_id": f"{session_id}:qa",
        "qa_round_id": f"{session_id}:round_1",
        "practice_session_id": session_id,
        "questions": [
            {
                "id": "q_01",
                "kind": "primary",
                "position": 1,
                "text": "How does VirtuJudge defensibility hold up against general-purpose LLMs?",
                "rubric_dimension": "technical_feasibility",
                "evidence_ids": ["ev_speech_001"],
            },
            {
                "id": "q_02",
                "kind": "primary",
                "position": 2,
                "text": "What are your unit economics and customer acquisition channels?",
                "rubric_dimension": "business_and_problem_solution_reasoning",
                "evidence_ids": ["ev_speech_001"],
            },
            {
                "id": "q_03",
                "kind": "primary",
                "position": 3,
                "text": "How do you handle multi-presenter video and audio calibration?",
                "rubric_dimension": "pitch_content_and_evidence",
                "evidence_ids": ["ev_speech_001"],
            },
        ],
        "answers": [
            {
                "id": "ans_01",
                "question_id": "q_01",
                "answered_by": "user_01",
                "status": "submitted",
                "transcript": "We build deep domain fine-tuning and deterministic feature extractors.",
                "duration_ms": 25000,
            },
            {
                "id": "ans_02",
                "question_id": "q_02",
                "answered_by": "user_01",
                "status": "submitted",
                "transcript": "Our CAC is under $200 with 6-month payback through accelerator partnerships.",
                "duration_ms": 30000,
            },
            {
                "id": "ans_03",
                "question_id": "q_03",
                "answered_by": "user_02",
                "status": "submitted",
                "transcript": "We correlate facial MAR landmarks with diarized speaker turns deterministically.",
                "duration_ms": 28000,
            },
        ],
        "assessments": [
            {
                "question_id": "q_01",
                "score": 0.85,
                "assessment_text": "Strong technical explanation of moat and proprietary algorithms.",
                "evidence_ids": ["ev_speech_001"],
            },
            {
                "question_id": "q_02",
                "score": 0.80,
                "assessment_text": "Clear CAC and payback metrics backed by realistic acquisition assumptions.",
                "evidence_ids": ["ev_speech_001"],
            },
            {
                "question_id": "q_03",
                "score": 0.78,
                "assessment_text": "Sound explanation of acoustic and visual multimodal synchronization.",
                "evidence_ids": ["ev_speech_001"],
            },
        ],
        "limitations": [],
    }


def _append_unique_limitation(limitations: list[Limitation], lim: Limitation) -> None:
    """Append a Limitation unless an equivalent code/scope/dimension entry is already present."""
    key = (lim.code, lim.scope, tuple(lim.affected_dimensions))
    existing = {(item.code, item.scope, tuple(item.affected_dimensions)) for item in limitations}
    if key not in existing:
        limitations.append(lim)


def _candidate_assessment_keys(answer_id: str, session_ids: list[str]) -> list[str]:
    """Return ordered, deduplicated object keys for an answer's assessment artifact."""
    keys: list[str] = []
    for sid in session_ids:
        if sid:
            candidate = f"ai/session/{sid}/answers/{answer_id}/assessment.json"
            if candidate not in keys:
                keys.append(candidate)
    fallback = f"ai/answer/{answer_id}/assessment.json"
    if fallback not in keys:
        keys.append(fallback)
    return keys


def _verify_assessment_artifact(
    payload: Any,
    *,
    expected_artifact_id: str | None,
    expected_answer_id: str,
    expected_question_id: str,
    expected_session_id: str,
) -> tuple[bool, float | None, str | None, list[str], list[Limitation], str | None]:
    """Verify an assessment.json payload and extract its score, text, evidence, and limitations.

    Returns:
        (verified, score, assessment_text, evidence_ids, limitations, status)
    """
    if not isinstance(payload, dict):
        return False, None, None, [], [], None

    kind = payload.get("kind")
    if kind is not None and kind != "answer_assessment":
        return False, None, None, [], [], None

    if expected_artifact_id:
        artifact_id = payload.get("artifact_id")
        if str(artifact_id or "") != expected_artifact_id:
            return False, None, None, [], [], None

    payload_answer_id = payload.get("answer_id")
    if payload_answer_id is not None and str(payload_answer_id) != expected_answer_id:
        return False, None, None, [], [], None

    payload_question_id = payload.get("question_id")
    if (
        payload_question_id is not None
        and expected_question_id
        and str(payload_question_id) != expected_question_id
    ):
        return False, None, None, [], [], None

    payload_session_id = payload.get("practice_session_id")
    if (
        payload_session_id is not None
        and expected_session_id
        and str(payload_session_id) != expected_session_id
    ):
        return False, None, None, [], [], None

    inner = payload.get("assessment")
    if inner is not None and not isinstance(inner, dict):
        return False, None, None, [], [], None

    source_dict = inner if isinstance(inner, dict) else payload
    status_raw = source_dict.get("status") or payload.get("status")
    status = str(status_raw) if status_raw is not None else None

    raw_text = source_dict.get("text")
    if raw_text is None:
        raw_text = payload.get("assessment_text")
    text = str(raw_text) if isinstance(raw_text, str) else None

    raw_evidence_ids = source_dict.get("evidence_ids")
    if raw_evidence_ids is None:
        raw_evidence_ids = payload.get("evidence_ids")
    evidence_ids = (
        [str(eid) for eid in raw_evidence_ids if str(eid)]
        if isinstance(raw_evidence_ids, list)
        else []
    )

    artifact_limitations: list[Limitation] = []
    for raw_lim in payload.get("limitations", []):
        if isinstance(raw_lim, dict):
            artifact_limitations.append(Limitation.model_validate(raw_lim))
        elif isinstance(raw_lim, Limitation):
            artifact_limitations.append(raw_lim)

    if status == "not_evaluated":
        return True, None, text, evidence_ids, artifact_limitations, "not_evaluated"

    score = extract_assessment_score(payload)
    if score is None:
        return False, None, text, evidence_ids, artifact_limitations, None

    return True, score, text, evidence_ids, artifact_limitations, "scored"


async def _recover_missing_assessment_scores(
    *,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    normalized_assessments: list[dict[str, Any]],
    limitations: list[Limitation],
    storage: ObjectStorageProtocol | None,
    session_ids: list[str],
    expected_session_id: str,
) -> None:
    """Read and verify assessment.json artifacts when qa.json omits assessment scores."""
    answers_by_qid: dict[str, dict[str, Any]] = {}
    for raw_ans in answers:
        if not isinstance(raw_ans, dict):
            continue
        qid_raw = raw_ans.get("question_id") or raw_ans.get("candidate_id")
        if qid_raw:
            answers_by_qid[str(qid_raw)] = raw_ans

    assessments_by_qid: dict[str, dict[str, Any]] = {}
    for asmt in normalized_assessments:
        qid_raw = asmt.get("question_id") or asmt.get("candidate_id")
        if qid_raw:
            assessments_by_qid[str(qid_raw)] = asmt

    # Build ordered list of (question_id, question_dict, answer_dict) for all non-skipped answers
    ordered_targets: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen_qids: set[str] = set()
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid_raw = q.get("id") or q.get("question_id") or q.get("candidate_id")
        if not qid_raw:
            continue
        qid = str(qid_raw)
        seen_qids.add(qid)
        ans = answers_by_qid.get(qid)
        if ans and ans.get("status") != "skipped":
            ordered_targets.append((qid, q, ans))

    for qid, ans in answers_by_qid.items():
        if qid not in seen_qids and ans.get("status") != "skipped":
            ordered_targets.append((qid, {}, ans))

    for qid, q, ans in ordered_targets:
        existing_asm = assessments_by_qid.get(qid)
        if existing_asm is not None and extract_assessment_score(existing_asm) is not None:
            continue

        if existing_asm is not None and existing_asm.get("status") == "not_evaluated":
            _append_unique_limitation(
                limitations,
                Limitation(
                    code="qa_assessment_unavailable",
                    scope="qa_quality",
                    message="One or more answered Q&A responses were marked not_evaluated.",
                    affected_dimensions=["qa_quality"],
                ),
            )
            continue

        answer_id_raw = ans.get("id") or ans.get("answer_id") or q.get("answer_id")
        answer_id = str(answer_id_raw) if answer_id_raw else ""
        expected_artifact_id_raw = ans.get("assessment_artifact_id") or (
            existing_asm.get("assessment_artifact_id") if existing_asm else None
        )
        expected_artifact_id = str(expected_artifact_id_raw) if expected_artifact_id_raw else None

        recovered = False
        verification_failed = False

        if storage is not None and answer_id:
            for key in _candidate_assessment_keys(answer_id, session_ids):
                try:
                    artifact_payload = await storage.read_json(key)
                except ObjectNotFoundError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "Failed reading assessment artifact for answer '%s' (%s).",
                        answer_id,
                        type(exc).__name__,
                    )
                    break

                (
                    verified,
                    score,
                    text,
                    evidence_ids,
                    artifact_lims,
                    art_status,
                ) = _verify_assessment_artifact(
                    artifact_payload,
                    expected_artifact_id=expected_artifact_id,
                    expected_answer_id=answer_id,
                    expected_question_id=qid,
                    expected_session_id=expected_session_id,
                )
                for lim in artifact_lims:
                    _append_unique_limitation(limitations, lim)

                if not verified:
                    verification_failed = True
                    logger.warning(
                        "Assessment artifact verification failed for answer '%s'.",
                        answer_id,
                    )
                    break

                target_asm = existing_asm
                if target_asm is None:
                    target_asm = {"question_id": qid}
                    normalized_assessments.append(target_asm)
                    assessments_by_qid[qid] = target_asm

                target_asm["answer_id"] = answer_id
                if expected_artifact_id and not target_asm.get("assessment_artifact_id"):
                    target_asm["assessment_artifact_id"] = expected_artifact_id
                elif not target_asm.get("assessment_artifact_id") and isinstance(
                    artifact_payload, dict
                ):
                    art_id = artifact_payload.get("artifact_id")
                    if art_id:
                        target_asm["assessment_artifact_id"] = str(art_id)

                if not target_asm.get("assessment_text") and text:
                    target_asm["assessment_text"] = text
                if not target_asm.get("evidence_ids") and evidence_ids:
                    target_asm["evidence_ids"] = evidence_ids

                if art_status == "not_evaluated" or score is None:
                    target_asm["score"] = None
                    target_asm["status"] = "not_evaluated"
                else:
                    target_asm["score"] = score
                    target_asm["status"] = "scored"
                    recovered = True
                break

        if not recovered:
            target_asm = existing_asm
            if target_asm is None:
                target_asm = {"question_id": qid}
                normalized_assessments.append(target_asm)
                assessments_by_qid[qid] = target_asm
            target_asm["score"] = None
            target_asm["status"] = "not_evaluated"
            if answer_id and not target_asm.get("answer_id"):
                target_asm["answer_id"] = answer_id

            if verification_failed:
                _append_unique_limitation(
                    limitations,
                    Limitation(
                        code="qa_assessment_verification_failed",
                        scope="qa_quality",
                        message=(
                            "Answer assessment artifact failed verification for one or more "
                            "answered Q&A responses."
                        ),
                        affected_dimensions=["qa_quality"],
                    ),
                )
            else:
                _append_unique_limitation(
                    limitations,
                    Limitation(
                        code="qa_assessment_unavailable",
                        scope="qa_quality",
                        message=(
                            "Answer assessment score was unavailable for one or more "
                            "answered Q&A responses."
                        ),
                        affected_dimensions=["qa_quality"],
                    ),
                )


async def _load_verified_json_artifact(
    ref: ArtifactRef,
    storage: ObjectStorageProtocol,
    *,
    artifact_role: str,
) -> dict[str, Any] | None:
    """Download an artifact's raw bytes, verify its SHA-256 checksum, and parse JSON."""
    with tempfile.TemporaryDirectory(prefix="virtujudge_report_") as tmpdir:
        dest = Path(tmpdir) / f"{artifact_role}.json"
        try:
            await storage.download_file(ref.object_key, dest)
            if not dest.is_file():
                raise ObjectStorageError(
                    f"Download succeeded but file not found for artifact '{ref.artifact_id}'."
                )
        except Exception as exc:
            if os.getenv("AI_PROVIDER_MODE") == "live":
                logger.error(
                    "CRITICAL: Failed to load %s artifact '%s' in live mode (%s).",
                    artifact_role,
                    ref.artifact_id,
                    type(exc).__name__,
                )
                raise
            logger.debug(
                "Could not load %s artifact '%s' from storage (%s). Using synthetic fallback.",
                artifact_role,
                ref.artifact_id,
                type(exc).__name__,
            )
            return None

        verify_file_checksum(
            dest,
            ref.checksum,
            stage="ingestion",
            artifact_id=ref.artifact_id,
        )
        data: dict[str, Any] = json.loads(dest.read_text(encoding="utf-8"))
        return data


async def load_report_evidence(
    analysis_ref: ArtifactRef,
    qa_ref: ArtifactRef,
    storage: ObjectStorageProtocol | None = None,
) -> ReportEvidenceBundle:
    """Fetch, parse, and validate session analysis and Q&A evidence for report generation.

    Args:
        analysis_ref: ArtifactRef for analysis.json.
        qa_ref: ArtifactRef for qa.json.
        storage: ObjectStorage protocol implementation (or None for synthetic).

    Returns:
        ReportEvidenceBundle with all evidence components resolved and normalized.
    """
    analysis_data: dict[str, Any] | None = None
    if storage is not None:
        analysis_data = await _load_verified_json_artifact(
            analysis_ref,
            storage,
            artifact_role="analysis",
        )

    if analysis_data is None:
        analysis_data = build_synthetic_analysis_data(session_id=analysis_ref.artifact_id)

    qa_data: dict[str, Any] | None = None
    if storage is not None:
        qa_data = await _load_verified_json_artifact(
            qa_ref,
            storage,
            artifact_role="qa",
        )

    if qa_data is None:
        qa_data = build_synthetic_qa_data(session_id=analysis_data.get("session_id", "session_001"))

    # Extract session identifiers
    analysis_key_parts = analysis_ref.object_key.split("/")
    analysis_session_from_key = (
        analysis_key_parts[2]
        if analysis_ref.object_key.startswith("ai/session/") and len(analysis_key_parts) >= 4
        else ""
    )
    qa_session_id = str(qa_data.get("practice_session_id") or "")
    session_id = (
        analysis_data.get("session_id")
        or analysis_data.get("practice_session_id")
        or qa_session_id
        or analysis_session_from_key
        or analysis_ref.artifact_id
    )
    practice_session_id = analysis_data.get("practice_session_id") or qa_session_id or session_id

    # Extract transcript
    transcript_info = analysis_data.get("transcript", {})
    transcript_text = transcript_info.get("text", "")
    transcript_segments = transcript_info.get("segments", [])

    # Extract document chunks
    document_chunks: list[dict[str, Any]] = []
    if "document_chunks" in analysis_data:
        document_chunks = analysis_data["document_chunks"]
    elif "evidence_bundle" in analysis_data:
        ev_items = analysis_data["evidence_bundle"].get("items", [])
        document_chunks = [item for item in ev_items if item.get("source") == "documents"]

    # Extract by_speaker profile
    by_speaker = analysis_data.get("by_speaker", {})

    # Extract diarization segments
    diarization_segments = analysis_data.get("diarization_segments", [])
    if not diarization_segments and "observations" in analysis_data:
        diarization_segments = analysis_data["observations"].get("diarization", [])

    # Extract speaker labels
    speaker_labels = (
        list(by_speaker.keys()) if by_speaker else analysis_data.get("speaker_labels", [])
    )

    # Extract limitations
    limitations: list[Limitation] = []
    for lim in analysis_data.get("limitations", []):
        if isinstance(lim, dict):
            _append_unique_limitation(limitations, Limitation.model_validate(lim))
        elif isinstance(lim, Limitation):
            _append_unique_limitation(limitations, lim)

    for lim in qa_data.get("limitations", []):
        if isinstance(lim, dict):
            _append_unique_limitation(limitations, Limitation.model_validate(lim))
        elif isinstance(lim, Limitation):
            _append_unique_limitation(limitations, lim)

    # Extract Q&A elements
    questions = qa_data.get("questions", [])
    if not questions and "qa_round" in qa_data:
        questions = qa_data["qa_round"].get("questions", [])

    answers = qa_data.get("answers", [])
    if not answers and "qa_round" in qa_data:
        answers = qa_data["qa_round"].get("answers", [])

    assessments = qa_data.get("assessments", [])
    if not isinstance(assessments, list):
        assessments = []
    normalized_assessments: list[dict[str, Any]] = []
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        normalized = dict(assessment)
        nested = assessment.get("assessment")
        if isinstance(nested, dict):
            if normalized.get("assessment_text") is None:
                normalized["assessment_text"] = nested.get("text")
            if normalized.get("score") is None:
                normalized["score"] = nested.get("score")
            if not normalized.get("evidence_ids"):
                normalized["evidence_ids"] = nested.get("evidence_ids", [])
        normalized_assessments.append(normalized)

    candidate_session_ids = [
        sid
        for sid in (qa_session_id, practice_session_id, session_id, analysis_session_from_key)
        if sid
    ]
    await _recover_missing_assessment_scores(
        questions=questions,
        answers=answers,
        normalized_assessments=normalized_assessments,
        limitations=limitations,
        storage=storage,
        session_ids=candidate_session_ids,
        expected_session_id=qa_session_id,
    )

    return ReportEvidenceBundle(
        session_id=session_id,
        practice_session_id=practice_session_id,
        transcript_full_text=transcript_text,
        transcript_segments=transcript_segments,
        document_chunks=document_chunks,
        by_speaker=by_speaker,
        diarization_segments=diarization_segments,
        speaker_labels=speaker_labels,
        limitations=limitations,
        questions=questions,
        answers=answers,
        assessments=normalized_assessments,
        metadata=analysis_data.get("metadata", {}),
    )


__all__ = [
    "ReportEvidenceBundle",
    "build_synthetic_analysis_data",
    "build_synthetic_qa_data",
    "load_report_evidence",
]
