"""Evidence and Q&A artifact loader for report generation (AI-07).

Fetches and unpacks session analysis artifacts (`analysis.json`) and Q&A round
artifacts (`qa.json`) from object storage. Provides deterministic synthetic defaults
when running in test environments with mock artifact references.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import ArtifactRef, Limitation
from app.storage.base import ObjectStorageProtocol

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


def build_synthetic_analysis_data(session_id: str = "01JTESTSESSION0000000000001") -> dict[str, Any]:
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
        try:
            analysis_data = await storage.read_json(analysis_ref.object_key)
        except Exception as exc:
            if os.getenv("AI_PROVIDER_MODE") == "live":
                logger.error(
                    "CRITICAL: Failed to load analysis artifact '%s' in live mode: %s",
                    analysis_ref.object_key,
                    exc,
                )
                raise
            logger.debug(
                "Could not load analysis artifact from storage (%s): %s. Using synthetic fallback.",
                analysis_ref.object_key,
                exc,
            )

    if analysis_data is None:
        analysis_data = build_synthetic_analysis_data(session_id=analysis_ref.artifact_id)

    qa_data: dict[str, Any] | None = None
    if storage is not None:
        try:
            qa_data = await storage.read_json(qa_ref.object_key)
        except Exception as exc:
            if os.getenv("AI_PROVIDER_MODE") == "live":
                logger.error(
                    "CRITICAL: Failed to load QA artifact '%s' in live mode: %s",
                    qa_ref.object_key,
                    exc,
                )
                raise
            logger.debug(
                "Could not load qa artifact from storage (%s): %s. Using synthetic fallback.",
                qa_ref.object_key,
                exc,
            )

    if qa_data is None:
        qa_data = build_synthetic_qa_data(session_id=analysis_data.get("session_id", "session_001"))

    # Extract session identifiers
    session_id = (
        analysis_data.get("session_id")
        or analysis_data.get("practice_session_id")
        or analysis_ref.artifact_id
    )
    practice_session_id = analysis_data.get("practice_session_id", session_id)

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
    speaker_labels = list(by_speaker.keys()) if by_speaker else analysis_data.get("speaker_labels", [])

    # Extract limitations
    limitations: list[Limitation] = []
    for lim in analysis_data.get("limitations", []):
        if isinstance(lim, dict):
            limitations.append(Limitation.model_validate(lim))
        elif isinstance(lim, Limitation):
            limitations.append(lim)

    for lim in qa_data.get("limitations", []):
        if isinstance(lim, dict):
            limitations.append(Limitation.model_validate(lim))
        elif isinstance(lim, Limitation):
            limitations.append(lim)

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
