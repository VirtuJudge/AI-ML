"""Question generation stage for VirtuJudge AI-ML pipeline.

Coordinates evidence aggregation and the multi-judge evaluation panel to generate
three grounded, objective primary questions citing presentation speech and slides.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.contracts import Limitation, PrimaryQuestion
from app.providers.base import JudgeModelProvider
from app.providers.types import (
    AudioObservation,
    DocumentChunk,
    VisualObservation,
)
from app.stages.audio import AudioStageResult
from app.stages.evidence import EvidenceBundle, build_evidence_bundle
from app.stages.speech import SpeechStageResult
from app.stages.vision import VisionStageResult


class QuestionStageResult(BaseModel):
    """Result of the question generation stage."""

    primary_questions: list[PrimaryQuestion]
    evidence_bundle: EvidenceBundle
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def run_question_stage(
    judge_provider: JudgeModelProvider,
    *,
    speech_result: SpeechStageResult | None = None,
    transcript: str | None = None,
    speaker_labels: list[str] | None = None,
    vision_result: VisionStageResult | None = None,
    audio_result: AudioStageResult | None = None,
    document_chunks: list[DocumentChunk] | None = None,
    visual_observations: list[VisualObservation] | None = None,
    audio_observations: list[AudioObservation] | None = None,
    rubric_id: str = "startup_pitch",
    extra_limitations: list[Limitation] | None = None,
) -> QuestionStageResult:
    """Run the question generation stage using a multi-judge provider.

    Aggregates multi-modal evidence into a structured EvidenceBundle, passes
    spoken transcript and document chunks to the judge panel, and returns
    exactly 3 grounded primary questions alongside the bundle and limitations.
    """
    # 1. Build unified evidence bundle
    bundle = build_evidence_bundle(
        speech_result=speech_result,
        transcript=transcript,
        speaker_labels=speaker_labels,
        vision_result=vision_result,
        audio_result=audio_result,
        document_chunks=document_chunks,
        visual_observations=visual_observations,
        audio_observations=audio_observations,
        rubric_id=rubric_id,
        extra_limitations=extra_limitations,
    )

    # 2. Query judge provider for grounded questions
    transcript_text = bundle.transcript_full_text
    questions = await judge_provider.generate_questions(
        transcript=transcript_text,
        rubric_id=rubric_id,
        document_chunks=document_chunks,
        evidence_bundle=bundle,
    )

    # 3. Strictly validate panel questions against bundle grounding & rubric dimensions
    from app.stages.judge_panel import validate_panel_questions

    validated_questions, panel_limitations = validate_panel_questions(
        list(questions),
        bundle,
    )
    all_limitations = list(bundle.limitations) + panel_limitations

    # 4. Compile metadata
    metadata: dict[str, Any] = {
        "question_count": len(validated_questions),
        "rubric_id": rubric_id,
        "evidence_item_count": len(bundle.items),
        "has_documents": bundle.has_documents,
        "has_vision": bundle.has_vision,
        "has_audio": bundle.has_audio,
    }

    return QuestionStageResult(
        primary_questions=validated_questions,
        evidence_bundle=bundle,
        limitations=all_limitations,
        metadata=metadata,
    )


__all__ = ["QuestionStageResult", "run_question_stage"]
