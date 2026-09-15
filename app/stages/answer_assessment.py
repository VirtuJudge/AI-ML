"""Answer assessment stage for VirtuJudge AI-ML.

Evaluates single-speaker Q&A answers against the question and rubric dimension
via LLM reasoning, generating grounded assessments and optional follow-up questions.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import Limitation
from app.providers.base import AnswerAssessment, JudgeModelProvider
from app.prompts.judges import JUDGE_DIRECTED_INSULT_REGEX

logger = logging.getLogger(__name__)


class AnswerAssessmentResult(BaseModel):
    """Result of answer assessment stage."""

    assessment: AnswerAssessment = Field(
        default_factory=lambda: AnswerAssessment(
            assessment_text="",
            score=0.0,
            evidence_ids=[],
            follow_up=None,
        )
    )
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def run_answer_assessment_stage(
    answer_transcript: str,
    question_text: str,
    rubric_dimension: str,
    judge_provider: JudgeModelProvider,
    *,
    remaining_follow_ups: int = 0,
) -> AnswerAssessmentResult:
    """Run answer assessment stage: evaluate answer transcript against rubric via LLM.

    Args:
        answer_transcript: Spoken answer transcript text.
        question_text: The question text that was answered.
        rubric_dimension: The primary rubric dimension being evaluated.
        judge_provider: LLM judge model provider.
        remaining_follow_ups: Remaining follow-up quota.

    Returns:
        AnswerAssessmentResult with assessment, limitations, and metadata.
    """
    if not answer_transcript or not answer_transcript.strip():
        return AnswerAssessmentResult(
            assessment=AnswerAssessment(
                assessment_text="",
                score=0.0,
                evidence_ids=[],
                follow_up=None,
            ),
            limitations=[],
            metadata={
                "stage": "answer_assessment",
                "skipped": True,
                "rubric_dimension": rubric_dimension,
            },
        )

    if JUDGE_DIRECTED_INSULT_REGEX.search(answer_transcript):
        return AnswerAssessmentResult(
            assessment=AnswerAssessment(
                assessment_text=(
                    "The response contains direct abusive language aimed at the judge or "
                    "the question, so it receives 0/100."
                ),
                score=0.0,
                evidence_ids=[],
                follow_up=None,
            ),
            limitations=[],
            metadata={
                "stage": "answer_assessment",
                "rubric_dimension": rubric_dimension,
                "disqualified_for_judge_directed_abuse": True,
            },
        )

    assessment = await judge_provider.assess_answer(
        answer_transcript=answer_transcript.strip(),
        question_text=question_text,
        rubric_dimension=rubric_dimension,
        remaining_follow_ups=remaining_follow_ups,
    )

    metadata: dict[str, Any] = {
        "stage": "answer_assessment",
        "judge_provider": judge_provider.__class__.__name__,
        "rubric_dimension": rubric_dimension,
        "has_follow_up": assessment.follow_up is not None,
    }

    return AnswerAssessmentResult(
        assessment=assessment,
        limitations=[],
        metadata=metadata,
    )


__all__ = [
    "AnswerAssessmentResult",
    "run_answer_assessment_stage",
]
