"""Failed answer assessments are not_evaluated, never an ordinary zero (issue #55)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.providers.base import AnswerAssessment
from app.providers.groq_judge import GroqJudgeModelProvider
from app.providers.groq_pool import GroqKeyPool, GroqPoolError
from app.stages.answer_assessment import run_answer_assessment_stage
from app.stages.scoring import evaluate_qa_scores

VALID = '{"assessment_text": "Clear unit economics.", "score": 0.7, "evidence_ids": []}'
VALID_ZERO = (
    '{"assessment_text": "Did not address the question.", "score": 0.0, "evidence_ids": []}'
)
MALFORMED = '{"assessment_text": "no score here"}'


def _provider(*responses: object) -> GroqJudgeModelProvider:
    pool = AsyncMock(spec=GroqKeyPool)
    pool.chat.side_effect = list(responses)
    return GroqJudgeModelProvider(key_pool=pool)


async def _assess(provider: GroqJudgeModelProvider) -> AnswerAssessment:
    return await provider.assess_answer(
        answer_transcript="We charge per seat and break even at 400 seats.",
        question_text="How do you make money?",
        rubric_dimension="market_and_business_model",
    )


async def test_primary_success_is_scored() -> None:
    result = await _assess(_provider(VALID))
    assert (result.status, result.score) == ("scored", 0.7)
    assert result.metadata["assessment_source"] == "primary"
    assert result.limitations == []


async def test_fallback_model_success_is_scored() -> None:
    result = await _assess(_provider(GroqPoolError("HTTP 503"), VALID))
    assert (result.status, result.score) == ("scored", 0.7)
    assert result.metadata["assessment_source"] == "fallback_model"


async def test_total_failure_is_not_evaluated() -> None:
    result = await _assess(_provider(GroqPoolError("HTTP 503"), GroqPoolError("timeout")))
    assert result.status == "not_evaluated"
    assert result.score is None
    assert result.metadata["assessment_source"] == "unavailable"
    assert [lim.code for lim in result.limitations] == ["answer_assessment_unavailable"]


async def test_malformed_output_twice_is_not_evaluated() -> None:
    result = await _assess(_provider(MALFORMED, MALFORMED))
    assert result.status == "not_evaluated" and result.score is None


async def test_valid_zero_stays_a_scored_zero() -> None:
    result = await _assess(_provider(VALID_ZERO))
    assert (result.status, result.score) == ("scored", 0.0)


async def test_stage_carries_status_limitation_and_source() -> None:
    stage = await run_answer_assessment_stage(
        answer_transcript="Some answer.",
        question_text="Q?",
        rubric_dimension="market_and_business_model",
        judge_provider=_provider(GroqPoolError("down"), GroqPoolError("down")),
    )
    assert stage.assessment.status == "not_evaluated"
    assert [lim.code for lim in stage.limitations] == ["answer_assessment_unavailable"]
    assert stage.metadata["assessment_status"] == "not_evaluated"
    assert stage.metadata["assessment_source"] == "unavailable"


async def test_skipped_answer_policy_unchanged() -> None:
    stage = await run_answer_assessment_stage(
        answer_transcript="   ",
        question_text="Q?",
        rubric_dimension="market_and_business_model",
        judge_provider=_provider(),
    )
    assert (stage.assessment.status, stage.assessment.score) == ("scored", 0.0)
    assert stage.metadata["skipped"] is True


def test_mixed_session_excludes_unavailable_but_keeps_valid_zero() -> None:
    questions = [{"id": q} for q in ("q1", "q2", "q3")]
    answers = [{"question_id": q, "status": "submitted"} for q in ("q1", "q2", "q3")]
    assessments = [
        {"question_id": "q1", "score": 0.8, "status": "scored"},
        {"question_id": "q2", "score": 0.0, "status": "scored"},  # genuinely poor answer
        {"question_id": "q3", "score": None, "status": "not_evaluated"},  # provider failure
    ]
    result = evaluate_qa_scores(questions=questions, answers=answers, assessments=assessments)
    assert result.score == pytest.approx(0.4)  # mean of 0.8 and 0.0; q3 excluded
    assert result.not_evaluated_question_ids == ["q3"]
    assert [lim.code for lim in result.limitations] == ["qa_assessment_unavailable"]


def test_assessment_status_and_score_must_agree() -> None:
    with pytest.raises(ValueError):
        AnswerAssessment(assessment_text="x", status="not_evaluated", score=0.0)
    with pytest.raises(ValueError):
        AnswerAssessment(assessment_text="x", status="scored", score=None)
