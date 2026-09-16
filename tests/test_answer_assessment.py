"""Unit tests for app.stages.answer_assessment (run_answer_assessment_stage)."""

import pytest

from app.contracts import FollowUpQuestion
from app.providers.fake_judge import FakeJudgeModelProvider
from app.stages.answer_assessment import AnswerAssessmentResult, run_answer_assessment_stage


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_with_follow_up() -> None:
    """Verify assessment with remaining_follow_ups=1 produces assessment and follow-up."""
    judge_provider = FakeJudgeModelProvider()
    rubric_dimension = "market_and_business_model"
    question_text = "What is your projected CAC at scale?"
    answer_transcript = "Our CAC will decrease significantly as referral loops kick in."

    result = await run_answer_assessment_stage(
        answer_transcript=answer_transcript,
        question_text=question_text,
        rubric_dimension=rubric_dimension,
        judge_provider=judge_provider,
        remaining_follow_ups=1,
    )

    assert isinstance(result, AnswerAssessmentResult)
    assert result.assessment.assessment_text != ""
    assert result.assessment.score == 0.65
    assert result.assessment.evidence_ids == []
    assert result.assessment.follow_up is not None
    assert isinstance(result.assessment.follow_up, FollowUpQuestion)
    assert result.assessment.follow_up.text != ""
    assert result.assessment.follow_up.reason != ""
    assert result.assessment.follow_up.rubric_dimension == rubric_dimension
    assert result.assessment.follow_up.evidence_ids == []
    assert result.metadata["stage"] == "answer_assessment"
    assert result.metadata["has_follow_up"] is True
    assert result.metadata["judge_provider"] == "FakeJudgeModelProvider"


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_without_follow_up() -> None:
    """Verify assessment with remaining_follow_ups=0 produces assessment but follow_up is None."""
    judge_provider = FakeJudgeModelProvider()
    rubric_dimension = "technology_and_moat"
    question_text = "How does your proprietary tech protect against competitors?"
    answer_transcript = "We have patented algorithms and proprietary data flywheels."

    result = await run_answer_assessment_stage(
        answer_transcript=answer_transcript,
        question_text=question_text,
        rubric_dimension=rubric_dimension,
        judge_provider=judge_provider,
        remaining_follow_ups=0,
    )

    assert isinstance(result, AnswerAssessmentResult)
    assert result.assessment.assessment_text != ""
    assert result.assessment.score == 0.65
    assert result.assessment.evidence_ids == []
    assert result.assessment.follow_up is None
    assert result.metadata["has_follow_up"] is False
    assert result.metadata["rubric_dimension"] == rubric_dimension


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_skipped_empty_transcript() -> None:
    """Verify empty transcript produces empty assessment, no follow-up, and skipped metadata."""
    judge_provider = FakeJudgeModelProvider()
    rubric_dimension = "execution_and_milestones"
    question_text = "What milestones must be achieved?"

    result = await run_answer_assessment_stage(
        answer_transcript="",
        question_text=question_text,
        rubric_dimension=rubric_dimension,
        judge_provider=judge_provider,
        remaining_follow_ups=1,
    )

    assert isinstance(result, AnswerAssessmentResult)
    assert result.assessment.assessment_text == ""
    assert result.assessment.score == 0.0
    assert result.assessment.evidence_ids == []
    assert result.assessment.follow_up is None
    assert result.limitations == []
    assert result.metadata.get("skipped") is True
    assert result.metadata.get("stage") == "answer_assessment"
    assert result.metadata.get("rubric_dimension") == rubric_dimension


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_skipped_whitespace_transcript() -> None:
    """Verify whitespace-only transcript is treated as skipped."""
    judge_provider = FakeJudgeModelProvider()

    result = await run_answer_assessment_stage(
        answer_transcript="   \n\t  ",
        question_text="Any question?",
        rubric_dimension="team_and_execution",
        judge_provider=judge_provider,
        remaining_follow_ups=2,
    )

    assert result.assessment.assessment_text == ""
    assert result.assessment.score == 0.0
    assert result.assessment.evidence_ids == []
    assert result.assessment.follow_up is None
    assert result.metadata.get("skipped") is True


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_follow_up_contract_fields() -> None:
    """Verify follow-up question strictly conforms to the FollowUpQuestion contract."""
    judge_provider = FakeJudgeModelProvider()
    rubric_dimension = "market_and_business_model"

    result = await run_answer_assessment_stage(
        answer_transcript="We plan to expand to enterprise customers next quarter.",
        question_text="What are your go-to-market assumptions?",
        rubric_dimension=rubric_dimension,
        judge_provider=judge_provider,
        remaining_follow_ups=2,
    )

    follow_up = result.assessment.follow_up
    assert follow_up is not None
    assert isinstance(follow_up, FollowUpQuestion)
    assert isinstance(follow_up.text, str) and len(follow_up.text.strip()) > 0
    assert isinstance(follow_up.reason, str) and len(follow_up.reason.strip()) > 0
    assert follow_up.rubric_dimension == rubric_dimension
    assert isinstance(follow_up.evidence_ids, list)
    assert follow_up.evidence_ids == []


@pytest.mark.asyncio
async def test_run_answer_assessment_stage_disqualifies_direct_judge_abuse() -> None:
    """Direct insults aimed at the judge receive no score without querying the model."""
    result = await run_answer_assessment_stage(
        answer_transcript="You are an idiot and this question is stupid.",
        question_text="What is your CAC?",
        rubric_dimension="market_and_business_model",
        judge_provider=FakeJudgeModelProvider(),
        remaining_follow_ups=1,
    )

    assert result.assessment.score == 0.0
    assert result.assessment.follow_up is None
    assert result.metadata["disqualified_for_judge_directed_abuse"] is True
