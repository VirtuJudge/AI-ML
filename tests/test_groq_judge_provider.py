"""Unit tests for app.providers.groq_judge (GroqJudgeModelProvider)."""

from unittest.mock import AsyncMock

import pytest

from app.contracts import PrimaryQuestion
from app.providers.base import JudgeModelProvider
from app.providers.groq_judge import GroqJudgeModelProvider
from app.providers.groq_pool import GroqKeyPool, GroqPoolError
from app.providers.types import DocumentChunk
from app.stages.evidence import EvidenceBundle, EvidenceItem


def _make_test_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        items=[
            EvidenceItem(
                evidence_id="ev_speech_001",
                source="speech",
                title="Revenue Model",
                text="We charge $50/mo per active seat.",
                rubric_dimensions=["market_and_business_model"],
            ),
            EvidenceItem(
                evidence_id="ev_speech_002",
                source="speech",
                title="Proprietary Pipeline",
                text="Our pipeline uses a custom C++ engine for real-time inference.",
                rubric_dimensions=["technology_and_moat"],
            ),
            EvidenceItem(
                evidence_id="ev_doc_001",
                source="documents",
                page_or_slide=2,
                title="Pilot Traction",
                text="3 enterprise pilots completed with 95% retention.",
                rubric_dimensions=["execution_and_milestones"],
            ),
        ],
        transcript_full_text="We charge $50/mo per active seat. Our pipeline uses a custom engine.",
        speaker_labels=["SPEAKER_00"],
    )


def test_groq_judge_provider_satisfies_protocol() -> None:
    """Verify GroqJudgeModelProvider conforms to JudgeModelProvider Protocol."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    judge_protocol: JudgeModelProvider = provider
    assert hasattr(judge_protocol, "generate_questions")
    assert callable(judge_protocol.generate_questions)


@pytest.mark.asyncio
async def test_generate_questions_happy_path() -> None:
    """Verify parallel 3-judge panel executes and returns 3 grounded questions."""
    mock_pool = AsyncMock(spec=GroqKeyPool)

    # Return valid responses tailored per judge role
    async def mock_chat(model: str, messages: list[dict[str, str]], **kwargs: object) -> str:
        role_pref = kwargs.get("preferred_key_index")
        if role_pref == 0:
            return (
                '{"text": "What is the expected customer lifetime value given $50/mo?", '
                '"reason": "Validates unit economics.", '
                '"rubric_dimension": "market_and_business_model", '
                '"evidence_ids": ["ev_speech_001"]}'
            )
        elif role_pref == 1:
            return (
                '{"text": "How will the custom C++ engine scale horizontally under peak load?", '
                '"reason": "Tests architecture feasibility.", '
                '"rubric_dimension": "technology_and_moat", '
                '"evidence_ids": ["ev_speech_002"]}'
            )
        else:
            return (
                '{"text": "What concrete criteria secured the 95% retention across the 3 pilots?", '
                '"reason": "Validates execution milestones.", '
                '"rubric_dimension": "execution_and_milestones", '
                '"evidence_ids": ["ev_doc_001"]}'
            )

    mock_pool.chat.side_effect = mock_chat
    provider = GroqJudgeModelProvider(key_pool=mock_pool)

    bundle = _make_test_bundle()
    questions = await provider.generate_questions(
        transcript=bundle.transcript_full_text,
        rubric_id="startup_pitch",
        evidence_bundle=bundle,
    )

    assert len(questions) == 3
    assert all(isinstance(q, PrimaryQuestion) for q in questions)
    assert len({q.rubric_dimension for q in questions}) == 3
    assert mock_pool.chat.call_count == 3

    dims = {q.rubric_dimension for q in questions}
    assert "market_and_business_model" in dims
    assert "technology_and_moat" in dims
    assert "execution_and_milestones" in dims


@pytest.mark.asyncio
async def test_generate_questions_with_markdown_fences() -> None:
    """Verify extraction handles Markdown ```json ... ``` code fences."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    mock_pool.chat.return_value = """```json
{
  "text": "What are your variable cloud costs per active seat?",
  "reason": "Clarifies gross margin assumptions.",
  "rubric_dimension": "market_and_business_model",
  "evidence_ids": ["ev_speech_001"]
}
```"""
    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    bundle = _make_test_bundle()

    questions = await provider.generate_questions(
        transcript=bundle.transcript_full_text,
        rubric_id="startup_pitch",
        evidence_bundle=bundle,
    )

    assert len(questions) == 3
    assert "variable cloud costs" in questions[0].text


@pytest.mark.asyncio
async def test_generate_questions_retries_with_fallback_model() -> None:
    """Verify judge retries with fallback model and spare key on primary failure."""
    mock_pool = AsyncMock(spec=GroqKeyPool)

    # First call fails, second succeeds with fallback
    mock_pool.chat.side_effect = [
        GroqPoolError("Model timeout"),
        (
            '{"text": "Fallback question text?", "reason": "Reason", '
            '"rubric_dimension": "market_and_business_model", '
            '"evidence_ids": ["ev_speech_001"]}'
        ),
        (
            '{"text": "Tech question text?", "reason": "Reason", '
            '"rubric_dimension": "technology_and_moat", '
            '"evidence_ids": ["ev_speech_002"]}'
        ),
        (
            '{"text": "Product question text?", "reason": "Reason", '
            '"rubric_dimension": "execution_and_milestones", '
            '"evidence_ids": ["ev_doc_001"]}'
        ),
    ]

    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    bundle = _make_test_bundle()

    questions = await provider.generate_questions(
        transcript=bundle.transcript_full_text,
        rubric_id="startup_pitch",
        evidence_bundle=bundle,
    )

    assert len(questions) == 3
    assert any(q.text == "Fallback question text?" for q in questions)


@pytest.mark.asyncio
async def test_generate_questions_complete_failure_emits_grounded_fallback() -> None:
    """Verify total provider failure returns deterministic grounded fallbacks."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    mock_pool.chat.side_effect = GroqPoolError("Network unreachable")

    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    bundle = _make_test_bundle()

    questions = await provider.generate_questions(
        transcript=bundle.transcript_full_text,
        rubric_id="startup_pitch",
        evidence_bundle=bundle,
    )

    assert len(questions) == 3
    for q in questions:
        assert isinstance(q, PrimaryQuestion)
        assert len(q.candidate_id) == 26
        assert len(q.evidence_ids) >= 1
        assert bundle.has_evidence_id(q.evidence_ids[0])


@pytest.mark.asyncio
async def test_generate_questions_without_prebuilt_bundle() -> None:
    """Verify provider creates bundle automatically when not provided."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    mock_pool.chat.return_value = (
        '{"text": "What is the CAC payback period?", '
        '"reason": "Evaluates financial timeline.", '
        '"rubric_dimension": "market_and_business_model", '
        '"evidence_ids": ["ev_speech_001"]}'
    )

    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            page_or_slide=1,
            text="VirtuJudge Pitch Deck.",
        )
    ]

    questions = await provider.generate_questions(
        transcript="Our company automates sales pitch judging.",
        rubric_id="startup_pitch",
        document_chunks=chunks,
    )

    assert len(questions) == 3


@pytest.mark.asyncio
async def test_generate_questions_preserves_per_judge_limitations() -> None:
    """Verify limitations from rejected judge validations are captured in bundle.limitations."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    # Judge 0 attempt 1 cites non-existent evidence ID; retry succeeds
    mock_pool.chat.side_effect = [
        (
            '{"text": "Ungrounded question?", "reason": "Reason", '
            '"rubric_dimension": "market_and_business_model", '
            '"evidence_ids": ["ev_nonexistent_999"]}'
        ),
        (
            '{"text": "Grounded business question?", "reason": "Reason", '
            '"rubric_dimension": "market_and_business_model", '
            '"evidence_ids": ["ev_speech_001"]}'
        ),
        (
            '{"text": "Grounded tech question?", "reason": "Reason", '
            '"rubric_dimension": "technology_and_moat", '
            '"evidence_ids": ["ev_speech_002"]}'
        ),
        (
            '{"text": "Grounded execution question?", "reason": "Reason", '
            '"rubric_dimension": "execution_and_milestones", '
            '"evidence_ids": ["ev_doc_001"]}'
        ),
    ]

    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    bundle = _make_test_bundle()

    questions = await provider.generate_questions(
        transcript=bundle.transcript_full_text,
        rubric_id="startup_pitch",
        evidence_bundle=bundle,
    )

    assert len(questions) == 3
    limitation_codes = [lim.code for lim in bundle.limitations]
    assert "unsupported_question_rejected" in limitation_codes


def test_parse_and_validate_helper() -> None:
    """Verify _parse_and_validate deduplicated helper handles valid and invalid payloads."""
    from app.stages.judge_panel import JUDGE_PANEL

    mock_pool = AsyncMock(spec=GroqKeyPool)
    provider = GroqJudgeModelProvider(key_pool=mock_pool)
    bundle = _make_test_bundle()
    judge = JUDGE_PANEL[0]

    # Non-JSON payload returns None and malformed_question_json limitation
    non_json = "I am sorry, as an AI model I cannot assist with this."
    val, lims = provider._parse_and_validate(non_json, judge, bundle)
    assert val is None
    assert len(lims) == 1
    assert lims[0].code == "malformed_question_json"

    # Valid JSON payload returns PrimaryQuestion and no limitations
    valid_json = (
        '{"text": "What is the churn rate?", "reason": "Evaluates retention.", '
        '"rubric_dimension": "market_and_business_model", "evidence_ids": ["ev_speech_001"]}'
    )
    val2, lims2 = provider._parse_and_validate(valid_json, judge, bundle)
    assert val2 is not None
    assert val2.text == "What is the churn rate?"
    assert len(lims2) == 0


def test_groq_judge_fallback_assessment_has_no_followup_and_empty_evidence() -> None:
    """Verify fallback assessment never generates follow-ups and returns honest empty evidence."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    provider = GroqJudgeModelProvider(key_pool=mock_pool)

    # Even with remaining_follow_ups=5, fallback assessment MUST NOT generate follow-ups
    fallback = provider._create_fallback_assessment(
        question_text="What is your CAC?",
        rubric_dimension="market_and_business_model",
        remaining_follow_ups=5,
    )
    assert fallback.follow_up is None
    assert fallback.evidence_ids == []
    assert "team member" in fallback.assessment_text


def test_groq_judge_parse_assessment_empty_evidence_ids() -> None:
    """Verify missing evidence in model output results in empty list, not fabricated IDs."""
    mock_pool = AsyncMock(spec=GroqKeyPool)
    provider = GroqJudgeModelProvider(key_pool=mock_pool)

    # Payload with empty evidence_ids
    raw_text = (
        '{"assessment_text": "Good answer on unit economics.", '
        '"evidence_ids": [], "follow_up": null}'
    )
    res = provider._parse_assessment(
        raw_text=raw_text,
        rubric_dimension="market_and_business_model",
        remaining_follow_ups=0,
    )
    assert res is not None
    assert res.evidence_ids == []
    assert res.follow_up is None



