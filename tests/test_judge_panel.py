"""Unit tests for app.stages.judge_panel (Personas, Prompts, and Guardrails)."""

import ulid

from app.contracts import PrimaryQuestion
from app.stages.evidence import EvidenceBundle, EvidenceItem
from app.stages.judge_panel import (
    JUDGE_PANEL,
    create_fallback_question,
    format_prompt_for_judge,
    strip_evidence_ids_from_question_text,
    validate_judge_question,
    validate_panel_questions,
)


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


def test_judge_panel_configuration() -> None:
    """Verify JUDGE_PANEL defines exactly 3 judges with distinct key indices."""
    assert len(JUDGE_PANEL) == 3
    roles = [j.role for j in JUDGE_PANEL]
    assert roles == ["business_strategist", "technical_evaluator", "product_analyst"]

    key_indices = [j.preferred_key_index for j in JUDGE_PANEL]
    assert key_indices == [0, 1, 2]

    for j in JUDGE_PANEL:
        assert j.system_prompt
        assert len(j.rubric_dimensions) >= 2
        assert j.default_rubric_dimension in j.rubric_dimensions


def test_format_prompt_for_judge() -> None:
    """Verify prompt formatting injects evidence summary and transcript."""
    bundle = _make_test_bundle()
    messages = format_prompt_for_judge(JUDGE_PANEL[0], bundle)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Business Strategist" in messages[0]["content"]
    assert "ev_speech_001" in messages[1]["content"]
    assert "We charge $50/mo" in messages[1]["content"]


def test_validate_judge_question_valid() -> None:
    """Verify a valid grounded question passes validation and has valid ULID."""
    bundle = _make_test_bundle()
    q = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is the gross margin on the $50/mo subscription?",
        reason="Assesses unit economics feasibility.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )

    validated, lims = validate_judge_question(q, JUDGE_PANEL[0], bundle)
    assert validated is not None
    assert validated.text == q.text
    assert validated.rubric_dimension == "market_and_business_model"
    assert validated.evidence_ids == ["ev_speech_001"]
    assert len(lims) == 0
    assert validated.candidate_id == q.candidate_id


def test_validate_judge_question_strips_evidence_ids_from_candidate_text() -> None:
    """Evidence references remain structured metadata, never part of the question wording."""
    bundle = _make_test_bundle()
    q = PrimaryQuestion(
        candidate_id="q-1",
        text="How is the CAC payback measured? (ev_speech_001)",
        reason="Tests whether the claim is measurable.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )

    validated, _ = validate_judge_question(q, JUDGE_PANEL[0], bundle)

    assert validated is not None
    assert validated.text == "How is the CAC payback measured?"
    assert validated.evidence_ids == ["ev_speech_001"]
    assert strip_evidence_ids_from_question_text("Evidence ID: ev_speech_001") == ""


def test_validate_judge_question_preserves_custom_candidate_ids() -> None:
    """Verify non-ULID candidate IDs like 'q-1' are preserved per Backend contract."""
    bundle = _make_test_bundle()

    # Case 1: Custom short candidate ID preserved
    q_custom = PrimaryQuestion(
        candidate_id="q-1",
        text="What is the gross margin on the $50/mo subscription?",
        reason="Assesses unit economics feasibility.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )
    val_custom, _ = validate_judge_question(q_custom, JUDGE_PANEL[0], bundle)
    assert val_custom is not None
    assert val_custom.candidate_id == "q-1"

    # Case 2: Empty candidate ID automatically gets a new ULID
    q_empty = PrimaryQuestion(
        candidate_id="",
        text="What is the gross margin on the $50/mo subscription?",
        reason="Assesses unit economics feasibility.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )
    val_empty, _ = validate_judge_question(q_empty, JUDGE_PANEL[0], bundle)
    assert val_empty is not None
    assert len(val_empty.candidate_id) == 26
    assert ulid.parse(val_empty.candidate_id) is not None



def test_validate_judge_question_rejects_emotional_claims() -> None:
    """Verify guardrails strictly reject questions commenting on confidence or anxiety."""
    bundle = _make_test_bundle()
    emotional_questions = [
        PrimaryQuestion(
            candidate_id="q1",
            text="Why did the speaker appear nervous when discussing revenue?",
            reason="Nervousness indicates possible deception.",
            rubric_dimension="market_and_business_model",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="q2",
            text="Can you explain your revenue model?",
            reason="The speaker lacked confidence during the slide delivery.",
            rubric_dimension="market_and_business_model",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="q3",
            text="Is the team being dishonest about customer churn?",
            reason="The body language shows fear of answering.",
            rubric_dimension="business_reasoning",
            evidence_ids=["ev_speech_001"],
        ),
        PrimaryQuestion(
            candidate_id="q4",
            text="Is the founder a liar about their traction?",
            reason="Lying suggests false metrics.",
            rubric_dimension="business_reasoning",
            evidence_ids=["ev_speech_001"],
        ),
    ]

    for q in emotional_questions:
        validated, lims = validate_judge_question(q, JUDGE_PANEL[0], bundle)
        assert validated is None
        assert len(lims) == 1
        assert lims[0].code == "subjective_emotional_claim_rejected"


def test_validate_judge_question_rejects_unsupported_evidence() -> None:
    """Verify unknown/unsupported evidence IDs fail validation per acceptance criteria."""
    bundle = _make_test_bundle()
    q = PrimaryQuestion(
        candidate_id="q1",
        text="What is your customer retention strategy?",
        reason="Customer retention is critical.",
        rubric_dimension="customer_retention",
        evidence_ids=["ev_fabricated_999"],
    )

    validated, lims = validate_judge_question(q, JUDGE_PANEL[0], bundle)
    assert validated is None
    assert len(lims) == 1
    assert lims[0].code == "unsupported_question_rejected"


def test_validate_panel_questions_deduplication() -> None:
    """Verify duplicate questions are filtered and filled to maintain exactly 3."""
    bundle = _make_test_bundle()
    q1 = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is your projected customer acquisition cost at scale?",
        reason="Assesses financial model.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )
    # Duplicate question text
    q2 = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is your projected customer acquisition cost at scale?",
        reason="Assesses financial model.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )

    final_questions, lims = validate_panel_questions([q1, q2], bundle)
    assert len(final_questions) == 3
    # Check that all 3 question texts are distinct
    texts = [q.text for q in final_questions]
    assert len(set(texts)) == 3
    assert any(lim.code == "duplicate_question_filtered" for lim in lims)


def test_create_fallback_question() -> None:
    """Verify create_fallback_question produces a fully compliant question."""
    bundle = _make_test_bundle()
    for judge in JUDGE_PANEL:
        q = create_fallback_question(judge, bundle)
        assert len(q.candidate_id) == 26
        assert q.text == judge.fallback_text
        assert q.reason == judge.fallback_reason
        assert q.rubric_dimension == judge.default_rubric_dimension
        assert len(q.evidence_ids) >= 1
        assert bundle.has_evidence_id(q.evidence_ids[0])


def test_create_fallback_question_empty_bundle() -> None:
    """Verify create_fallback_question does not fabricate citations when bundle is empty."""
    bundle = EvidenceBundle(items=[])
    for judge in JUDGE_PANEL:
        q = create_fallback_question(judge, bundle)
        assert q.evidence_ids == []


def test_format_evidence_for_prompt() -> None:
    """Verify format_summary_for_prompt formats evidence summary."""
    bundle = _make_test_bundle()
    text = bundle.format_summary_for_prompt()
    assert "ev_speech_001" in text
    assert "We charge $50/mo" in text


def test_validate_panel_questions_with_judges_argument() -> None:
    """Verify validate_panel_questions accepts optional judges list."""
    bundle = _make_test_bundle()
    q = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is the gross margin on the subscription?",
        reason="Assesses unit economics.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )
    # Call with canonical signature and explicit judges argument
    final_questions, _ = validate_panel_questions([q], bundle, judges=JUDGE_PANEL)
    assert len(final_questions) == 3
