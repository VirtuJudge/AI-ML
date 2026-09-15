"""Unit tests for deterministic rubric scoring engine and Q&A integration."""

import pytest

from app.contracts import Limitation, ScoreComponent
from app.stages.scoring import (
    STARTUP_PITCH_RUBRIC_V1,
    calculate_individual_delivery_scores,
    calculate_overall_score,
    calculate_qa_score,
    canonicalize_dimension,
    evaluate_rubric,
    normalize_effective_weights,
    score_to_display,
    score_to_label,
)


# ============================================================================
# 1. Score Display and Label Mapping Tests
# ============================================================================


def test_score_to_display() -> None:
    """Verify conversion of normalized scores [0.0, 1.0] to rounded display scores [0, 100]."""
    assert score_to_display(0.0) == 0
    assert score_to_display(1.0) == 100
    assert score_to_display(0.744) == 74
    assert score_to_display(0.746) == 75
    assert score_to_display(0.595) == 60
    assert score_to_display(None) is None


def test_score_to_label() -> None:
    """Verify label boundaries on [0, 100] display scale:

    - [0, 40): needs_work
    - [40, 60): developing
    - [60, 80): good
    - [80, 100]: strong
    """
    assert score_to_label(0.0) == "needs_work"
    assert score_to_label(0.44) == "needs_work"
    assert score_to_label(0.444) == "needs_work"

    assert score_to_label(0.45) == "developing"
    assert score_to_label(0.50) == "developing"
    assert score_to_label(0.64) == "developing"

    assert score_to_label(0.65) == "good"
    assert score_to_label(0.70) == "good"
    assert score_to_label(0.84) == "good"

    assert score_to_label(0.85) == "strong"
    assert score_to_label(0.95) == "strong"
    assert score_to_label(1.00) == "strong"

    assert score_to_label(None) is None


def test_canonicalize_dimension() -> None:
    """Verify dimension aliases map to canonical rubric keys."""
    assert (
        canonicalize_dimension("business_reasoning")
        == "business_and_problem_solution_reasoning"
    )
    assert (
        canonicalize_dimension("market_and_business_model")
        == "business_and_problem_solution_reasoning"
    )
    assert canonicalize_dimension("delivery") == "delivery_and_body_language"
    assert canonicalize_dimension("timing") == "timing_and_speech_mechanics"
    assert canonicalize_dimension("qa") == "qa_quality"
    assert (
        canonicalize_dimension("pitch_content_and_evidence")
        == "pitch_content_and_evidence"
    )


# ============================================================================
# 2. Effective Weight Normalization Tests
# ============================================================================


def test_normalize_effective_weights_all_scored() -> None:
    """When all 6 rubric dimensions are scored, effective weights equal configured weights."""
    components = [
        ScoreComponent(
            dimension=dim,
            status="scored",
            normalized_score=0.80,
            configured_weight=weight,
        )
        for dim, weight in STARTUP_PITCH_RUBRIC_V1.items()
    ]

    normalize_effective_weights(components)

    for comp in components:
        assert comp.effective_weight == comp.configured_weight

    total_effective = sum(c.effective_weight for c in components)
    assert pytest.approx(total_effective, 0.001) == 1.0


def test_normalize_effective_weights_missing_dimension() -> None:
    """When a dimension cannot be evaluated (e.g. vision missing), effective weights normalize

    only across available scored dimensions.
    """
    # 5 scored dimensions, 1 not_evaluated (delivery_and_body_language = 0.15)
    components = [
        ScoreComponent(
            dimension="pitch_content_and_evidence",
            status="scored",
            normalized_score=0.80,
            configured_weight=0.25,
        ),
        ScoreComponent(
            dimension="business_and_problem_solution_reasoning",
            status="scored",
            normalized_score=0.75,
            configured_weight=0.20,
        ),
        ScoreComponent(
            dimension="technical_feasibility",
            status="scored",
            normalized_score=0.70,
            configured_weight=0.15,
        ),
        ScoreComponent(
            dimension="delivery_and_body_language",
            status="not_evaluated",
            configured_weight=0.15,
            limitation_code="no_video_stream",
        ),
        ScoreComponent(
            dimension="timing_and_speech_mechanics",
            status="scored",
            normalized_score=0.85,
            configured_weight=0.05,
        ),
        ScoreComponent(
            dimension="qa_quality",
            status="scored",
            normalized_score=0.90,
            configured_weight=0.20,
        ),
    ]

    normalize_effective_weights(components)

    # Scored sum = 0.25 + 0.20 + 0.15 + 0.05 + 0.20 = 0.85
    # Effective weights:
    # 0.25 / 0.85 = 0.2941
    # 0.20 / 0.85 = 0.2353
    # 0.15 / 0.85 = 0.1765
    # 0.05 / 0.85 = 0.0588
    # 0.20 / 0.85 = 0.2353
    scored_comps = [c for c in components if c.status == "scored"]
    unscored_comps = [c for c in components if c.status == "not_evaluated"]

    assert len(scored_comps) == 5
    assert len(unscored_comps) == 1
    assert unscored_comps[0].effective_weight is None

    total_effective = sum(c.effective_weight for c in scored_comps)
    assert pytest.approx(total_effective, 0.001) == 1.0

    comp_map = {c.dimension: c.effective_weight for c in components}
    assert comp_map["pitch_content_and_evidence"] == 0.2941
    assert comp_map["business_and_problem_solution_reasoning"] == 0.2353
    assert comp_map["delivery_and_body_language"] is None


def test_normalize_effective_weights_all_unscored() -> None:
    """When all dimensions are not_evaluated, effective weights and overall score are None."""
    components = [
        ScoreComponent(
            dimension="pitch_content_and_evidence",
            status="not_evaluated",
            configured_weight=0.25,
        ),
        ScoreComponent(
            dimension="qa_quality",
            status="not_evaluated",
            configured_weight=0.20,
        ),
    ]

    normalize_effective_weights(components)
    assert components[0].effective_weight is None
    assert components[1].effective_weight is None

    overall = calculate_overall_score(components)
    assert overall is None


# ============================================================================
# 3. Overall Score Calculation Tests
# ============================================================================


def test_calculate_overall_score() -> None:
    """Verify weighted overall score calculation."""
    components = [
        ScoreComponent(
            dimension="pitch_content_and_evidence",
            status="scored",
            normalized_score=0.80,
            configured_weight=0.25,
            effective_weight=0.25,
        ),
        ScoreComponent(
            dimension="business_and_problem_solution_reasoning",
            status="scored",
            normalized_score=0.70,
            configured_weight=0.20,
            effective_weight=0.20,
        ),
        ScoreComponent(
            dimension="technical_feasibility",
            status="scored",
            normalized_score=0.60,
            configured_weight=0.15,
            effective_weight=0.15,
        ),
        ScoreComponent(
            dimension="delivery_and_body_language",
            status="scored",
            normalized_score=0.80,
            configured_weight=0.15,
            effective_weight=0.15,
        ),
        ScoreComponent(
            dimension="timing_and_speech_mechanics",
            status="scored",
            normalized_score=0.90,
            configured_weight=0.05,
            effective_weight=0.05,
        ),
        ScoreComponent(
            dimension="qa_quality",
            status="scored",
            normalized_score=0.85,
            configured_weight=0.20,
            effective_weight=0.20,
        ),
    ]
    # Expected:
    # 0.80*0.25 + 0.70*0.20 + 0.60*0.15 + 0.80*0.15 + 0.90*0.05 + 0.85*0.20
    # = 0.200 + 0.140 + 0.090 + 0.120 + 0.045 + 0.170 = 0.765
    overall = calculate_overall_score(components)
    assert overall == 0.765
    assert score_to_display(overall) == 77
    assert score_to_label(overall) == "good"


# ============================================================================
# 4. Q&A Scoring & Skipped Question Tests
# ============================================================================


def test_calculate_qa_score_all_answered() -> None:
    """Verify Q&A scoring when all questions are answered."""
    questions = [
        {"id": "q1", "text": "Q1"},
        {"id": "q2", "text": "Q2"},
        {"id": "q3", "text": "Q3"},
    ]
    answers = [
        {"question_id": "q1", "status": "submitted", "transcript": "Ans 1"},
        {"question_id": "q2", "status": "submitted", "transcript": "Ans 2"},
        {"question_id": "q3", "status": "submitted", "transcript": "Ans 3"},
    ]
    assessments = [
        {"question_id": "q1", "score": 0.90, "evidence_ids": ["ev_ans_1"]},
        {"question_id": "q2", "score": 0.80, "evidence_ids": ["ev_ans_2"]},
        {"question_id": "q3", "score": 0.70, "evidence_ids": ["ev_ans_3"]},
    ]

    qa_score, ev_ids = calculate_qa_score(questions, answers, assessments)

    assert qa_score == 0.80  # (0.90 + 0.80 + 0.70) / 3
    assert len(ev_ids) == 3


def test_calculate_qa_score_skipped_question_scores_zero() -> None:
    """CRITICAL RULE: Explicitly skipped questions score 0.0 and count in the denominator.

    The system must NOT remove the Q&A weight or omit the question to artificially
    inflate the score.
    """
    questions = [
        {"id": "q1", "text": "Q1"},
        {"id": "q2", "text": "Q2"},
        {"id": "q3", "text": "Q3"},
    ]
    answers = [
        {"question_id": "q1", "status": "submitted", "transcript": "Ans 1"},
        {"question_id": "q2", "status": "submitted", "transcript": "Ans 2"},
        {"question_id": "q3", "status": "skipped"},  # SKIPPED
    ]
    assessments = [
        {"question_id": "q1", "score": 0.90, "evidence_ids": ["ev_1"]},
        {"question_id": "q2", "score": 0.70, "evidence_ids": ["ev_2"]},
    ]

    qa_score, ev_ids = calculate_qa_score(questions, answers, assessments)

    # (0.90 + 0.70 + 0.0) / 3 = 1.60 / 3 = 0.5333
    assert qa_score == 0.5333
    assert score_to_display(qa_score) == 53
    assert score_to_label(qa_score) == "developing"


def test_calculate_qa_score_all_skipped() -> None:
    """When all questions are skipped, Q&A score is 0.0."""
    questions = [
        {"id": "q1", "text": "Q1"},
        {"id": "q2", "text": "Q2"},
    ]
    answers = [
        {"question_id": "q1", "status": "skipped"},
        {"question_id": "q2", "status": "skipped"},
    ]

    qa_score, ev_ids = calculate_qa_score(questions, answers)
    assert qa_score == 0.0
    assert score_to_display(qa_score) == 0
    assert score_to_label(qa_score) == "needs_work"


def test_calculate_qa_score_missing_assessment_score_awards_no_credit() -> None:
    """A transcript alone must not silently earn the historical 75% fallback."""
    questions = [{"id": "q1", "text": "Q1"}]
    answers = [{"question_id": "q1", "status": "submitted", "transcript": "A response"}]
    assessments = [{"question_id": "q1", "assessment_text": "Incomplete response."}]

    qa_score, _ = calculate_qa_score(questions, answers, assessments)

    assert qa_score == 0.0


def test_calculate_qa_score_empty_questions() -> None:
    """Graceful handling when no questions exist."""
    qa_score, ev_ids = calculate_qa_score([], [])
    assert qa_score == 0.0
    assert ev_ids == []


# ============================================================================
# 5. Full Rubric Evaluation Tests
# ============================================================================


def test_evaluate_rubric_full() -> None:
    """Verify evaluate_rubric computes all dimensions, weights, and overall score."""
    raw_scores = {
        "pitch_content_and_evidence": 0.85,
        "business_and_problem_solution_reasoning": 0.78,
        "technical_feasibility": 0.80,
        "delivery_and_body_language": 0.72,
        "timing_and_speech_mechanics": 0.88,
        "qa_quality": 0.80,
    }

    evidence_map = {
        "pitch_content_and_evidence": ["ev_speech_001", "ev_doc_001"],
        "qa_quality": ["ev_qa_001"],
    }

    overall, components = evaluate_rubric(
        raw_scores,
        rubric_id="startup_pitch",
        evidence_by_dimension=evidence_map,
    )

    assert overall is not None
    assert len(components) == 6

    # All scored
    assert all(c.status == "scored" for c in components)
    assert all(c.effective_weight == c.configured_weight for c in components)

    # Check evidence binding
    p_comp = next(c for c in components if c.dimension == "pitch_content_and_evidence")
    assert p_comp.evidence_ids == ["ev_speech_001", "ev_doc_001"]
    assert p_comp.label == "strong"
    assert p_comp.display_score == 85


def test_evaluate_rubric_with_missing_dimension_and_limitation() -> None:
    """Verify evaluate_rubric attaches limitation code and normalizes remaining dimensions."""
    raw_scores = {
        "pitch_content_and_evidence": 0.80,
        "business_and_problem_solution_reasoning": 0.70,
        "technical_feasibility": 0.75,
        "delivery_and_body_language": None,  # MISSING (e.g. video corrupt)
        "timing_and_speech_mechanics": 0.85,
        "qa_quality": 0.80,
    }

    lims = [
        Limitation(
            code="video_stream_unavailable",
            scope="vision",
            message="No video track present.",
            affected_dimensions=["delivery_and_body_language"],
        )
    ]

    overall, components = evaluate_rubric(
        raw_scores,
        rubric_id="startup_pitch",
        limitations=lims,
    )

    assert overall is not None
    del_comp = next(c for c in components if c.dimension == "delivery_and_body_language")
    assert del_comp.status == "not_evaluated"
    assert del_comp.limitation_code == "video_stream_unavailable"
    assert del_comp.effective_weight is None
    assert del_comp.normalized_score is None

    # Remaining 5 dimensions normalized
    scored = [c for c in components if c.status == "scored"]
    assert len(scored) == 5
    assert pytest.approx(sum(c.effective_weight for c in scored), 0.001) == 1.0


# ============================================================================
# 6. Individual Delivery Scoring Tests
# ============================================================================


def test_calculate_individual_delivery_scores() -> None:
    """Verify individual delivery scoring from windowed acoustic & visual observations."""
    speaker_profile = {
        "speaking_time_ms": 60000,
        "windows": [
            {
                "start_ms": 0,
                "end_ms": 10000,
                "acoustic": {
                    "speaking_rate_wpm": 140.0,
                    "pause_duration_ms": 200,
                    "filler_count": 0,
                },
                "visual": {
                    "gaze_direction": 1.0,
                    "posture_openness": 0.88,
                    "shoulder_symmetry_ratio": 0.96,
                },
            },
            {
                "start_ms": 10000,
                "end_ms": 20000,
                "acoustic": {
                    "speaking_rate_wpm": 145.0,
                    "pause_duration_ms": 300,
                    "filler_count": 1,
                },
                "visual": {
                    "gaze_direction": 1.1,
                    "posture_openness": 0.85,
                    "shoulder_symmetry_ratio": 0.94,
                },
            },
        ],
    }

    comps = calculate_individual_delivery_scores(speaker_profile)

    assert len(comps) == 2
    del_comp = next(c for c in comps if c.dimension == "delivery_and_body_language")
    timing_comp = next(c for c in comps if c.dimension == "timing_and_speech_mechanics")

    assert del_comp.status == "scored"
    assert del_comp.normalized_score is not None
    assert del_comp.normalized_score > 0.80
    assert del_comp.label in ("good", "strong")

    assert timing_comp.status == "scored"
    assert timing_comp.normalized_score is not None
    assert timing_comp.normalized_score > 0.80
    assert timing_comp.label in ("good", "strong")


def test_calculate_individual_delivery_scores_missing_video() -> None:
    """Verify missing visual observations sets delivery_and_body_language to not_evaluated."""
    speaker_profile = {
        "speaking_time_ms": 30000,
        "windows": [
            {
                "start_ms": 0,
                "end_ms": 10000,
                "acoustic": {
                    "speaking_rate_wpm": 138.0,
                    "pause_duration_ms": 200,
                    "filler_count": 0,
                },
            }
        ],
    }

    comps = calculate_individual_delivery_scores(speaker_profile)

    del_comp = next(c for c in comps if c.dimension == "delivery_and_body_language")
    timing_comp = next(c for c in comps if c.dimension == "timing_and_speech_mechanics")

    assert del_comp.status == "not_evaluated"
    assert del_comp.limitation_code == "no_visual_observations"
    assert timing_comp.status == "scored"
