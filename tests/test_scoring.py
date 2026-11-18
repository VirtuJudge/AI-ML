"""Unit tests for deterministic rubric scoring engine and Q&A integration."""

import pytest
from pydantic import ValidationError

from app.contracts import Limitation, ScoreComponent
from app.stages.scoring import (
    DEFAULT_LABEL_BANDS,
    RUBRIC_DEFINITIONS,
    STARTUP_PITCH_RUBRIC_V1,
    WEIGHTING_POLICY_VERSION,
    LabelBands,
    RubricDefinition,
    calculate_individual_delivery_scores,
    calculate_overall_score,
    calculate_qa_score,
    canonicalize_dimension,
    evaluate_qa_scores,
    evaluate_rubric,
    get_rubric_definition,
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
    """Verify default label boundaries on [0, 100] display scale:

    - [0, 40): needs_work
    - [40, 60): developing
    - [60, 80): good
    - [80, 100]: strong
    """
    assert score_to_label(0.0) == "needs_work"
    assert score_to_label(0.39) == "needs_work"
    assert score_to_label(0.394) == "needs_work"

    assert score_to_label(0.395) == "developing"
    assert score_to_label(0.40) == "developing"
    assert score_to_label(0.50) == "developing"
    assert score_to_label(0.59) == "developing"
    assert score_to_label(0.594) == "developing"

    assert score_to_label(0.595) == "good"
    assert score_to_label(0.60) == "good"
    assert score_to_label(0.62) == "good"
    assert score_to_label(0.70) == "good"
    assert score_to_label(0.79) == "good"
    assert score_to_label(0.794) == "good"

    assert score_to_label(0.795) == "strong"
    assert score_to_label(0.80) == "strong"
    assert score_to_label(0.82) == "strong"
    assert score_to_label(0.85) == "strong"
    assert score_to_label(0.95) == "strong"
    assert score_to_label(1.00) == "strong"

    assert score_to_label(None) is None


def test_canonicalize_dimension() -> None:
    """Verify dimension aliases map to canonical rubric keys."""
    assert canonicalize_dimension("business_reasoning") == "business_and_problem_solution_reasoning"
    assert (
        canonicalize_dimension("market_and_business_model")
        == "business_and_problem_solution_reasoning"
    )
    assert canonicalize_dimension("delivery") == "delivery_and_body_language"
    assert canonicalize_dimension("timing") == "timing_and_speech_mechanics"
    assert canonicalize_dimension("qa") == "qa_quality"
    assert canonicalize_dimension("pitch_content_and_evidence") == "pitch_content_and_evidence"


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
    """When a presentation dimension cannot be evaluated (e.g. vision missing), Q&A stays fixed

    at 0.20 and the remaining 0.80 is redistributed proportionally across scored presentation
    dimensions.
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

    # Non-Q&A scored sum = 0.25 + 0.20 + 0.15 + 0.05 = 0.65; remaining budget = 0.80
    # Effective weights:
    # pitch:     0.80 * (0.25 / 0.65) = 0.3077
    # business:  0.80 * (0.20 / 0.65) = 0.2462
    # technical: 0.80 * (0.15 / 0.65) = 0.1846
    # timing:    0.80 * (0.05 / 0.65) = 0.0615
    # qa_quality: 0.20 (fixed)
    scored_comps = [c for c in components if c.status == "scored"]
    unscored_comps = [c for c in components if c.status == "not_evaluated"]

    assert len(scored_comps) == 5
    assert len(unscored_comps) == 1
    assert unscored_comps[0].effective_weight is None

    total_effective = sum(
        c.effective_weight for c in scored_comps if c.effective_weight is not None
    )
    assert pytest.approx(total_effective, 0.001) == 1.0

    comp_map = {c.dimension: c.effective_weight for c in components}
    assert comp_map["pitch_content_and_evidence"] == 0.3077
    assert comp_map["business_and_problem_solution_reasoning"] == 0.2462
    assert comp_map["technical_feasibility"] == 0.1846
    assert comp_map["timing_and_speech_mechanics"] == 0.0615
    assert comp_map["delivery_and_body_language"] is None
    assert comp_map["qa_quality"] == 0.20


def test_normalize_effective_weights_only_qa_scored() -> None:
    """When only qa_quality is scored and all other dimensions are not_evaluated, Q&A takes 1.0."""
    components = [
        ScoreComponent(
            dimension="pitch_content_and_evidence",
            status="not_evaluated",
            configured_weight=0.25,
        ),
        ScoreComponent(
            dimension="qa_quality",
            status="scored",
            normalized_score=0.75,
            configured_weight=0.20,
        ),
    ]

    normalize_effective_weights(components)

    assert components[0].effective_weight is None
    assert components[1].effective_weight == 1.0
    assert calculate_overall_score(components) == 0.75


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


def test_calculate_qa_score_missing_assessment_score_is_not_evaluated() -> None:
    """An answered question with no recoverable score becomes not_evaluated, never 0.0."""
    questions = [{"id": "q1", "text": "Q1"}]
    answers = [{"question_id": "q1", "status": "submitted", "transcript": "A response"}]
    assessments = [{"question_id": "q1", "assessment_text": "Incomplete response."}]

    qa_score, _ = calculate_qa_score(questions, answers, assessments)
    result = evaluate_qa_scores(questions, answers, assessments)

    assert qa_score is None
    assert result.score is None
    assert result.not_evaluated_question_ids == ["q1"]
    assert len(result.limitations) == 1
    assert result.limitations[0].code == "qa_assessment_unavailable"
    assert "qa_quality" in result.limitations[0].affected_dimensions


def test_evaluate_qa_scores_partial_missing_excludes_unevaluated_question() -> None:
    """One unrecoverable answer score becomes not_evaluated with limitation, never 0.0."""
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
        {"question_id": "q1", "score": 0.90, "evidence_ids": ["ev_1"]},
        {"question_id": "q2", "score": 0.80, "evidence_ids": ["ev_2"]},
        {"question_id": "q3", "assessment_text": "No score present."},
    ]

    result = evaluate_qa_scores(questions, answers, assessments)

    assert result.score == 0.85
    assert result.not_evaluated_question_ids == ["q3"]
    assert len(result.limitations) == 1
    assert result.limitations[0].code == "qa_assessment_unavailable"


def test_evaluate_qa_scores_respects_explicit_not_evaluated_and_explicit_zero() -> None:
    """Explicit not_evaluated status is never scored 0.0; explicit scored 0.0 remains 0.0."""
    questions = [{"id": "q1", "text": "Q1"}]
    answers = [{"question_id": "q1", "status": "submitted", "transcript": "Ans 1"}]

    not_eval_res = evaluate_qa_scores(
        questions,
        answers,
        [{"question_id": "q1", "status": "not_evaluated", "score": 0.0}],
    )
    assert not_eval_res.score is None
    assert not_eval_res.not_evaluated_question_ids == ["q1"]

    explicit_zero_res = evaluate_qa_scores(
        questions,
        answers,
        [{"question_id": "q1", "status": "scored", "score": 0.0}],
    )
    assert explicit_zero_res.score == 0.0
    assert explicit_zero_res.not_evaluated_question_ids == []


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

    # Remaining 5 dimensions normalized with Q&A fixed at 0.20
    scored = [c for c in components if c.status == "scored"]
    assert len(scored) == 5
    assert (
        pytest.approx(sum(c.effective_weight for c in scored if c.effective_weight), 0.001) == 1.0
    )
    qa_comp = next(c for c in components if c.dimension == "qa_quality")
    assert qa_comp.effective_weight == 0.20


def test_evaluate_rubric_keeps_qa_at_20_percent_for_each_missing_modality() -> None:
    """Issue #76: Q&A effective weight remains 0.20 when optional modalities are unavailable."""
    assert WEIGHTING_POLICY_VERSION == "fixed_qa_20_v1"

    # 1. Issue #76 reproduction: video unavailable (delivery and timing None), content 0.8, Q&A 0.5
    overall_no_video, comps_no_video = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.8,
            "business_and_problem_solution_reasoning": 0.8,
            "technical_feasibility": 0.8,
            "delivery_and_body_language": None,
            "timing_and_speech_mechanics": None,
            "qa_quality": 0.5,
        }
    )
    weights_no_video = {c.dimension: c.effective_weight for c in comps_no_video}
    assert weights_no_video == {
        "pitch_content_and_evidence": 0.3333,
        "business_and_problem_solution_reasoning": 0.2667,
        "technical_feasibility": 0.20,
        "delivery_and_body_language": None,
        "timing_and_speech_mechanics": None,
        "qa_quality": 0.20,
    }
    assert overall_no_video == 0.74

    # 2. Only vision missing (delivery_and_body_language None)
    _, comps_no_vision = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.8,
            "business_and_problem_solution_reasoning": 0.8,
            "technical_feasibility": 0.8,
            "delivery_and_body_language": None,
            "timing_and_speech_mechanics": 0.8,
            "qa_quality": 0.5,
        }
    )
    weights_no_vision = {c.dimension: c.effective_weight for c in comps_no_vision}
    assert weights_no_vision == {
        "pitch_content_and_evidence": 0.3077,
        "business_and_problem_solution_reasoning": 0.2462,
        "technical_feasibility": 0.1846,
        "delivery_and_body_language": None,
        "timing_and_speech_mechanics": 0.0615,
        "qa_quality": 0.20,
    }

    # 3. Only audio features missing (timing_and_speech_mechanics None)
    _, comps_no_audio = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.8,
            "business_and_problem_solution_reasoning": 0.8,
            "technical_feasibility": 0.8,
            "delivery_and_body_language": 0.8,
            "timing_and_speech_mechanics": None,
            "qa_quality": 0.5,
        }
    )
    weights_no_audio = {c.dimension: c.effective_weight for c in comps_no_audio}
    assert weights_no_audio == {
        "pitch_content_and_evidence": 0.2667,
        "business_and_problem_solution_reasoning": 0.2133,
        "technical_feasibility": 0.16,
        "delivery_and_body_language": 0.16,
        "timing_and_speech_mechanics": None,
        "qa_quality": 0.20,
    }

    # 4. Supporting documents / technical_feasibility unavailable
    _, comps_no_docs = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.8,
            "business_and_problem_solution_reasoning": 0.8,
            "technical_feasibility": None,
            "delivery_and_body_language": 0.8,
            "timing_and_speech_mechanics": 0.8,
            "qa_quality": 0.5,
        }
    )
    weights_no_docs = {c.dimension: c.effective_weight for c in comps_no_docs}
    assert weights_no_docs == {
        "pitch_content_and_evidence": 0.3077,
        "business_and_problem_solution_reasoning": 0.2462,
        "technical_feasibility": None,
        "delivery_and_body_language": 0.1846,
        "timing_and_speech_mechanics": 0.0615,
        "qa_quality": 0.20,
    }

    # 5. Q&A unavailable: qa_quality is not_evaluated and remaining dimensions sum to 1.0
    _, comps_no_qa = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.8,
            "business_and_problem_solution_reasoning": 0.8,
            "technical_feasibility": 0.8,
            "delivery_and_body_language": None,
            "timing_and_speech_mechanics": None,
            "qa_quality": None,
        }
    )
    weights_no_qa = {c.dimension: c.effective_weight for c in comps_no_qa}
    assert weights_no_qa == {
        "pitch_content_and_evidence": 0.4167,
        "business_and_problem_solution_reasoning": 0.3333,
        "technical_feasibility": 0.25,
        "delivery_and_body_language": None,
        "timing_and_speech_mechanics": None,
        "qa_quality": None,
    }


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


@pytest.mark.parametrize(
    ("normalized_score", "expected_display", "expected_label"),
    [
        (0.39, 39, "needs_work"),
        (0.40, 40, "developing"),
        (0.59, 59, "developing"),
        (0.60, 60, "good"),
        (0.79, 79, "good"),
        (0.80, 80, "strong"),
    ],
)
def test_score_to_label_exact_boundaries(
    normalized_score: float,
    expected_display: int,
    expected_label: str,
) -> None:
    """Verify exact boundary transitions at 39/40, 59/60, and 79/80."""
    assert score_to_display(normalized_score) == expected_display
    assert score_to_label(normalized_score) == expected_label


def test_label_bands_come_from_rubric_definition_and_default_to_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify label bands default to 40/60/80 and honor per-rubric overrides."""
    default_def = get_rubric_definition("startup_pitch")
    assert default_def.label_bands == DEFAULT_LABEL_BANDS
    assert default_def.label_bands == LabelBands(developing=40, good=60, strong=80)

    # Unknown rubric IDs fall back to the default 40/60/80 bands
    assert get_rubric_definition("unknown_rubric").label_bands == LabelBands(
        developing=40, good=60, strong=80
    )
    assert score_to_label(0.62, rubric_id="unknown_rubric") == "good"
    assert score_to_label(0.82, rubric_id="unknown_rubric") == "strong"

    # Custom rubric definition overrides label bands in score_to_label, evaluate_rubric,
    # and calculate_individual_delivery_scores
    custom_rubric = RubricDefinition(
        rubric_id="strict_pitch",
        version=1,
        weights=STARTUP_PITCH_RUBRIC_V1,
        label_bands=LabelBands(developing=50, good=70, strong=90),
    )
    monkeypatch.setitem(RUBRIC_DEFINITIONS, "strict_pitch", custom_rubric)

    assert score_to_label(0.49, rubric_id="strict_pitch") == "needs_work"
    assert score_to_label(0.50, rubric_id="strict_pitch") == "developing"
    assert score_to_label(0.69, rubric_id="strict_pitch") == "developing"
    assert score_to_label(0.70, rubric_id="strict_pitch") == "good"
    assert score_to_label(0.89, rubric_id="strict_pitch") == "good"
    assert score_to_label(0.90, rubric_id="strict_pitch") == "strong"

    _, components = evaluate_rubric(
        {
            "pitch_content_and_evidence": 0.85,
            "business_and_problem_solution_reasoning": 0.65,
            "technical_feasibility": 0.45,
            "delivery_and_body_language": 0.90,
            "timing_and_speech_mechanics": 0.70,
            "qa_quality": 0.50,
        },
        rubric_id="strict_pitch",
    )
    labels_by_dim = {c.dimension: c.label for c in components}
    assert labels_by_dim == {
        "pitch_content_and_evidence": "good",
        "business_and_problem_solution_reasoning": "developing",
        "technical_feasibility": "needs_work",
        "delivery_and_body_language": "strong",
        "timing_and_speech_mechanics": "good",
        "qa_quality": "developing",
    }

    delivery_comps = calculate_individual_delivery_scores(
        {
            "speaking_time_ms": 60000,
            "windows": [
                {
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
                }
            ],
        },
        rubric_id="strict_pitch",
    )
    # delivery_norm = 0.91 -> strong under strict_pitch (>=90); timing_norm = 0.95 -> strong
    assert all(c.label == "strong" for c in delivery_comps)


def test_label_bands_validation_rejects_invalid_order() -> None:
    """Verify LabelBands rejects non-increasing or out-of-range boundaries."""
    with pytest.raises(ValidationError):
        LabelBands(developing=60, good=60, strong=80)
    with pytest.raises(ValidationError):
        LabelBands(developing=40, good=80, strong=75)
    with pytest.raises(ValidationError):
        LabelBands(developing=-1, good=60, strong=80)
    with pytest.raises(ValidationError):
        LabelBands(developing=40, good=60, strong=101)
