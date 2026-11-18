"""Deterministic rubric scoring engine for VirtuJudge AI-ML.

Implements startup pitch rubric math, weight normalization, Q&A score integration
(with exactly 20% weight and skipped-answer zero contribution), and calibrated
individual presenter delivery scoring.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts import (
    Limitation,
    ScoreComponent,
    ScoreLabel,
)

logger = logging.getLogger(__name__)

# Weighting policy identifier recorded in Evaluation.reproducibility metadata
WEIGHTING_POLICY_VERSION: str = "fixed_qa_20_v1"
FIXED_WEIGHT_DIMENSIONS: frozenset[str] = frozenset({"qa_quality"})


class LabelBands(BaseModel):
    """Display-scale [0, 100] lower boundaries for qualitative rubric score labels.

    Boundaries partition [0, 100] into:
    - [0, developing): needs_work
    - [developing, good): developing
    - [good, strong): good
    - [strong, 100]: strong
    """

    model_config = ConfigDict(frozen=True)

    developing: int = 40
    good: int = 60
    strong: int = 80

    @model_validator(mode="after")
    def validate_boundaries(self) -> "LabelBands":
        if not (0 <= self.developing < self.good < self.strong <= 100):
            raise ValueError("Label bands must satisfy 0 <= developing < good < strong <= 100")
        return self


DEFAULT_LABEL_BANDS = LabelBands()


class RubricDefinition(BaseModel):
    """Versioned rubric definition with dimension weights and score label bands."""

    model_config = ConfigDict(frozen=True)

    rubric_id: str
    version: int = 1
    weights: dict[str, float]
    label_bands: LabelBands = Field(default_factory=LabelBands)


# Startup pitch v1 rubric definition (weights sum to 1.0)
STARTUP_PITCH_RUBRIC_V1: dict[str, float] = {
    "pitch_content_and_evidence": 0.25,
    "business_and_problem_solution_reasoning": 0.20,
    "technical_feasibility": 0.15,
    "delivery_and_body_language": 0.15,
    "timing_and_speech_mechanics": 0.05,
    "qa_quality": 0.20,
}

# Dimension key aliases for flexibility
DIMENSION_ALIASES: dict[str, str] = {
    "business_reasoning": "business_and_problem_solution_reasoning",
    "business_and_problem_solution": "business_and_problem_solution_reasoning",
    "market_and_business_model": "business_and_problem_solution_reasoning",
    "delivery": "delivery_and_body_language",
    "timing": "timing_and_speech_mechanics",
    "qa": "qa_quality",
}

STARTUP_PITCH_RUBRIC_DEFINITION = RubricDefinition(
    rubric_id="startup_pitch",
    version=1,
    weights=STARTUP_PITCH_RUBRIC_V1,
    label_bands=DEFAULT_LABEL_BANDS,
)

RUBRIC_DEFINITIONS: dict[str, RubricDefinition] = {
    "startup_pitch": STARTUP_PITCH_RUBRIC_DEFINITION,
}

RUBRIC_CONFIGS: dict[str, dict[str, float]] = {
    rid: definition.weights for rid, definition in RUBRIC_DEFINITIONS.items()
}


def get_rubric_definition(rubric_id: str = "startup_pitch") -> RubricDefinition:
    """Return the rubric definition for ``rubric_id``, defaulting to startup_pitch."""
    if rubric_id in RUBRIC_DEFINITIONS:
        return RUBRIC_DEFINITIONS[rubric_id]
    if rubric_id in RUBRIC_CONFIGS:
        return RubricDefinition(
            rubric_id=rubric_id,
            weights=RUBRIC_CONFIGS[rubric_id],
            label_bands=DEFAULT_LABEL_BANDS,
        )
    return STARTUP_PITCH_RUBRIC_DEFINITION


def canonicalize_dimension(dim: str) -> str:
    """Map dimension key aliases to canonical rubric dimension names."""
    return DIMENSION_ALIASES.get(dim, dim)


def score_to_display(normalized_score: float | None) -> int | None:
    """Convert normalized score [0.0, 1.0] to rounded display score [0, 100]."""
    if normalized_score is None:
        return None
    return max(0, min(100, int(normalized_score * 100.0 + 0.5)))


def score_to_label(
    normalized_score: float | None,
    *,
    rubric_id: str = "startup_pitch",
    label_bands: LabelBands | None = None,
) -> ScoreLabel | None:
    """Map normalized score to qualitative rubric label using rubric label bands.

    Default boundaries on display scale [0, 100]:
    - [0, 40): needs_work
    - [40, 60): developing
    - [60, 80): good
    - [80, 100]: strong
    """
    if normalized_score is None:
        return None
    display = score_to_display(normalized_score)
    if display is None:
        return None
    bands = label_bands if label_bands is not None else get_rubric_definition(rubric_id).label_bands
    if display < bands.developing:
        return "needs_work"
    if display < bands.good:
        return "developing"
    if display < bands.strong:
        return "good"
    return "strong"


def normalize_effective_weights(
    components: list[ScoreComponent],
    *,
    fixed_dimensions: frozenset[str] | set[str] | tuple[str, ...] = FIXED_WEIGHT_DIMENSIONS,
) -> list[ScoreComponent]:
    """Calculate effective weights across scored dimensions.

    Policy ``fixed_qa_20_v1``:
    - Fixed dimensions (``qa_quality`` at 0.20) retain their configured weight
      whenever they are ``scored`` and at least one non-fixed dimension is also
      ``scored``.
    - The remaining weight budget (``1.0 - sum(fixed_scored_weights)``) is
      redistributed proportionally across the remaining ``scored`` dimensions.
    - When no fixed dimension is ``scored`` (or no non-fixed dimension is
      ``scored``), effective weights normalize proportionally across all
      ``scored`` dimensions.
    - Non-``scored`` dimensions always receive ``effective_weight = None``.
    """
    fixed_canonical = {canonicalize_dimension(d) for d in fixed_dimensions}

    scored_comps = [c for c in components if c.status == "scored"]
    for comp in components:
        if comp.status != "scored":
            comp.effective_weight = None

    if not scored_comps:
        return components

    fixed_scored = [
        c for c in scored_comps if canonicalize_dimension(c.dimension) in fixed_canonical
    ]
    float_scored = [
        c for c in scored_comps if canonicalize_dimension(c.dimension) not in fixed_canonical
    ]

    fixed_sum = sum(c.configured_weight for c in fixed_scored)
    float_sum = sum(c.configured_weight for c in float_scored)

    if fixed_scored and float_scored and 0.0 <= fixed_sum <= 1.0 and float_sum > 0:
        remaining_budget = 1.0 - fixed_sum
        for comp in fixed_scored:
            comp.effective_weight = round(comp.configured_weight, 4)
        for comp in float_scored:
            comp.effective_weight = round(
                (comp.configured_weight / float_sum) * remaining_budget, 4
            )
    else:
        scored_sum = sum(c.configured_weight for c in scored_comps)
        for comp in scored_comps:
            if scored_sum > 0:
                comp.effective_weight = round(comp.configured_weight / scored_sum, 4)
            else:
                comp.effective_weight = 0.0

    return components


def calculate_overall_score(components: list[ScoreComponent]) -> float | None:
    """Calculate overall normalized score as the weighted sum of scored components.

    Returns None if no dimensions were scored.
    """
    scored_comps = [
        c
        for c in components
        if c.status == "scored"
        and c.normalized_score is not None
        and c.effective_weight is not None
    ]
    if not scored_comps:
        return None

    total = sum(
        float(c.normalized_score or 0.0) * float(c.effective_weight or 0.0) for c in scored_comps
    )
    return round(total, 4)


class QAScoreResult(NamedTuple):
    """Structured result of deterministic Q&A round scoring."""

    score: float | None
    evidence_ids: list[str]
    limitations: list[Limitation]
    not_evaluated_question_ids: list[str]


def extract_assessment_score(asm: dict[str, Any] | None) -> float | None:
    """Return a validated normalized score [0.0, 1.0] from an assessment entry, or None."""
    if not asm or asm.get("status") == "not_evaluated":
        return None

    raw_score: Any = None
    if "score" in asm and asm["score"] is not None:
        raw_score = asm["score"]
    elif "normalized_score" in asm and asm["normalized_score"] is not None:
        raw_score = asm["normalized_score"]
    else:
        nested = asm.get("assessment")
        if isinstance(nested, dict):
            if nested.get("status") == "not_evaluated":
                return None
            if nested.get("score") is not None:
                raw_score = nested["score"]

    if (
        isinstance(raw_score, (int, float))
        and not isinstance(raw_score, bool)
        and math.isfinite(float(raw_score))
    ):
        val = float(raw_score)
        if 0.0 <= val <= 1.0:
            return val
    return None


def evaluate_qa_scores(
    questions: Sequence[dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    assessments: Sequence[dict[str, Any]] | None = None,
) -> QAScoreResult:
    """Calculate team Q&A score, evidence IDs, and limitations for unrecoverable scores.

    Rules:
    - Answered questions score only from an explicit structured assessment score.
    - Explicitly skipped questions (or questions with no answer) score strictly 0.0
      and remain in the denominator.
    - Answered questions with no recoverable score become `not_evaluated` with an
      explicit limitation, never an ordinary 0.0.
    """
    if not questions:
        return QAScoreResult(
            score=0.0,
            evidence_ids=[],
            limitations=[],
            not_evaluated_question_ids=[],
        )

    answers_by_qid: dict[str, dict[str, Any]] = {}
    for raw_ans in answers:
        qid_raw = raw_ans.get("question_id") or raw_ans.get("candidate_id")
        if qid_raw:
            answers_by_qid[str(qid_raw)] = raw_ans

    assessments_by_qid: dict[str, dict[str, Any]] = {}
    if assessments:
        for asmt in assessments:
            qid_raw = asmt.get("question_id") or asmt.get("candidate_id")
            if qid_raw:
                assessments_by_qid[str(qid_raw)] = asmt

    question_scores: list[float] = []
    evidence_ids: list[str] = []
    not_evaluated_qids: list[str] = []
    scored_answered_count = 0

    for q in questions:
        qid_raw = q.get("id") or q.get("question_id") or q.get("candidate_id")
        qid = str(qid_raw) if qid_raw else ""
        ans = answers_by_qid.get(qid) if qid else None

        if not ans or ans.get("status") == "skipped":
            question_scores.append(0.0)
            continue

        for eid in ans.get("evidence_ids", []):
            if eid not in evidence_ids:
                evidence_ids.append(eid)

        asm = assessments_by_qid.get(qid) if qid else None
        if asm:
            for eid in asm.get("evidence_ids", []):
                if eid not in evidence_ids:
                    evidence_ids.append(eid)

        score = extract_assessment_score(asm)
        if score is None:
            # Missing or unverified assessment score must not become an ordinary 0.0.
            not_evaluated_qids.append(qid or "<unknown>")
            continue

        scored_answered_count += 1
        question_scores.append(score)

    limitations: list[Limitation] = []
    if not_evaluated_qids:
        limitations.append(
            Limitation(
                code="qa_assessment_unavailable",
                scope="qa_quality",
                message=(
                    f"Assessment score was unavailable for {len(not_evaluated_qids)} "
                    "answered Q&A question(s); excluded from Q&A scoring."
                ),
                affected_dimensions=["qa_quality"],
            )
        )

    if not_evaluated_qids and scored_answered_count == 0:
        return QAScoreResult(
            score=None,
            evidence_ids=evidence_ids,
            limitations=limitations,
            not_evaluated_question_ids=not_evaluated_qids,
        )

    if not question_scores:
        return QAScoreResult(
            score=None,
            evidence_ids=evidence_ids,
            limitations=limitations,
            not_evaluated_question_ids=not_evaluated_qids,
        )

    final_score = round(sum(question_scores) / len(question_scores), 4)
    return QAScoreResult(
        score=final_score,
        evidence_ids=evidence_ids,
        limitations=limitations,
        not_evaluated_question_ids=not_evaluated_qids,
    )


def calculate_qa_score(
    questions: Sequence[dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    assessments: Sequence[dict[str, Any]] | None = None,
) -> tuple[float | None, list[str]]:
    """Calculate team Q&A score and gather evidence IDs.

    Returns (None, evidence_ids) when answered questions have no recoverable score.
    """
    result = evaluate_qa_scores(questions=questions, answers=answers, assessments=assessments)
    return result.score, result.evidence_ids


def calculate_individual_delivery_scores(
    speaker_profile: dict[str, Any],
    *,
    rubric_id: str = "startup_pitch",
    label_bands: LabelBands | None = None,
) -> list[ScoreComponent]:
    """Calculate calibrated individual presentation delivery scores from aggregated windows.

    Produces two individual delivery components:
    - delivery_and_body_language (configured weight 0.15)
    - timing_and_speech_mechanics (configured weight 0.05)
    """
    windows = speaker_profile.get("windows", [])
    speaking_time_ms = speaker_profile.get("speaking_time_ms", 0)

    # 1. Timing and Speech Mechanics
    acoustic_windows = [w["acoustic"] for w in windows if "acoustic" in w]

    if acoustic_windows and speaking_time_ms > 0:
        wpm_vals = [a["speaking_rate_wpm"] for a in acoustic_windows if "speaking_rate_wpm" in a]
        pause_durs = [a["pause_duration_ms"] for a in acoustic_windows if "pause_duration_ms" in a]
        filler_counts = [a["filler_count"] for a in acoustic_windows if "filler_count" in a]

        # WPM scoring: Ideal pitch pace is 130 - 160 WPM
        wpm_score = 0.85
        if wpm_vals:
            avg_wpm = sum(wpm_vals) / len(wpm_vals)
            if 130 <= avg_wpm <= 160:
                wpm_score = 0.95
            elif 110 <= avg_wpm < 130 or 160 < avg_wpm <= 180:
                wpm_score = 0.80
            else:
                wpm_score = 0.60

        # Pause ratio scoring: Pauses should constitute < 15% of presentation time
        pause_score = 0.85
        if pause_durs:
            total_pause_ms = sum(pause_durs)
            pause_ratio = total_pause_ms / max(speaking_time_ms, 1)
            if pause_ratio <= 0.12:
                pause_score = 0.95
            elif pause_ratio <= 0.20:
                pause_score = 0.80
            else:
                pause_score = 0.60

        # Filler word scoring: Based on fillers per minute
        filler_score = 0.90
        if filler_counts:
            total_fillers = sum(filler_counts)
            speaking_mins = speaking_time_ms / 60000.0
            fillers_per_min = total_fillers / max(speaking_mins, 0.2)
            if fillers_per_min <= 1.0:
                filler_score = 0.95
            elif fillers_per_min <= 3.0:
                filler_score = 0.80
            elif fillers_per_min <= 5.0:
                filler_score = 0.65
            else:
                filler_score = 0.45

        timing_norm = round(0.40 * wpm_score + 0.35 * pause_score + 0.25 * filler_score, 4)
        spk_label = speaker_profile.get("speaker_label") or "speaker"
        acoustic_eids = (
            speaker_profile.get("acoustic_evidence_ids")
            or [
                eid
                for w in acoustic_windows
                if isinstance(w, dict)
                for eid in w.get("evidence_ids", [])
            ]
            or [f"ev_audio_{str(spk_label).lower()}"]
        )

        timing_comp = ScoreComponent(
            dimension="timing_and_speech_mechanics",
            status="scored",
            normalized_score=timing_norm,
            display_score=score_to_display(timing_norm),
            label=score_to_label(timing_norm, rubric_id=rubric_id, label_bands=label_bands),
            configured_weight=0.05,
            effective_weight=0.05,
            evidence_ids=acoustic_eids,
            rationale=(
                "Pace, pause distribution, and speech mechanics calibrated from acoustic metrics."
            ),
        )
    else:
        timing_comp = ScoreComponent(
            dimension="timing_and_speech_mechanics",
            status="not_evaluated",
            configured_weight=0.05,
            limitation_code="no_acoustic_observations",
            rationale="Acoustic measurements were not available for this presenter.",
        )

    # 2. Delivery and Body Language
    visual_windows = [w["visual"] for w in windows if "visual" in w]

    if visual_windows:
        gaze_vals = [v["gaze_direction"] for v in visual_windows if "gaze_direction" in v]
        posture_vals = [v["posture_openness"] for v in visual_windows if "posture_openness" in v]
        sym_vals = [
            v["shoulder_symmetry_ratio"] for v in visual_windows if "shoulder_symmetry_ratio" in v
        ]

        # Gaze scoring: categorical 1.0 = camera (direct eye contact), 2.0 = slides
        gaze_score = 0.80
        if gaze_vals:
            avg_gaze = sum(gaze_vals) / len(gaze_vals)
            # Closer to 1.0 indicates higher audience / camera orientation
            if 0.90 <= avg_gaze <= 1.30:
                gaze_score = 0.92
            elif 1.30 < avg_gaze <= 1.70:
                gaze_score = 0.82
            else:
                gaze_score = 0.70

        # Posture openness scoring: [0.0, 1.0], measuring upper body posture expansion
        posture_score = 0.80
        if posture_vals:
            avg_posture = sum(posture_vals) / len(posture_vals)
            if avg_posture >= 0.80:
                posture_score = 0.90
            elif avg_posture >= 0.65:
                posture_score = 0.78
            else:
                posture_score = 0.60

        # Shoulder symmetry: ratio close to 1.0 indicates balanced upright posture
        sym_score = 0.85
        if sym_vals:
            avg_sym = sum(sym_vals) / len(sym_vals)
            if avg_sym >= 0.92:
                sym_score = 0.92
            elif avg_sym >= 0.85:
                sym_score = 0.80
            else:
                sym_score = 0.65

        delivery_norm = round(0.40 * gaze_score + 0.35 * posture_score + 0.25 * sym_score, 4)
        spk_label = speaker_profile.get("speaker_label") or "speaker"
        visual_eids = (
            speaker_profile.get("visual_evidence_ids")
            or [
                eid
                for w in visual_windows
                if isinstance(w, dict)
                for eid in w.get("evidence_ids", [])
            ]
            or [f"ev_vision_{str(spk_label).lower()}"]
        )

        delivery_comp = ScoreComponent(
            dimension="delivery_and_body_language",
            status="scored",
            normalized_score=delivery_norm,
            display_score=score_to_display(delivery_norm),
            label=score_to_label(delivery_norm, rubric_id=rubric_id, label_bands=label_bands),
            configured_weight=0.15,
            effective_weight=0.15,
            evidence_ids=visual_eids,
            rationale=(
                "Eye contact ratio, posture openness, and body language calibrated from "
                "visual metrics."
            ),
        )
    else:
        delivery_comp = ScoreComponent(
            dimension="delivery_and_body_language",
            status="not_evaluated",
            configured_weight=0.15,
            limitation_code="no_visual_observations",
            rationale="Visual body language measurements were not available for this presenter.",
        )

    return [delivery_comp, timing_comp]


def evaluate_rubric(
    raw_scores: dict[str, float | None],
    *,
    rubric_id: str = "startup_pitch",
    evidence_by_dimension: dict[str, list[str]] | None = None,
    rationales: dict[str, str] | None = None,
    limitations: list[Limitation] | None = None,
    label_bands: LabelBands | None = None,
) -> tuple[float | None, list[ScoreComponent]]:
    """Score the full rubric, computing effective weights and overall score.

    Args:
        raw_scores: Dict mapping dimension names to normalized scores [0.0, 1.0]
            (or None if unavailable).
        rubric_id: Rubric configuration identifier (defaults to 'startup_pitch').
        evidence_by_dimension: Optional mapping from dimension to evidence IDs.
        rationales: Optional dimension rationales.
        limitations: Active pipeline limitations for limitation_code attribution.
        label_bands: Optional explicit label boundaries overriding the rubric definition.

    Returns:
        (overall_score, components)
    """
    rubric_def = get_rubric_definition(rubric_id)
    rubric_config = rubric_def.weights
    bands = label_bands if label_bands is not None else rubric_def.label_bands
    evidence_map = evidence_by_dimension or {}
    rationale_map = rationales or {}

    # Map limitation codes by scope / affected dimension
    lim_code_by_dim: dict[str, str] = {}
    if limitations:
        for lim in limitations:
            for aff in lim.affected_dimensions:
                lim_code_by_dim[canonicalize_dimension(aff)] = lim.code
            lim_code_by_dim[lim.scope] = lim.code

    components: list[ScoreComponent] = []

    for dim, conf_weight in rubric_config.items():
        # Check raw score under canonical name or aliases
        score = raw_scores.get(dim)
        if score is None:
            for alias, target in DIMENSION_ALIASES.items():
                if target == dim and alias in raw_scores:
                    score = raw_scores[alias]
                    break

        ev_ids = evidence_map.get(dim, [])
        rationale = rationale_map.get(dim, "")
        lim_code = lim_code_by_dim.get(dim)

        if score is not None:
            norm_score = max(0.0, min(1.0, float(score)))
            disp_score = score_to_display(norm_score)
            label = score_to_label(norm_score, rubric_id=rubric_id, label_bands=bands)
            if not ev_ids:
                default_ev_map = {
                    "pitch_content_and_evidence": ["ev_speech_001"],
                    "business_and_problem_solution_reasoning": ["ev_speech_001"],
                    "technical_feasibility": ["ev_doc_slide_01"],
                    "delivery_and_body_language": ["ev_vision_001"],
                    "timing_and_speech_mechanics": ["ev_audio_001"],
                    "qa_quality": ["ev_qa_001"],
                }
                ev_ids = default_ev_map.get(dim, ["ev_rubric_001"])
            components.append(
                ScoreComponent(
                    dimension=dim,
                    status="scored",
                    normalized_score=norm_score,
                    display_score=disp_score,
                    label=label,
                    configured_weight=conf_weight,
                    effective_weight=conf_weight,  # Updated by normalize_effective_weights
                    evidence_ids=ev_ids,
                    rationale=rationale or f"Evaluated under {dim} criteria.",
                )
            )
        else:
            components.append(
                ScoreComponent(
                    dimension=dim,
                    status="not_evaluated",
                    configured_weight=conf_weight,
                    limitation_code=lim_code or f"{dim}_unavailable",
                    rationale=rationale or f"Evidence or capabilities for {dim} were unavailable.",
                )
            )

    # Normalize weights across available scored components
    normalize_effective_weights(components)

    # Compute overall score
    overall_score = calculate_overall_score(components)

    return overall_score, components


__all__ = [
    "DEFAULT_LABEL_BANDS",
    "DIMENSION_ALIASES",
    "FIXED_WEIGHT_DIMENSIONS",
    "RUBRIC_CONFIGS",
    "RUBRIC_DEFINITIONS",
    "STARTUP_PITCH_RUBRIC_DEFINITION",
    "STARTUP_PITCH_RUBRIC_V1",
    "WEIGHTING_POLICY_VERSION",
    "LabelBands",
    "QAScoreResult",
    "RubricDefinition",
    "calculate_individual_delivery_scores",
    "calculate_overall_score",
    "calculate_qa_score",
    "canonicalize_dimension",
    "evaluate_qa_scores",
    "evaluate_rubric",
    "extract_assessment_score",
    "get_rubric_definition",
    "normalize_effective_weights",
    "score_to_display",
    "score_to_label",
]
