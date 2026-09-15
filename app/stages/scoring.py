"""Deterministic rubric scoring engine for VirtuJudge AI-ML.

Implements startup pitch rubric math, weight normalization, Q&A score integration
(with exactly 20% weight and skipped-answer zero contribution), and calibrated
individual presenter delivery scoring.
"""

import logging
from collections.abc import Sequence
from typing import Any

from app.contracts import (
    Limitation,
    ScoreComponent,
    ScoreLabel,
)

logger = logging.getLogger(__name__)

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

RUBRIC_CONFIGS: dict[str, dict[str, float]] = {
    "startup_pitch": STARTUP_PITCH_RUBRIC_V1,
}


def canonicalize_dimension(dim: str) -> str:
    """Map dimension key aliases to canonical rubric dimension names."""
    return DIMENSION_ALIASES.get(dim, dim)


def score_to_display(normalized_score: float | None) -> int | None:
    """Convert normalized score [0.0, 1.0] to rounded display score [0, 100]."""
    if normalized_score is None:
        return None
    return max(0, min(100, int(normalized_score * 100.0 + 0.5)))


def score_to_label(normalized_score: float | None) -> ScoreLabel | None:
    """Map normalized score to qualitative rubric label.

    Boundaries on display scale [0, 100]:
    - [0, 45): needs_work
    - [45, 65): developing
    - [65, 85): good
    - [85, 100]: strong (reserved for exceptional, well-substantiated work)
    """
    if normalized_score is None:
        return None
    display = score_to_display(normalized_score)
    if display is None:
        return None
    if display < 45:
        return "needs_work"
    if display < 65:
        return "developing"
    if display < 85:
        return "good"
    return "strong"


def normalize_effective_weights(
    components: list[ScoreComponent],
) -> list[ScoreComponent]:
    """Calculate effective weights across scored dimensions.

    When one or more dimensions cannot be evaluated (e.g. video unavailable),
    effective weights are normalized only across available scored dimensions:
        effective_weight_i = configured_weight_i / sum(configured_weights of scored dimensions)
    """
    scored_sum = sum(
        c.configured_weight
        for c in components
        if c.status == "scored"
    )

    for comp in components:
        if comp.status == "scored":
            if scored_sum > 0:
                comp.effective_weight = round(comp.configured_weight / scored_sum, 4)
            else:
                comp.effective_weight = 0.0
        else:
            comp.effective_weight = None

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
        float(c.normalized_score or 0.0) * float(c.effective_weight or 0.0)
        for c in scored_comps
    )
    return round(total, 4)


def calculate_qa_score(
    questions: Sequence[dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    assessments: Sequence[dict[str, Any]] | None = None,
) -> tuple[float, list[str]]:
    """Calculate team Q&A score and gather evidence IDs.

    Rules:
    - Q&A contributes to the team evaluation.
    - Answered questions score only from an explicit structured assessment score.
    - Skipped questions score strictly 0.0.
    - Skipped questions do NOT remove the question from the denominator or drop Q&A weight.
    """
    if not questions:
        return 0.0, []

    # Map answers by question_id
    answers_by_qid: dict[str, dict[str, Any]] = {}
    for a in answers:
        qid = a.get("question_id")
        if qid:
            answers_by_qid[qid] = a

    # Map assessments by question_id if available
    assessments_by_qid: dict[str, dict[str, Any]] = {}
    if assessments:
        for asmt in assessments:
            qid = asmt.get("question_id")
            if qid:
                assessments_by_qid[qid] = asmt

    question_scores: list[float] = []
    evidence_ids: list[str] = []

    for q in questions:
        qid_raw = q.get("id") or q.get("question_id") or q.get("candidate_id")
        qid = str(qid_raw) if qid_raw else ""
        ans = answers_by_qid.get(qid) if qid else None

        if not ans or ans.get("status") == "skipped":
            # Skipped answer contributes 0.0
            question_scores.append(0.0)
            continue

        # Check for explicit assessment score
        score = None
        asm = assessments_by_qid.get(qid) if qid else None
        if asm:
            if "score" in asm and asm["score"] is not None:
                score = float(asm["score"])
            elif "normalized_score" in asm and asm["normalized_score"] is not None:
                score = float(asm["normalized_score"])

            # Extract evidence
            e_ids = asm.get("evidence_ids", [])
            for eid in e_ids:
                if eid not in evidence_ids:
                    evidence_ids.append(eid)

        if score is None:
            # A transcript proves that the candidate spoke, not that the answer met the
            # rubric. Never invent credit when an assessment artifact lacks a score.
            score = 0.0

        question_scores.append(max(0.0, min(1.0, score)))

        # Also collect answer evidence if present
        ans_eids = ans.get("evidence_ids", [])
        for eid in ans_eids:
            if eid not in evidence_ids:
                evidence_ids.append(eid)

    total_q = len(question_scores)
    final_score = sum(question_scores) / total_q if total_q > 0 else 0.0
    return round(final_score, 4), evidence_ids


def calculate_individual_delivery_scores(
    speaker_profile: dict[str, Any],
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
        acoustic_eids = speaker_profile.get("acoustic_evidence_ids") or [
            eid
            for w in acoustic_windows
            if isinstance(w, dict)
            for eid in w.get("evidence_ids", [])
        ] or [f"ev_audio_{str(spk_label).lower()}"]

        timing_comp = ScoreComponent(
            dimension="timing_and_speech_mechanics",
            status="scored",
            normalized_score=timing_norm,
            display_score=score_to_display(timing_norm),
            label=score_to_label(timing_norm),
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
            v["shoulder_symmetry_ratio"]
            for v in visual_windows
            if "shoulder_symmetry_ratio" in v
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
        visual_eids = speaker_profile.get("visual_evidence_ids") or [
            eid
            for w in visual_windows
            if isinstance(w, dict)
            for eid in w.get("evidence_ids", [])
        ] or [f"ev_vision_{str(spk_label).lower()}"]

        delivery_comp = ScoreComponent(
            dimension="delivery_and_body_language",
            status="scored",
            normalized_score=delivery_norm,
            display_score=score_to_display(delivery_norm),
            label=score_to_label(delivery_norm),
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
) -> tuple[float | None, list[ScoreComponent]]:
    """Score the full rubric, computing effective weights and overall score.

    Args:
        raw_scores: Dict mapping dimension names to normalized scores [0.0, 1.0]
            (or None if unavailable).
        rubric_id: Rubric configuration identifier (defaults to 'startup_pitch').
        evidence_by_dimension: Optional mapping from dimension to evidence IDs.
        rationales: Optional dimension rationales.
        limitations: Active pipeline limitations for limitation_code attribution.

    Returns:
        (overall_score, components)
    """
    rubric_config = RUBRIC_CONFIGS.get(rubric_id, STARTUP_PITCH_RUBRIC_V1)
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
            label = score_to_label(norm_score)
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
    "DIMENSION_ALIASES",
    "RUBRIC_CONFIGS",
    "STARTUP_PITCH_RUBRIC_V1",
    "calculate_individual_delivery_scores",
    "calculate_overall_score",
    "calculate_qa_score",
    "canonicalize_dimension",
    "evaluate_rubric",
    "normalize_effective_weights",
    "score_to_display",
    "score_to_label",
]
