"""Report stage orchestration for VirtuJudge final evaluation (AI-07).

Coordinates evidence loading, deterministic rubric score calculation,
judge model evaluation for grounded team/member feedback, and markdown
report generation into a validated Evaluation model and canonical report.md.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from pydantic import BaseModel, Field
import ulid

from app.contracts import (
    ArtifactRef,
    Evaluation,
    FeedbackSection,
    Finding,
    Limitation,
    MemberFeedback,
    RubricRef,
    ScoreComponent,
    SpeakerMapping,
    SpeakingInterval,
)
from app.providers.base import JudgeModelProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.stages.aggregation import format_duration_ms
from app.stages.report_loader import ReportEvidenceBundle, load_report_evidence
from app.stages.report_markdown import generate_markdown_report
from app.stages.scoring import (
    STARTUP_PITCH_RUBRIC_V1,
    calculate_individual_delivery_scores,
    calculate_qa_score,
    evaluate_rubric,
)
from app.storage.base import ObjectStorageProtocol

logger = logging.getLogger(__name__)


class ReportStageResult(BaseModel):
    """Consolidated result returned by the report orchestration stage."""

    evaluation: Evaluation
    report_markdown: str
    member_feedback_user_ids: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)


def _format_qa_summary_text(questions: list[dict[str, Any]], answers: list[dict[str, Any]]) -> str:
    """Format compact text summary of Q&A exchanges for judge LLM prompts."""
    if not questions:
        return "No Q&A exchanges recorded."

    answers_by_qid = {a.get("question_id"): a for a in answers}
    lines: list[str] = []

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("id")
        q_text = q.get("text", "")
        q_dim = q.get("rubric_dimension", "")
        ans = answers_by_qid.get(q_id)

        if ans:
            status = ans.get("status", "submitted")
            if status == "skipped":
                ans_text = "[SKIPPED BY TEAM — Score: 0.0]"
            else:
                raw_ans = ans.get("transcript") or ans.get("answer_transcript", "")
                ans_text = raw_ans.strip() or "[EMPTY SUBMISSION]"
        else:
            ans_text = "[NO RESPONSE RECORDED]"

        lines.append(f"Question {idx} ({q_dim}): {q_text}\nAnswer: {ans_text}")

    return "\n\n".join(lines)


async def run_report_stage(
    *,
    report_id: str,
    analysis_ref: ArtifactRef,
    qa_ref: ArtifactRef,
    speaker_mappings: list[SpeakerMapping] | None = None,
    storage: ObjectStorageProtocol | None = None,
    judge_provider: JudgeModelProvider | None = None,
    rubric_id: str = "startup_pitch",
    pipeline_version: str = "0.1.0",
    session_title: str = "Startup Pitch Practice Session",
    evidence_bundle_override: ReportEvidenceBundle | None = None,
) -> ReportStageResult:
    """Execute the end-to-end evaluation and report generation stage.

    1. Loads evidence and Q&A artifacts from object storage (or synthetic defaults).
    2. Calls JudgeModelProvider to generate grounded executive summary and feedback.
    3. Calculates deterministic Q&A scores (20% rubric weight, skips=0.0) and individual delivery scores.
    4. Evaluates startup pitch rubric, computing normalized effective weights and overall score.
    5. Maps active speaking turn intervals to each mapped presenter's MemberFeedback.
    6. Formats canonical human-readable Markdown report (report.md).
    7. Validates and returns the complete Evaluation model and report text.
    """
    judge = judge_provider or FakeJudgeModelProvider()

    # 1. Load evidence bundle
    if evidence_bundle_override is not None:
        bundle = evidence_bundle_override
    else:
        bundle = await load_report_evidence(
            analysis_ref=analysis_ref,
            qa_ref=qa_ref,
            storage=storage,
        )

    # 2. Query JudgeModelProvider for structured qualitative feedback
    qa_summary_text = _format_qa_summary_text(bundle.questions, bundle.answers)
    feedback_result = await judge.generate_report_feedback(
        transcript_summary=bundle.transcript_full_text,
        qa_summary=qa_summary_text,
        speaker_profiles=bundle.by_speaker,
        rubric_id=rubric_id,
    )

    # 3. Deterministic Rubric Calculations
    # Calculate deterministic Q&A score (20% rubric weight; skipped questions score 0.0)
    qa_score, qa_evidence_ids = calculate_qa_score(
        questions=bundle.questions,
        answers=bundle.answers,
        assessments=bundle.assessments,
    )

    # Calculate average delivery scores across all active speakers for the team-level rubric components
    team_delivery_scores: list[float] = []
    team_timing_scores: list[float] = []
    for spk_prof in bundle.by_speaker.values():
        spk_comps = calculate_individual_delivery_scores(spk_prof)
        for sc in spk_comps:
            if sc.dimension == "delivery_and_body_language" and sc.status == "scored" and sc.normalized_score is not None:
                team_delivery_scores.append(sc.normalized_score)
            elif sc.dimension == "timing_and_speech_mechanics" and sc.status == "scored" and sc.normalized_score is not None:
                team_timing_scores.append(sc.normalized_score)

    team_delivery = (
        sum(team_delivery_scores) / len(team_delivery_scores)
        if team_delivery_scores
        else feedback_result.dimension_scores.get("delivery_and_body_language")
    )
    team_timing = (
        sum(team_timing_scores) / len(team_timing_scores)
        if team_timing_scores
        else feedback_result.dimension_scores.get("timing_and_speech_mechanics")
    )

    # Combine dimension scores for full rubric evaluation
    raw_dimension_scores: dict[str, float | None] = {
        "pitch_content_and_evidence": feedback_result.dimension_scores.get("pitch_content_and_evidence", 0.80),
        "business_and_problem_solution_reasoning": feedback_result.dimension_scores.get("business_and_problem_solution_reasoning", 0.75),
        "technical_feasibility": feedback_result.dimension_scores.get("technical_feasibility", 0.80),
        "delivery_and_body_language": team_delivery if team_delivery is not None else 0.75,
        "timing_and_speech_mechanics": team_timing if team_timing is not None else 0.70,
        "qa_quality": qa_score,
    }

    # Rationales
    dimension_rationales = dict(feedback_result.dimension_rationales)
    if "qa_quality" not in dimension_rationales:
        skipped_count = len([a for a in bundle.answers if a.get("status") == "skipped"])
        total_q = len(bundle.questions)
        dimension_rationales["qa_quality"] = (
            f"Deterministic evaluation across {total_q} question(s) with {skipped_count} skipped."
        )

    # Evidence mapping per dimension
    evidence_by_dim: dict[str, list[str]] = {
        "qa_quality": qa_evidence_ids,
    }
    for chunk in bundle.document_chunks:
        cid = chunk.get("chunk_id") or chunk.get("evidence_id")
        if cid:
            evidence_by_dim.setdefault("technical_feasibility", []).append(cid)
    for seg in bundle.transcript_segments:
        sid = seg.get("segment_id") or seg.get("evidence_id")
        if sid:
            evidence_by_dim.setdefault("pitch_content_and_evidence", []).append(sid)

    overall_score, rubric_components = evaluate_rubric(
        raw_scores=raw_dimension_scores,
        rubric_id=rubric_id,
        evidence_by_dimension=evidence_by_dim,
        rationales=dimension_rationales,
        limitations=bundle.limitations,
    )

    # 4. Map Individual Presenter MemberFeedback
    member_feedbacks: list[MemberFeedback] = []
    member_user_ids: list[str] = []

    # Build speaker mapping index
    mappings_to_process: list[tuple[str, str, str]] = []
    if speaker_mappings:
        for sm in speaker_mappings:
            spk_label = sm.speaker_label
            user_id = sm.user_id
            display_name = sm.display_name or f"Presenter ({spk_label})"
            mappings_to_process.append((spk_label, user_id, display_name))
    else:
        # Fallback to speakers found in by_speaker or speaker_labels
        speaker_keys = list(bundle.by_speaker.keys()) or bundle.speaker_labels or ["SPEAKER_00"]
        for spk_label in sorted(speaker_keys):
            mappings_to_process.append(
                (spk_label, f"user_{spk_label.lower()}", f"Presenter ({spk_label})")
            )

    for spk_label, user_id, display_name in mappings_to_process:
        member_user_ids.append(user_id)
        spk_profile = bundle.by_speaker.get(spk_label, {})

        # Extract speaking intervals
        raw_intervals = spk_profile.get("intervals", [])
        speaking_intervals: list[SpeakingInterval] = []
        for item in raw_intervals:
            if isinstance(item, SpeakingInterval):
                speaking_intervals.append(item)
            elif isinstance(item, dict):
                start_ms = item.get("start_ms", 0)
                end_ms = item.get("end_ms", 0)
                fmt = item.get("formatted") or f"{start_ms//60000:02d}:{(start_ms%60000)//1000:02d} - {end_ms//60000:02d}:{(end_ms%60000)//1000:02d}"
                speaking_intervals.append(
                    SpeakingInterval(start_ms=start_ms, end_ms=end_ms, formatted=fmt)
                )

        speaking_time_ms = spk_profile.get("speaking_time_ms", 0)
        delivery_components = calculate_individual_delivery_scores(spk_profile)

        # Retrieve member findings from LLM feedback
        mbr_strengths = list(feedback_result.member_strengths.get(spk_label, []))
        mbr_improvements = list(feedback_result.member_improvements.get(spk_label, []))

        # Default fallbacks if LLM omitted findings for this presenter
        if not mbr_strengths:
            mbr_strengths = [
                Finding(
                    id=f"f_{spk_label.lower()}_s1",
                    kind="strength",
                    title="Engaged Vocal Presence",
                    detail=f"{display_name} maintained clear vocal delivery throughout their presentation turns.",
                    rubric_dimension="delivery_and_body_language",
                    speaker_labels=[spk_label],
                )
            ]
        if not mbr_improvements:
            mbr_improvements = [
                Finding(
                    id=f"f_{spk_label.lower()}_i1",
                    kind="improvement",
                    title="Slide Transition Calibration",
                    detail=f"{display_name} can use deliberate pauses when introducing new slides.",
                    recommendation="Insert a 1-2 second pause before transitioning to new sections.",
                    rubric_dimension="timing_and_speech_mechanics",
                    speaker_labels=[spk_label],
                )
            ]

        time_display = format_duration_ms(speaking_time_ms)
        summary = (
            f"Active presentation delivery analysis for {display_name} across "
            f"{len(speaking_intervals)} turn(s) ({time_display} total)."
        )

        member_feedbacks.append(
            MemberFeedback(
                user_id=user_id,
                display_name=display_name,
                speaker_labels=[spk_label],
                speaking_intervals=speaking_intervals,
                speaking_time_ms=speaking_time_ms,
                summary=summary,
                strengths=mbr_strengths,
                improvements=mbr_improvements,
                delivery_components=delivery_components,
            )
        )

    # 5. Whole-Team Feedback Section & Consolidated Findings
    team_feedback = FeedbackSection(
        summary=feedback_result.executive_summary,
        strengths=feedback_result.team_strengths,
        improvements=feedback_result.team_improvements,
        score_components=rubric_components,
        limitations=bundle.limitations,
    )

    all_findings: list[Finding] = list(feedback_result.team_strengths) + list(
        feedback_result.team_improvements
    )
    for mf in member_feedbacks:
        all_findings.extend(mf.strengths)
        all_findings.extend(mf.improvements)

    # 6. Construct Evaluation Model
    eval_id = ulid.new().str
    session_id = bundle.session_id or report_id
    qa_round_id = bundle.metadata.get("qa_round_id") or f"{report_id}:qa_round"

    reproducibility: dict[str, Any] = {
        "pipeline_version": pipeline_version,
        "rubric_id": rubric_id,
        "scoring_engine": "startup_pitch_v1",
        "evaluator_provider": type(judge).__name__,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    evaluation = Evaluation(
        id=eval_id,
        analysis_attempt_id=session_id,
        qa_round_id=qa_round_id,
        rubric=RubricRef(rubric_id=rubric_id, version=1),
        overall_score=overall_score,
        components=rubric_components,
        findings=all_findings,
        team_feedback=team_feedback,
        member_feedback=member_feedbacks,
        limitations=bundle.limitations,
        reproducibility=reproducibility,
    )

    # 7. Generate Canonical Markdown Report
    qa_data = {
        "questions": bundle.questions,
        "answers": bundle.answers,
        "assessments": bundle.assessments,
    }
    report_markdown = generate_markdown_report(
        evaluation,
        session_title=session_title,
        qa_data=qa_data,
        by_speaker=bundle.by_speaker,
    )

    return ReportStageResult(
        evaluation=evaluation,
        report_markdown=report_markdown,
        member_feedback_user_ids=member_user_ids,
        limitations=bundle.limitations,
    )


__all__ = [
    "ReportStageResult",
    "run_report_stage",
]
