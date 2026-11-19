"""Report stage orchestration for VirtuJudge final evaluation (AI-07).

Coordinates evidence loading, deterministic rubric score calculation,
judge model evaluation for grounded team/member feedback, and markdown
report generation into a validated Evaluation model and canonical report.md.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

import ulid
from pydantic import BaseModel, Field

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
from app.runtime_mode import RuntimeModePolicy
from app.stages.aggregation import format_duration_ms
from app.stages.common import append_unique_limitation
from app.stages.report_loader import ReportEvidenceBundle, load_report_evidence
from app.stages.report_markdown import generate_markdown_report
from app.stages.scoring import (
    WEIGHTING_POLICY_VERSION,
    calculate_individual_delivery_scores,
    evaluate_qa_scores,
    evaluate_rubric,
    extract_assessment_score,
    score_to_display,
    score_to_label,
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


def _group_speaker_mappings(
    speaker_mappings: list[SpeakerMapping],
) -> list[tuple[str, str, list[str]]]:
    """Group speaker mappings by user_id while preserving order and deduplicating labels."""
    grouped: dict[str, dict[str, Any]] = {}
    for sm in speaker_mappings:
        entry = grouped.get(sm.user_id)
        if entry is None:
            grouped[sm.user_id] = {
                "display_name": (sm.display_name or "").strip(),
                "speaker_labels": [sm.speaker_label],
            }
        else:
            if sm.speaker_label not in entry["speaker_labels"]:
                entry["speaker_labels"].append(sm.speaker_label)
            if not entry["display_name"] and sm.display_name and sm.display_name.strip():
                entry["display_name"] = sm.display_name.strip()

    result: list[tuple[str, str, list[str]]] = []
    for user_id, entry in grouped.items():
        labels: list[str] = entry["speaker_labels"]
        display_name = entry["display_name"] or f"Presenter ({labels[0]})"
        result.append((user_id, display_name, labels))
    return result


def _collect_evidence_ids(*sources: dict[str, Any] | None) -> list[str]:
    """Collect deduplicated, non-empty evidence IDs across question/answer/assessment dicts."""
    collected: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        raw_ids = source.get("evidence_ids", [])
        if not isinstance(raw_ids, list):
            continue
        for eid in raw_ids:
            eid_str = str(eid or "").strip()
            if eid_str and eid_str not in collected:
                collected.append(eid_str)
    return collected


def _build_member_qa_feedback(
    *,
    user_id: str,
    display_name: str,
    speaker_labels: list[str],
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
    rubric_id: str,
) -> FeedbackSection | None:
    """Build grounded Q&A feedback for a mapped member who submitted at least one answer."""
    submitted_answers = [
        ans
        for ans in answers
        if isinstance(ans, dict)
        and str(ans.get("answered_by") or "").strip() == user_id
        and str(ans.get("status") or "submitted").strip().lower() not in {"skipped", "draft"}
    ]
    if not submitted_answers:
        return None

    questions_by_id: dict[str, dict[str, Any]] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid_raw = q.get("id") or q.get("question_id") or q.get("candidate_id")
        if qid_raw:
            questions_by_id[str(qid_raw)] = q

    assessments_by_qid: dict[str, dict[str, Any]] = {}
    assessments_by_aid: dict[str, dict[str, Any]] = {}
    for asm in assessments:
        if not isinstance(asm, dict):
            continue
        aid_raw = asm.get("answer_id")
        if aid_raw:
            assessments_by_aid[str(aid_raw)] = asm
        qid_raw = asm.get("question_id") or asm.get("candidate_id")
        if qid_raw:
            assessments_by_qid[str(qid_raw)] = asm

    qa_strengths: list[Finding] = []
    qa_improvements: list[Finding] = []
    scored_values: list[float] = []
    scored_evidence_ids: list[str] = []
    unscored_count = 0

    for idx, ans in enumerate(submitted_answers, start=1):
        qid = str(ans.get("question_id") or ans.get("candidate_id") or "")
        ans_id = str(ans.get("id") or ans.get("answer_id") or "")
        q = questions_by_id.get(qid, {})
        matched_asm = (assessments_by_aid.get(ans_id) if ans_id else None) or (
            assessments_by_qid.get(qid) if qid else None
        )

        score = extract_assessment_score(matched_asm)
        if score is None:
            unscored_count += 1
            continue

        scored_values.append(score)
        ev_ids = _collect_evidence_ids(matched_asm, ans, q)
        for eid in ev_ids:
            if eid not in scored_evidence_ids:
                scored_evidence_ids.append(eid)

        nested_asm = matched_asm.get("assessment") if isinstance(matched_asm, dict) else None
        raw_text = (matched_asm or {}).get("assessment_text")
        if not raw_text and isinstance(nested_asm, dict):
            raw_text = nested_asm.get("text")
        assessment_text = str(raw_text or "").strip()

        q_dim = str(q.get("rubric_dimension") or "qa_quality")
        q_text = str(q.get("text") or "").strip()
        finding_eids = ev_ids or ([f"ev_answer_{ans_id}"] if ans_id else [f"ev_qa_{qid or idx}"])
        dim_label = q_dim.replace("_", " ").title()
        disp_score = score_to_display(score)
        detail = assessment_text or f"Response to Q&A prompt scored {disp_score}/100."

        if score >= 0.60:
            qa_strengths.append(
                Finding(
                    id=f"f_{user_id}_qa_s{idx}",
                    kind="strength",
                    title=f"Grounded Q&A Response ({dim_label})",
                    detail=detail,
                    evidence_ids=finding_eids,
                    rubric_dimension=q_dim,
                    speaker_labels=list(speaker_labels),
                )
            )
        else:
            recommendation = (
                f"Provide concrete evidence addressing: {q_text[:180]}"
                if q_text
                else "Address the core question directly with concrete supporting evidence."
            )
            qa_improvements.append(
                Finding(
                    id=f"f_{user_id}_qa_i{idx}",
                    kind="improvement",
                    title=f"Strengthen Q&A Depth ({dim_label})",
                    detail=detail,
                    recommendation=recommendation,
                    evidence_ids=finding_eids,
                    rubric_dimension=q_dim,
                    speaker_labels=list(speaker_labels),
                )
            )

    qa_limitations: list[Limitation] = []
    if unscored_count > 0:
        qa_limitations.append(
            Limitation(
                code="qa_assessment_unavailable",
                scope="qa_feedback",
                message=(
                    f"Assessment score was unavailable for {unscored_count} Q&A response(s) "
                    f"submitted by {display_name}."
                ),
                affected_dimensions=["qa_quality"],
            )
        )

    if scored_values:
        mean_score = round(sum(scored_values) / len(scored_values), 4)
        comp_eids = scored_evidence_ids or [f"ev_qa_{user_id}"]
        qa_comp = ScoreComponent(
            dimension="qa_quality",
            status="scored",
            normalized_score=mean_score,
            display_score=score_to_display(mean_score),
            label=score_to_label(mean_score, rubric_id=rubric_id),
            configured_weight=0.20,
            effective_weight=0.20,
            evidence_ids=comp_eids,
            rationale=(
                f"Mean score across {len(scored_values)} evaluated Q&A response(s) "
                f"by {display_name}."
            ),
        )
        summary = (
            f"{display_name} answered {len(submitted_answers)} Q&A question(s) "
            f"({len(scored_values)} evaluated, mean score {score_to_display(mean_score)}/100)."
        )
    else:
        qa_comp = ScoreComponent(
            dimension="qa_quality",
            status="not_evaluated",
            configured_weight=0.20,
            limitation_code="qa_assessment_unavailable",
            rationale=(
                f"Q&A responses by {display_name} could not be scored because assessment "
                "scores were unavailable."
            ),
        )
        summary = (
            f"{display_name} submitted {len(submitted_answers)} Q&A response(s), "
            "but assessment scores were unavailable."
        )

    return FeedbackSection(
        summary=summary,
        strengths=qa_strengths,
        improvements=qa_improvements,
        score_components=[qa_comp],
        limitations=qa_limitations,
    )


async def run_report_stage(
    report_id: str,
    *,
    analysis_ref: ArtifactRef | None = None,
    qa_ref: ArtifactRef | None = None,
    storage: ObjectStorageProtocol | None = None,
    judge_provider: JudgeModelProvider | None = None,
    speaker_mappings: list[SpeakerMapping] | None = None,
    evidence_bundle_override: ReportEvidenceBundle | None = None,
    analysis_attempt: int | None = None,
    analysis_attempt_id: str | None = None,
    session_title: str = "Startup Pitch Practice Session",
    session_date: str | None = None,
    rubric_id: str = "startup_pitch",
    rubric_version: int = 1,
    pipeline_version: str = "0.1.0",
    mode_policy: RuntimeModePolicy | None = None,
    allow_synthetic_fixtures: bool | None = None,
) -> ReportStageResult:
    """Execute the end-to-end evaluation and report generation stage.

    1. Loads evidence and Q&A artifacts from object storage (or synthetic defaults).
    2. Calls JudgeModelProvider to generate grounded executive summary and feedback.
    3. Calculates deterministic Q&A scores (20% rubric weight, skips=0.0)
       and individual delivery scores.
    4. Evaluates startup pitch rubric, computing normalized effective weights and overall score.
    5. Maps active speaking turn intervals to each mapped presenter's MemberFeedback.
    6. Formats canonical human-readable Markdown report (report.md).
    7. Validates and returns the complete Evaluation model and report text.
    """
    judge = judge_provider or FakeJudgeModelProvider()
    start_time = time.monotonic()
    started_at = datetime.datetime.now(datetime.UTC).isoformat()

    # 1. Load evidence bundle
    if evidence_bundle_override is not None:
        bundle = evidence_bundle_override
    else:
        if analysis_ref is None or qa_ref is None:
            raise ValueError(
                "analysis_ref and qa_ref are required when evidence_bundle_override is not set."
            )
        bundle = await load_report_evidence(
            analysis_ref=analysis_ref,
            qa_ref=qa_ref,
            storage=storage,
            mode_policy=mode_policy,
            allow_synthetic_fixtures=allow_synthetic_fixtures,
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
    # Calculate deterministic Q&A score (20% rubric weight; skipped questions score 0.0;
    # answered questions without a recoverable score become not_evaluated with a limitation)
    qa_result = evaluate_qa_scores(
        questions=bundle.questions,
        answers=bundle.answers,
        assessments=bundle.assessments,
    )
    qa_score = qa_result.score
    qa_evidence_ids = qa_result.evidence_ids

    limitations = list(bundle.limitations)
    has_qa_limitation = any(
        "qa_quality" in lim.affected_dimensions
        and lim.code in ("qa_assessment_unavailable", "qa_assessment_verification_failed")
        for lim in limitations
    )
    if not has_qa_limitation:
        for lim in qa_result.limitations:
            limitations.append(lim)

    # Calculate average delivery scores across all active speakers for the
    # team-level rubric components
    team_delivery_scores: list[float] = []
    team_timing_scores: list[float] = []
    for spk_prof in bundle.by_speaker.values():
        spk_comps = calculate_individual_delivery_scores(spk_prof, rubric_id=rubric_id)
        for sc in spk_comps:
            if (
                sc.dimension == "delivery_and_body_language"
                and sc.status == "scored"
                and sc.normalized_score is not None
            ):
                team_delivery_scores.append(sc.normalized_score)
            elif (
                sc.dimension == "timing_and_speech_mechanics"
                and sc.status == "scored"
                and sc.normalized_score is not None
            ):
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
        "pitch_content_and_evidence": feedback_result.dimension_scores.get(
            "pitch_content_and_evidence"
        ),
        "business_and_problem_solution_reasoning": feedback_result.dimension_scores.get(
            "business_and_problem_solution_reasoning"
        ),
        "technical_feasibility": feedback_result.dimension_scores.get("technical_feasibility"),
        "delivery_and_body_language": team_delivery,
        "timing_and_speech_mechanics": team_timing,
        "qa_quality": qa_score if bundle.questions else None,
    }

    # Rationales
    dimension_rationales = dict(feedback_result.dimension_rationales)
    if bundle.questions and qa_score is None:
        dimension_rationales["qa_quality"] = (
            "Q&A responses could not be scored because assessment scores were unavailable."
        )
    elif qa_result.not_evaluated_question_ids:
        evaluated_count = len(bundle.questions) - len(qa_result.not_evaluated_question_ids)
        dimension_rationales["qa_quality"] = (
            f"Evaluated across {evaluated_count} scored or skipped question(s); "
            f"{len(qa_result.not_evaluated_question_ids)} answered question(s) could not be "
            "evaluated."
        )
    elif "qa_quality" not in dimension_rationales:
        skipped_count = len([a for a in bundle.answers if a.get("status") == "skipped"])
        total_q = len(bundle.questions)
        dimension_rationales["qa_quality"] = (
            f"Deterministic evaluation across {total_q} question(s) with {skipped_count} skipped."
        )

    # Evidence mapping per dimension
    evidence_by_dim: dict[str, list[str]] = {
        "qa_quality": list(qa_evidence_ids),
    }
    for chunk in bundle.document_chunks:
        cid = chunk.get("chunk_id") or chunk.get("evidence_id")
        if cid:
            evidence_by_dim.setdefault("technical_feasibility", []).append(cid)
            evidence_by_dim.setdefault("business_and_problem_solution_reasoning", []).append(cid)

    for seg in bundle.transcript_segments:
        sid = seg.get("segment_id") or seg.get("evidence_id")
        if sid:
            evidence_by_dim.setdefault("pitch_content_and_evidence", []).append(sid)
            evidence_by_dim.setdefault("business_and_problem_solution_reasoning", []).append(sid)

    for spk_label, spk_prof in bundle.by_speaker.items():
        if spk_prof.get("windows"):
            evidence_by_dim.setdefault("delivery_and_body_language", []).append(
                f"ev_vision_{spk_label.lower()}"
            )
            evidence_by_dim.setdefault("timing_and_speech_mechanics", []).append(
                f"ev_audio_{spk_label.lower()}"
            )

    overall_score, rubric_components = evaluate_rubric(
        raw_scores=raw_dimension_scores,
        rubric_id=rubric_id,
        evidence_by_dimension=evidence_by_dim,
        rationales=dimension_rationales,
        limitations=limitations,
    )

    if overall_score is None:
        append_unique_limitation(
            limitations,
            Limitation(
                code="all_dimensions_not_evaluated",
                scope="evaluation",
                message=(
                    "No rubric dimensions could be evaluated from the available evidence; "
                    "overall score is unavailable."
                ),
                affected_dimensions=[comp.dimension for comp in rubric_components],
            ),
        )

    # 4. Map Individual Presenter MemberFeedback
    member_feedbacks: list[MemberFeedback] = []
    member_user_ids: list[str] = []

    if not speaker_mappings:
        append_unique_limitation(
            limitations,
            Limitation(
                code="speaker_mappings_not_provided",
                scope="member_feedback",
                message=(
                    "No speaker mappings were provided; individual presenter feedback was omitted."
                ),
                affected_dimensions=[],
            ),
        )
    else:
        for user_id, display_name, spk_labels in _group_speaker_mappings(speaker_mappings):
            member_user_ids.append(user_id)

            speaking_intervals: list[SpeakingInterval] = []
            speaking_time_ms = 0
            combined_windows: list[Any] = []
            acoustic_eids: list[str] = []
            visual_eids: list[str] = []
            mbr_strengths: list[Finding] = []
            mbr_improvements: list[Finding] = []

            for spk_label in spk_labels:
                spk_profile = bundle.by_speaker.get(spk_label, {})
                for item in spk_profile.get("intervals", []):
                    if isinstance(item, SpeakingInterval):
                        speaking_intervals.append(item)
                    elif isinstance(item, dict):
                        start_ms = item.get("start_ms", 0)
                        end_ms = item.get("end_ms", 0)
                        s_min, s_sec = divmod(start_ms // 1000, 60)
                        e_min, e_sec = divmod(end_ms // 1000, 60)
                        fmt = (
                            item.get("formatted")
                            or f"{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}"
                        )
                        speaking_intervals.append(
                            SpeakingInterval(start_ms=start_ms, end_ms=end_ms, formatted=fmt)
                        )

                speaking_time_ms += int(spk_profile.get("speaking_time_ms", 0) or 0)
                combined_windows.extend(spk_profile.get("windows", []))
                for eid in spk_profile.get("acoustic_evidence_ids", []):
                    if eid not in acoustic_eids:
                        acoustic_eids.append(eid)
                for eid in spk_profile.get("visual_evidence_ids", []):
                    if eid not in visual_eids:
                        visual_eids.append(eid)

            speaking_intervals.sort(key=lambda inv: (inv.start_ms, inv.end_ms))

            combined_profile: dict[str, Any] = {
                "speaker_label": spk_labels[0],
                "speaking_time_ms": speaking_time_ms,
                "windows": combined_windows,
            }
            if acoustic_eids:
                combined_profile["acoustic_evidence_ids"] = acoustic_eids
            if visual_eids:
                combined_profile["visual_evidence_ids"] = visual_eids

            delivery_components = calculate_individual_delivery_scores(
                combined_profile, rubric_id=rubric_id
            )

            has_delivery_data = bool(speaking_intervals or combined_windows) or (
                speaking_time_ms > 0
            )
            if not has_delivery_data:
                # Delivery feedback must be grounded in a recorded turn, duration, or
                # observation. Do not turn missing media into invented praise or advice.
                summary = "No delivery data was recorded for this presenter."
            else:
                for spk_label in spk_labels:
                    mbr_strengths.extend(feedback_result.member_strengths.get(spk_label, []))
                    mbr_improvements.extend(feedback_result.member_improvements.get(spk_label, []))

                primary_label = spk_labels[0]
                if not mbr_strengths:
                    mbr_strengths = [
                        Finding(
                            id=f"f_{primary_label.lower()}_s1",
                            kind="strength",
                            title="Engaged Vocal Presence",
                            detail=(
                                f"{display_name} maintained clear vocal delivery "
                                "throughout their presentation turns."
                            ),
                            rubric_dimension="delivery_and_body_language",
                            speaker_labels=list(spk_labels),
                        )
                    ]
                if not mbr_improvements:
                    mbr_improvements = [
                        Finding(
                            id=f"f_{primary_label.lower()}_i1",
                            kind="improvement",
                            title="Slide Transition Calibration",
                            detail=(
                                f"{display_name} can use deliberate pauses when introducing "
                                "new slides."
                            ),
                            recommendation=(
                                "Insert a 1-2 second pause before transitioning to new sections."
                            ),
                            rubric_dimension="timing_and_speech_mechanics",
                            speaker_labels=list(spk_labels),
                        )
                    ]

                time_display = format_duration_ms(speaking_time_ms)
                summary = (
                    f"Active presentation delivery analysis for {display_name} across "
                    f"{len(speaking_intervals)} turn(s) ({time_display} total)."
                )

            qa_feedback = _build_member_qa_feedback(
                user_id=user_id,
                display_name=display_name,
                speaker_labels=spk_labels,
                questions=bundle.questions,
                answers=bundle.answers,
                assessments=bundle.assessments,
                rubric_id=rubric_id,
            )

            member_feedbacks.append(
                MemberFeedback(
                    user_id=user_id,
                    display_name=display_name,
                    speaker_labels=spk_labels,
                    speaking_intervals=speaking_intervals,
                    speaking_time_ms=speaking_time_ms,
                    summary=summary,
                    strengths=mbr_strengths,
                    improvements=mbr_improvements,
                    delivery_components=delivery_components,
                    qa_feedback=qa_feedback,
                )
            )

    # 5. Whole-Team Feedback Section & Consolidated Findings
    team_feedback = FeedbackSection(
        summary=feedback_result.executive_summary,
        strengths=feedback_result.team_strengths,
        improvements=feedback_result.team_improvements,
        score_components=rubric_components,
        limitations=limitations,
    )

    all_findings: list[Finding] = list(feedback_result.team_strengths) + list(
        feedback_result.team_improvements
    )
    for mf in member_feedbacks:
        all_findings.extend(mf.strengths)
        all_findings.extend(mf.improvements)
        if mf.qa_feedback is not None:
            all_findings.extend(mf.qa_feedback.strengths)
            all_findings.extend(mf.qa_feedback.improvements)

    # 6. Construct Evaluation Model
    eval_id = ulid.new().str
    session_id = bundle.practice_session_id or bundle.session_id or report_id
    attempt_num = (
        analysis_attempt
        if analysis_attempt is not None
        else (bundle.analysis_attempt or bundle.metadata.get("analysis_attempt") or 1)
    )
    resolved_attempt_id = (
        analysis_attempt_id
        or bundle.analysis_attempt_id
        or (
            str(bundle.metadata["analysis_attempt_id"])
            if bundle.metadata.get("analysis_attempt_id")
            else None
        )
        or f"{session_id}:attempt_{attempt_num}"
    )
    qa_round_id = (
        bundle.qa_round_id
        or (str(bundle.metadata["qa_round_id"]) if bundle.metadata.get("qa_round_id") else None)
        or f"{session_id}:round_{attempt_num}"
    )

    completed_at = datetime.datetime.now(datetime.UTC).isoformat()
    duration_ms = max(0, int((time.monotonic() - start_time) * 1000))

    reproducibility: dict[str, Any] = {
        "pipeline_version": pipeline_version,
        "rubric_id": rubric_id,
        "rubric_version": rubric_version,
        "weighting_policy_version": WEIGHTING_POLICY_VERSION,
        "prompt_versions": {
            "reporting": "1.0.0",
            "scoring": "1.0.0",
        },
        "contract_versions": {
            "data_contracts": "1.0.0",
            "backend_ai": "1.0.0",
        },
        "input_checksums": {
            "analysis": analysis_ref.checksum if analysis_ref else None,
            "qa": qa_ref.checksum if qa_ref else None,
        },
        "stage_versions": {
            "speech": "1.0.0",
            "vision": "1.0.0",
            "audio": "1.0.0",
            "documents": "1.0.0",
            "reporting": "1.0.0",
        },
        "model_runs": [
            {
                "capability": "evaluation_reporting",
                "provider": type(judge).__name__,
                "model_id": getattr(judge, "model_name", "judge_v1"),
                "adapter_version": "1.0.0",
                "duration_ms": duration_ms,
            }
        ],
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "cost_summary": {
            "estimated_usd": 0.0,
            "currency": "USD",
        },
        "scoring_engine": "startup_pitch_v1",
        "evaluator_provider": type(judge).__name__,
        "evaluated_at": completed_at,
    }
    if bundle.metadata.get("is_synthetic_fixture") or any(
        lim.code in {"synthetic_analysis_fixture", "synthetic_qa_fixture"}
        for lim in bundle.limitations
    ):
        reproducibility["synthetic_fixtures"] = True
        if bundle.metadata.get("synthetic_sources"):
            reproducibility["synthetic_sources"] = list(bundle.metadata["synthetic_sources"])

    evaluation = Evaluation(
        id=eval_id,
        analysis_attempt_id=resolved_attempt_id,
        qa_round_id=qa_round_id,
        rubric=RubricRef(rubric_id=rubric_id, version=rubric_version),
        overall_score=overall_score,
        components=rubric_components,
        findings=all_findings,
        team_feedback=team_feedback,
        member_feedback=member_feedbacks,
        limitations=limitations,
        reproducibility=reproducibility,
    )

    # 7. Generate Canonical Markdown Report
    md_assessments = [dict(a) for a in bundle.assessments]
    if qa_result.not_evaluated_question_ids:
        not_eval_set = set(qa_result.not_evaluated_question_ids)
        seen_md_qids: set[str] = set()
        for asm in md_assessments:
            asm_qid = str(asm.get("question_id") or asm.get("candidate_id") or "")
            if asm_qid in not_eval_set:
                asm["status"] = "not_evaluated"
                asm["score"] = None
                seen_md_qids.add(asm_qid)
        for missing_qid in qa_result.not_evaluated_question_ids:
            if missing_qid not in seen_md_qids and missing_qid != "<unknown>":
                md_assessments.append(
                    {"question_id": missing_qid, "status": "not_evaluated", "score": None}
                )

    qa_data = {
        "questions": bundle.questions,
        "answers": bundle.answers,
        "assessments": md_assessments,
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
        limitations=limitations,
    )


__all__ = [
    "ReportStageResult",
    "run_report_stage",
]
