"""Markdown report generator for VirtuJudge final evaluation (AI-07).

Transforms structured Evaluation models, Q&A records, and windowed speaker profiles
into the canonical human-readable report.md artifact conforming to Docs/Product/Requirements.md.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.contracts import Evaluation, MemberFeedback, ScoreComponent
from app.stages.aggregation import format_duration_ms
from app.stages.scoring import score_to_display, score_to_label

DIMENSION_DISPLAY_NAMES: dict[str, str] = {
    "pitch_content_and_evidence": "Pitch Content & Evidence",
    "business_and_problem_solution_reasoning": "Business & Problem-Solution Reasoning",
    "technical_feasibility": "Technical Feasibility",
    "delivery_and_body_language": "Delivery & Body Language",
    "timing_and_speech_mechanics": "Timing & Speech Mechanics",
    "qa_quality": "Q&A Quality",
}

SCORE_LABEL_DISPLAY: dict[str, str] = {
    "needs_work": "Needs Work",
    "developing": "Developing",
    "good": "Good",
    "strong": "Strong",
}


def _dim_title(dimension_key: str) -> str:
    """Format dimension key into friendly display title."""
    return DIMENSION_DISPLAY_NAMES.get(dimension_key, dimension_key.replace("_", " ").title())


def _label_title(label: str | None) -> str:
    """Format score label into capitalized title."""
    if not label:
        return "Not Rated"
    return SCORE_LABEL_DISPLAY.get(label, label.replace("_", " ").title())


def format_speaker_metrics_highlight(profile: dict[str, Any] | None) -> list[str]:
    """Compute and format vocal and visual metric highlights from a speaker's 10s windows."""
    if not profile:
        return ["*No active delivery metrics recorded for this presenter.*"]

    windows = profile.get("windows", [])
    if not windows:
        return ["*No windowed observations recorded during active turns.*"]

    wpm_vals = [
        w["acoustic"]["speaking_rate_wpm"]
        for w in windows
        if "acoustic" in w and "speaking_rate_wpm" in w["acoustic"]
    ]
    pitch_vals = [
        w["acoustic"]["pitch_mean_hz"]
        for w in windows
        if "acoustic" in w and "pitch_mean_hz" in w["acoustic"]
    ]
    pause_durs = [
        w["acoustic"]["pause_duration_ms"]
        for w in windows
        if "acoustic" in w and "pause_duration_ms" in w["acoustic"]
    ]
    pause_counts = [
        w["acoustic"]["pause_count"]
        for w in windows
        if "acoustic" in w and "pause_count" in w["acoustic"]
    ]
    fillers = [
        w["acoustic"]["filler_count"]
        for w in windows
        if "acoustic" in w and "filler_count" in w["acoustic"]
    ]

    gaze_vals = [
        w["visual"]["gaze_direction"]
        for w in windows
        if "visual" in w and "gaze_direction" in w["visual"]
    ]
    posture_vals = [
        w["visual"]["posture_openness"]
        for w in windows
        if "visual" in w and "posture_openness" in w["visual"]
    ]
    movement_vals = [
        w["visual"]["upper_body_movement_px"]
        for w in windows
        if "visual" in w and "upper_body_movement_px" in w["visual"]
    ]

    lines: list[str] = []

    # Acoustic highlights
    acoustic_parts: list[str] = []
    if wpm_vals:
        avg_wpm = sum(wpm_vals) / len(wpm_vals)
        acoustic_parts.append(f"**Pace:** {avg_wpm:.1f} WPM")
    if pitch_vals:
        avg_pitch = sum(pitch_vals) / len(pitch_vals)
        acoustic_parts.append(f"**Pitch:** {avg_pitch:.1f} Hz")
    if pause_durs:
        total_pause_ms = sum(pause_durs)
        total_pauses = sum(pause_counts)
        acoustic_parts.append(f"**Pauses:** {total_pause_ms} ms ({total_pauses} count)")
    if fillers:
        total_fillers = sum(fillers)
        acoustic_parts.append(f"**Fillers:** {total_fillers}")

    if acoustic_parts:
        lines.append("- 🎙️ **Acoustic & Pacing:** " + " | ".join(acoustic_parts))

    # Visual highlights
    visual_parts: list[str] = []
    if gaze_vals:
        avg_gaze = sum(gaze_vals) / len(gaze_vals)
        visual_parts.append(f"**Gaze Alignment:** {avg_gaze:.2f}")
    if posture_vals:
        avg_posture = sum(posture_vals) / len(posture_vals)
        visual_parts.append(f"**Posture Openness:** {avg_posture:.2f}")
    if movement_vals:
        avg_mov = sum(movement_vals) / len(movement_vals)
        visual_parts.append(f"**Upper Body Movement:** {avg_mov:.1f} px")

    if visual_parts:
        lines.append("- 👁️ **Visual & Body Language:** " + " | ".join(visual_parts))

    return lines or ["*Observations recorded without aggregated metric features.*"]


def generate_markdown_report(
    evaluation: Evaluation,
    *,
    session_title: str = "Startup Pitch Practice Session",
    session_date: str | None = None,
    qa_data: dict[str, Any] | None = None,
    by_speaker: dict[str, Any] | None = None,
) -> str:
    """Generate the complete, canonical markdown report (report.md) from an Evaluation model."""
    date_str = session_date or datetime.date.today().isoformat()
    overall_score_val = evaluation.overall_score

    # Overall score badge
    if overall_score_val is not None:
        display_score = score_to_display(overall_score_val)
        label_key = score_to_label(overall_score_val)
        label_str = _label_title(label_key)
        score_badge = f"{label_str} ({display_score}/100)"
    else:
        score_badge = "Not Rated"

    out: list[str] = []

    # 1. Header & Executive Summary
    out.append(f"# {session_title} — Final Evaluation Report")
    out.append("")
    out.append(f"- **Evaluation ID:** `{evaluation.id}`")
    out.append(f"- **Session ID:** `{evaluation.analysis_attempt_id}`")
    out.append(f"- **Date:** {date_str}")
    out.append(f"- **Rubric:** `{evaluation.rubric.rubric_id}` (v{evaluation.rubric.version})")
    out.append(f"- **Overall Performance:** 🏆 **{score_badge}**")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Executive Summary")
    out.append("")
    summary_text = evaluation.team_feedback.summary or "No executive summary provided."
    out.append(summary_text)
    out.append("")

    # 2. Rubric Score Breakdown Table
    out.append("## Rubric Score Breakdown")
    out.append("")
    out.append("| Rubric Dimension | Configured Weight | Effective Weight | Score | Rating | Status |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for comp in evaluation.components:
        dim_name = _dim_title(comp.dimension)
        cfg_wt = f"{comp.configured_weight * 100:.0f}%"
        eff_wt = f"{comp.effective_weight * 100:.0f}%" if comp.effective_weight is not None else "-"
        score_str = f"{comp.display_score} / 100" if comp.display_score is not None else "-"
        label_disp = _label_title(comp.label) if comp.label else "-"
        out.append(f"| {dim_name} | {cfg_wt} | {eff_wt} | {score_str} | {label_disp} | `{comp.status}` |")

    out.append("")

    # Dimension rationales
    rationales = [c for c in evaluation.components if c.rationale]
    if rationales:
        out.append("### Dimension Rationales")
        out.append("")
        for c in rationales:
            out.append(f"- **{_dim_title(c.dimension)}**: {c.rationale}")
        out.append("")

    # 3. Whole-Team Pitch Assessment
    out.append("## Team Pitch Assessment")
    out.append("")

    team_strengths = evaluation.team_feedback.strengths
    out.append("### Core Strengths")
    out.append("")
    if team_strengths:
        for f in team_strengths:
            eids = f" (`{', '.join(f.evidence_ids)}`)" if f.evidence_ids else ""
            out.append(f"- **{f.title}**{eids}: {f.detail}")
            if f.recommendation:
                out.append(f"  - *Advice:* {f.recommendation}")
    else:
        out.append("*No specific team strengths recorded.*")
    out.append("")

    team_improvements = evaluation.team_feedback.improvements
    out.append("### Priority Areas for Improvement")
    out.append("")
    if team_improvements:
        for f in team_improvements:
            eids = f" (`{', '.join(f.evidence_ids)}`)" if f.evidence_ids else ""
            out.append(f"- **{f.title}**{eids}: {f.detail}")
            if f.recommendation:
                out.append(f"  - *Recommendation:* {f.recommendation}")
    else:
        out.append("*No specific areas for improvement recorded.*")
    out.append("")

    # 4. Individual Presenter Delivery Feedback
    out.append("## Individual Presenter Delivery Feedback")
    out.append("")
    if evaluation.member_feedback:
        for member in evaluation.member_feedback:
            spk_label = member.speaker_labels[0] if member.speaker_labels else "SPEAKER_UNKNOWN"
            out.append(f"### {member.display_name} ({spk_label})")
            out.append("")

            # Turns & total speaking duration
            turn_strs = [inv.formatted for inv in member.speaking_intervals]
            turns_disp = ", ".join(turn_strs) if turn_strs else "None recorded"
            total_dur = format_duration_ms(member.speaking_time_ms)
            out.append(f"> ⏱️ **Active Speaking Turns:** {turns_disp} | **Total Speaking Time:** {total_dur}")
            out.append("")

            if member.summary:
                out.append(member.summary)
                out.append("")

            # Delivery score components table
            if member.delivery_components:
                out.append("#### Delivery Scores")
                out.append("")
                out.append("| Dimension | Score | Rating |")
                out.append("| :--- | :---: | :---: |")
                for dc in member.delivery_components:
                    d_name = _dim_title(dc.dimension)
                    d_score = f"{dc.display_score} / 100" if dc.display_score is not None else "-"
                    d_label = _label_title(dc.label) if dc.label else "-"
                    out.append(f"| {d_name} | {d_score} | {d_label} |")
                out.append("")

            # Windowed metrics highlight
            spk_profile = (by_speaker or {}).get(spk_label)
            metric_lines = format_speaker_metrics_highlight(spk_profile)
            out.append("#### Delivery Metrics Highlight")
            out.append("")
            for ml in metric_lines:
                out.append(ml)
            out.append("")

            # Member strengths
            out.append("#### Presenter Strengths")
            out.append("")
            if member.strengths:
                for s in member.strengths:
                    out.append(f"- **{s.title}**: {s.detail}")
                    if s.recommendation:
                        out.append(f"  - *Advice:* {s.recommendation}")
            else:
                out.append("*No presenter-specific strengths recorded.*")
            out.append("")

            # Member improvements
            out.append("#### Presenter Areas for Improvement")
            out.append("")
            if member.improvements:
                for imp in member.improvements:
                    out.append(f"- **{imp.title}**: {imp.detail}")
                    if imp.recommendation:
                        out.append(f"  - *Action Step:* {imp.recommendation}")
            else:
                out.append("*No presenter-specific improvements recorded.*")
            out.append("")
    else:
        out.append("*No mapped individual presenter feedback available.*")
        out.append("")

    # 5. Q&A Deep Dive (Team Performance)
    out.append("## Q&A Session Deep Dive")
    out.append("")
    if qa_data and qa_data.get("questions"):
        questions = qa_data.get("questions", [])
        answers_map: dict[str, dict[str, Any]] = {
            a.get("question_id", ""): a for a in qa_data.get("answers", [])
        }
        assessments_map: dict[str, dict[str, Any]] = {
            ass.get("question_id", ""): ass for ass in qa_data.get("assessments", [])
        }

        for idx, q in enumerate(questions, start=1):
            q_id = q.get("id", "")
            q_text = q.get("text", "")
            q_dim = _dim_title(q.get("rubric_dimension", ""))
            ans = answers_map.get(q_id, {})
            ass = assessments_map.get(q_id, {})

            out.append(f"### Question {idx}: {q_text}")
            out.append("")
            out.append(f"- **Rubric Dimension:** {q_dim}")

            ans_status = ans.get("status", "submitted")
            if ans_status == "skipped":
                out.append("- **Status:** ⚠️ *Skipped by team* (Score: 0/100)")
            else:
                answered_by = ans.get("answered_by") or "Team Member"
                out.append(f"- **Answered By:** {answered_by}")
                ans_transcript = ans.get("transcript")
                if ans_transcript:
                    out.append(f"- **Response:** *\"{ans_transcript}\"*")

            ass_score = ass.get("score")
            if ass_score is not None:
                ass_disp = score_to_display(float(ass_score))
                ass_label = _label_title(score_to_label(float(ass_score)))
                out.append(f"- **Assessment Score:** {ass_label} ({ass_disp}/100)")

            ass_text = ass.get("assessment_text")
            if ass_text:
                out.append(f"- **Judge Assessment:** {ass_text}")

            out.append("")
    else:
        out.append("*No Q&A exchange records available for this session.*")
        out.append("")

    # 6. Appendix: System Limitations & Reproducibility Metadata
    out.append("## Appendix")
    out.append("")
    out.append("### Pipeline Limitations & Boundaries")
    out.append("")
    if evaluation.limitations:
        for lim in evaluation.limitations:
            out.append(f"- **[{lim.code}]** *({lim.scope})*: {lim.message}")
    else:
        out.append("- *No systemic pipeline limitations recorded during evaluation.*")
    out.append("")

    out.append("### Reproducibility & Model Metadata")
    out.append("")
    if evaluation.reproducibility:
        for k, v in sorted(evaluation.reproducibility.items()):
            k_disp = k.replace("_", " ").title()
            out.append(f"- **{k_disp}:** `{v}`")
    else:
        out.append("- *No reproducibility metadata provided.*")
    out.append("")

    return "\n".join(out).strip() + "\n"


__all__ = [
    "DIMENSION_DISPLAY_NAMES",
    "SCORE_LABEL_DISPLAY",
    "format_speaker_metrics_highlight",
    "generate_markdown_report",
]
