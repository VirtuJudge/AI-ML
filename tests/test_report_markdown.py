"""Unit tests for app.stages.report_markdown (Markdown Report Generator)."""

from typing import Any

from app.contracts import (
    Evaluation,
    FeedbackSection,
    Finding,
    Limitation,
    MemberFeedback,
    RubricRef,
    ScoreComponent,
    SpeakingInterval,
)
from app.stages.report_markdown import (
    format_speaker_metrics_highlight,
    generate_markdown_report,
)


def _make_sample_evaluation() -> Evaluation:
    """Construct realistic Evaluation instance for testing."""
    return Evaluation(
        id="01JEVAL0000000000000000001",
        analysis_attempt_id="01JATTEMPT0000000000000001",
        qa_round_id="01JROUND000000000000000001",
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        overall_score=0.785,
        components=[
            ScoreComponent(
                dimension="pitch_content_and_evidence",
                status="scored",
                normalized_score=0.85,
                display_score=85,
                label="strong",
                configured_weight=0.25,
                effective_weight=0.25,
                evidence_ids=["ev_speech_001"],
                rationale="Compelling market definition supported by early pilot metrics.",
            ),
            ScoreComponent(
                dimension="business_and_problem_solution_reasoning",
                status="scored",
                normalized_score=0.75,
                display_score=75,
                label="good",
                configured_weight=0.20,
                effective_weight=0.20,
                evidence_ids=["ev_speech_002"],
                rationale="Sound unit economics; competitor matrix needs further granularity.",
            ),
            ScoreComponent(
                dimension="technical_feasibility",
                status="scored",
                normalized_score=0.82,
                display_score=82,
                label="strong",
                configured_weight=0.15,
                effective_weight=0.15,
                evidence_ids=["ev_doc_001"],
                rationale="Robust proprietary pipeline architecture and defensible IP.",
            ),
            ScoreComponent(
                dimension="delivery_and_body_language",
                status="scored",
                normalized_score=0.70,
                display_score=70,
                label="good",
                configured_weight=0.15,
                effective_weight=0.15,
                evidence_ids=[],
                rationale="Open posture and solid gaze stability across presenters.",
            ),
            ScoreComponent(
                dimension="timing_and_speech_mechanics",
                status="scored",
                normalized_score=0.68,
                display_score=68,
                label="good",
                configured_weight=0.05,
                effective_weight=0.05,
                evidence_ids=[],
                rationale="Well-calibrated pace; occasional filler clusters during transitions.",
            ),
            ScoreComponent(
                dimension="qa_quality",
                status="scored",
                normalized_score=0.80,
                display_score=80,
                label="strong",
                configured_weight=0.20,
                effective_weight=0.20,
                evidence_ids=[],
                rationale="Direct responses to technical and go-to-market questions.",
            ),
        ],
        findings=[
            Finding(
                id="f_team_s1",
                kind="strength",
                title="Grounded Value Proposition",
                detail="The team articulated the core enterprise problem with real customer data.",
                evidence_ids=["ev_speech_001"],
                rubric_dimension="pitch_content_and_evidence",
            ),
            Finding(
                id="f_team_i1",
                kind="improvement",
                title="Expand Competitor Defense",
                detail="Differentiate more sharply against legacy incumbent solutions.",
                recommendation="Add a side-by-side feature matrix to Slide 4.",
                evidence_ids=["ev_speech_002"],
                rubric_dimension="business_and_problem_solution_reasoning",
            ),
        ],
        team_feedback=FeedbackSection(
            summary=(
                "The team delivered a persuasive, evidence-grounded pitch with solid technical depth. "
                "Delivery was well-coordinated with clear speaking roles and steady pacing."
            ),
            strengths=[
                Finding(
                    id="f_team_s1",
                    kind="strength",
                    title="Grounded Value Proposition",
                    detail="The team articulated the core enterprise problem with real customer data.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="pitch_content_and_evidence",
                )
            ],
            improvements=[
                Finding(
                    id="f_team_i1",
                    kind="improvement",
                    title="Expand Competitor Defense",
                    detail="Differentiate more sharply against legacy incumbent solutions.",
                    recommendation="Add a side-by-side feature matrix to Slide 4.",
                    evidence_ids=["ev_speech_002"],
                    rubric_dimension="business_and_problem_solution_reasoning",
                )
            ],
        ),
        member_feedback=[
            MemberFeedback(
                user_id="user_founder_01",
                display_name="Jane Founder",
                speaker_labels=["SPEAKER_00"],
                speaking_intervals=[
                    SpeakingInterval(start_ms=0, end_ms=135000, formatted="00:00 - 02:15"),
                    SpeakingInterval(start_ms=210000, end_ms=270000, formatted="03:30 - 04:30"),
                ],
                speaking_time_ms=195000,
                summary="Jane presented the company vision, customer problem, and financial model.",
                delivery_components=[
                    ScoreComponent(
                        dimension="delivery_and_body_language",
                        status="scored",
                        normalized_score=0.78,
                        display_score=78,
                        label="good",
                        configured_weight=0.15,
                    ),
                    ScoreComponent(
                        dimension="timing_and_speech_mechanics",
                        status="scored",
                        normalized_score=0.74,
                        display_score=74,
                        label="good",
                        configured_weight=0.05,
                    ),
                ],
                strengths=[
                    Finding(
                        id="f_jane_s1",
                        kind="strength",
                        title="Vocal Projection & Pacing",
                        detail="Maintained a clear and engaging speaking rate during the opening hook.",
                        evidence_ids=["ev_speech_001"],
                        speaker_labels=["SPEAKER_00"],
                    )
                ],
                improvements=[
                    Finding(
                        id="f_jane_i1",
                        kind="improvement",
                        title="Slide Transition Pauses",
                        detail="Transitioned between slides without pausing for comprehension.",
                        recommendation="Pause 2 seconds after advancing to each new slide.",
                        evidence_ids=["ev_speech_001"],
                        speaker_labels=["SPEAKER_00"],
                    )
                ],
            ),
            MemberFeedback(
                user_id="user_cto_02",
                display_name="Alex CTO",
                speaker_labels=["SPEAKER_01"],
                speaking_intervals=[
                    SpeakingInterval(start_ms=135000, end_ms=210000, formatted="02:15 - 03:30")
                ],
                speaking_time_ms=75000,
                summary="Alex presented the proprietary architecture and defensibility.",
                delivery_components=[
                    ScoreComponent(
                        dimension="delivery_and_body_language",
                        status="scored",
                        normalized_score=0.72,
                        display_score=72,
                        label="good",
                        configured_weight=0.15,
                    ),
                    ScoreComponent(
                        dimension="timing_and_speech_mechanics",
                        status="scored",
                        normalized_score=0.70,
                        display_score=70,
                        label="good",
                        configured_weight=0.05,
                    ),
                ],
                strengths=[
                    Finding(
                        id="f_alex_s1",
                        kind="strength",
                        title="Direct Eye Alignment",
                        detail="Consistently faced the camera while explaining technical components.",
                        speaker_labels=["SPEAKER_01"],
                    )
                ],
                improvements=[
                    Finding(
                        id="f_alex_i1",
                        kind="improvement",
                        title="Reduce Filler Word Clusters",
                        detail="Frequent 'um' fillers detected during the pipeline explanation.",
                        recommendation="Embrace silent pauses when formulating technical points.",
                        speaker_labels=["SPEAKER_01"],
                    )
                ],
            ),
        ],
        limitations=[
            Limitation(
                code="visual_partial_obstruction",
                scope="vision",
                message="Lower body posture was not tracked due to camera framing.",
            )
        ],
        reproducibility={
            "pipeline_version": "0.1.0",
            "evaluator_model": "openai/gpt-oss-120b",
            "scoring_engine": "startup_pitch_v1",
        },
    )


def _make_sample_qa_data() -> dict[str, Any]:
    return {
        "questions": [
            {
                "id": "q_01",
                "text": "What is your customer acquisition cost (CAC) payback period?",
                "rubric_dimension": "business_and_problem_solution_reasoning",
            },
            {
                "id": "q_02",
                "text": "How does your proprietary engine scale horizontally under peak load?",
                "rubric_dimension": "technical_feasibility",
            },
            {
                "id": "q_03",
                "text": "What are the primary churn drivers identified during the pilot?",
                "rubric_dimension": "pitch_content_and_evidence",
            },
        ],
        "answers": [
            {
                "question_id": "q_01",
                "answered_by": "Jane Founder",
                "status": "submitted",
                "transcript": "Our CAC is under $200 with a 6-month payback via partnership channels.",
            },
            {
                "question_id": "q_02",
                "answered_by": "Alex CTO",
                "status": "submitted",
                "transcript": "We utilize stateless worker nodes backed by async task queues.",
            },
            {
                "question_id": "q_03",
                "status": "skipped",
            },
        ],
        "assessments": [
            {
                "question_id": "q_01",
                "score": 0.85,
                "assessment_text": "Strong articulation of unit economics and channel efficiency.",
            },
            {
                "question_id": "q_02",
                "score": 0.80,
                "assessment_text": "Clear explanation of distributed architecture scalability.",
            },
            {
                "question_id": "q_03",
                "score": 0.0,
                "assessment_text": "Question was skipped by the team.",
            },
        ],
    }


def _make_sample_by_speaker() -> dict[str, Any]:
    return {
        "SPEAKER_00": {
            "speaking_time_ms": 195000,
            "intervals": [{"formatted": "00:00 - 02:15"}, {"formatted": "03:30 - 04:30"}],
            "windows": [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "acoustic": {
                        "speaking_rate_wpm": 142.0,
                        "pitch_mean_hz": 150.0,
                        "pause_duration_ms": 300,
                        "pause_count": 2,
                        "filler_count": 1,
                    },
                    "visual": {
                        "gaze_direction": 1.0,
                        "posture_openness": 0.85,
                        "upper_body_movement_px": 3.2,
                    },
                }
            ],
        },
        "SPEAKER_01": {
            "speaking_time_ms": 75000,
            "intervals": [{"formatted": "02:15 - 03:30"}],
            "windows": [
                {
                    "start_ms": 135000,
                    "end_ms": 145000,
                    "acoustic": {
                        "speaking_rate_wpm": 128.0,
                        "pitch_mean_hz": 120.0,
                        "pause_duration_ms": 500,
                        "pause_count": 3,
                        "filler_count": 4,
                    },
                    "visual": {
                        "gaze_direction": 0.95,
                        "posture_openness": 0.78,
                        "upper_body_movement_px": 2.1,
                    },
                }
            ],
        },
    }


def test_generate_markdown_report_comprehensive() -> None:
    """Verify markdown report generates all required sections, tables, timestamps, and metrics."""
    evaluation = _make_sample_evaluation()
    qa_data = _make_sample_qa_data()
    by_speaker = _make_sample_by_speaker()

    md = generate_markdown_report(
        evaluation,
        session_title="VirtuJudge Seed Pitch Practice",
        session_date="2026-09-14",
        qa_data=qa_data,
        by_speaker=by_speaker,
    )

    # 1. Header & Executive Summary
    assert "# VirtuJudge Seed Pitch Practice — Final Evaluation Report" in md
    assert "01JEVAL0000000000000000001" in md
    assert "`startup_pitch` (v1)" in md
    # 0.785 rounds to 79 -> Good (79/100)
    assert "Good (79/100)" in md
    assert "The team delivered a persuasive, evidence-grounded pitch" in md

    # 2. Score Breakdown Table
    assert "## Rubric Score Breakdown" in md
    assert "| Pitch Content & Evidence | 25% | 25% | 85 / 100 | Strong | `scored` |" in md
    assert "| Business & Problem-Solution Reasoning | 20% | 20% | 75 / 100 | Good | `scored` |" in md
    assert "| Q&A Quality | 20% | 20% | 80 / 100 | Strong | `scored` |" in md
    assert "### Dimension Rationales" in md
    assert "Compelling market definition" in md

    # 3. Whole-Team Assessment
    assert "## Team Pitch Assessment" in md
    assert "### Core Strengths" in md
    assert "**Grounded Value Proposition** (`ev_speech_001`)" in md
    assert "### Priority Areas for Improvement" in md
    assert "**Expand Competitor Defense** (`ev_speech_002`)" in md
    assert "*Recommendation:* Add a side-by-side feature matrix to Slide 4." in md

    # 4. Individual Presenter Delivery Feedback
    assert "## Individual Presenter Delivery Feedback" in md

    # Jane Founder
    assert "### Jane Founder (SPEAKER_00)" in md
    assert "⏱️ **Active Speaking Turns:** 00:00 - 02:15, 03:30 - 04:30 | **Total Speaking Time:** 3m 15s" in md
    assert "| Delivery & Body Language | 78 / 100 | Good |" in md
    assert "| Timing & Speech Mechanics | 74 / 100 | Good |" in md
    assert "#### Delivery Coaching" in md
    assert "**Pace:**" not in md
    assert "**Posture Openness:**" not in md
    assert "The overall pace was easy to follow" in md
    assert "**Vocal Projection & Pacing**" in md
    assert "**Slide Transition Pauses**" in md
    assert "Pause 2 seconds after advancing" in md

    # Alex CTO
    assert "### Alex CTO (SPEAKER_01)" in md
    assert "⏱️ **Active Speaking Turns:** 02:15 - 03:30 | **Total Speaking Time:** 1m 15s" in md
    assert "| Delivery & Body Language | 72 / 100 | Good |" in md
    assert "The delivery was slow enough to lose momentum" in md
    assert "**Direct Eye Alignment**" in md
    assert "**Reduce Filler Word Clusters**" in md

    # 5. Q&A Deep Dive (Team Level)
    assert "## Q&A Session Deep Dive" in md
    assert "### Question 1: What is your customer acquisition cost (CAC) payback period?" in md
    assert "**Answered By:** Jane Founder" in md
    assert 'Our CAC is under $200 with a 6-month payback' in md
    assert "Strong (85/100)" in md

    assert "### Question 2: How does your proprietary engine scale horizontally under peak load?" in md
    assert "**Answered By:** Alex CTO" in md
    assert 'We utilize stateless worker nodes' in md

    assert "### Question 3: What are the primary churn drivers identified during the pilot?" in md
    assert "⚠️ *Skipped by team* (Score: 0/100)" in md

    # System metadata does not belong in the reader-facing report.
    assert "## Appendix" not in md
    assert "[visual_partial_obstruction]" not in md
    assert "**Evaluator Model:**" not in md
    assert "**Scoring Engine:**" not in md


def test_generate_markdown_report_minimal_fallback() -> None:
    """Verify generator cleanly formats minimal Evaluation with zero exceptions."""
    minimal_eval = Evaluation(
        id="01JEVAL_MINIMAL",
        analysis_attempt_id="01JATTEMPT_MINIMAL",
        qa_round_id="01JROUND_MINIMAL",
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        overall_score=None,
        components=[],
        findings=[],
        team_feedback=FeedbackSection(summary=""),
        member_feedback=[],
        limitations=[],
        reproducibility={},
    )

    md = generate_markdown_report(minimal_eval)

    assert "# Startup Pitch Practice Session — Final Evaluation Report" in md
    assert "Not Rated" in md
    assert "*No mapped individual presenter feedback available.*" in md
    assert "*No Q&A exchange records available for this session.*" in md
    assert "## Appendix" not in md


def test_format_speaker_metrics_highlight_empty_and_populated() -> None:
    """Verify metric highlight helper handles empty and populated speaker profiles."""
    assert "No active delivery metrics" in format_speaker_metrics_highlight(None)[0]
    assert "No windowed observations" in format_speaker_metrics_highlight({"windows": []})[0]

    profile = {
        "windows": [
            {
                "acoustic": {
                    "speaking_rate_wpm": 150.0,
                    "pitch_mean_hz": 180.0,
                    "pause_duration_ms": 400,
                    "pause_count": 2,
                    "filler_count": 1,
                },
                "visual": {
                    "gaze_direction": 0.90,
                    "posture_openness": 0.80,
                    "upper_body_movement_px": 5.0,
                },
            },
            {
                "acoustic": {
                    "speaking_rate_wpm": 160.0,
                    "pitch_mean_hz": 190.0,
                    "pause_duration_ms": 600,
                    "pause_count": 3,
                    "filler_count": 2,
                },
                "visual": {
                    "gaze_direction": 1.0,
                    "posture_openness": 0.82,
                    "upper_body_movement_px": 4.0,
                },
            },
        ]
    }
    lines = format_speaker_metrics_highlight(profile)
    assert len(lines) == 3
    assert "Pacing" in lines[0]
    assert "easy to follow" in lines[0]
    assert "Fluency" in lines[1]
    assert "filler words" in lines[1]
    assert "Audience connection" in lines[2]
    assert not any("WPM" in line or "Hz" in line or "px" in line for line in lines)


def test_generate_markdown_report_speaker_boundary_isolation() -> None:
    """Verify each presenter's feedback is strictly compartmentalized under their own section."""
    evaluation = _make_sample_evaluation()
    md = generate_markdown_report(evaluation)

    # Jane's section should appear before Alex's section
    jane_idx = md.find("### Jane Founder (SPEAKER_00)")
    alex_idx = md.find("### Alex CTO (SPEAKER_01)")
    assert jane_idx != -1
    assert alex_idx != -1
    assert jane_idx < alex_idx

    jane_section = md[jane_idx:alex_idx]
    alex_section = md[alex_idx:]

    # Jane's section contains Jane's turns and findings only
    assert "00:00 - 02:15" in jane_section
    assert "Vocal Projection & Pacing" in jane_section
    assert "02:15 - 03:30" not in jane_section
    assert "Direct Eye Alignment" not in jane_section

    # Alex's section contains Alex's turns and findings only
    assert "02:15 - 03:30" in alex_section
    assert "Direct Eye Alignment" in alex_section
    assert "00:00 - 02:15" not in alex_section
    assert "Vocal Projection & Pacing" not in alex_section


def test_generate_markdown_report_question_id_candidate_id_lookup() -> None:
    """Verify lookup succeeds when questions, answers, or assessments use
    question_id or candidate_id.
    """
    evaluation = _make_sample_evaluation()
    qa_data = {
        "questions": [
            {
                "question_id": "qid_via_key",
                "text": "How do you defend against fast followers?",
                "rubric_dimension": "technology_and_moat",
            },
            {
                "candidate_id": "cid_via_key",
                "question_text": "What is your gross margin profile?",
                "rubric_dimension": "market_and_business_model",
            },
        ],
        "answers": [
            {
                "question_id": "qid_via_key",
                "answered_by": "Jane Founder",
                "status": "submitted",
                "transcript": "We patent our feature extraction and model orchestration pipelines.",
            },
            {
                "candidate_id": "cid_via_key",
                "answered_by": "Alex CTO",
                "status": "submitted",
                "transcript": "Our gross margin is 82% at current cloud volume.",
            },
        ],
        "assessments": [
            {
                "question_id": "qid_via_key",
                "score": 0.88,
                "assessment_text": "Convincing IP protection and defensibility moat.",
            },
            {
                "candidate_id": "cid_via_key",
                "score": 0.84,
                "assessment_text": "Clear software-like margin economics.",
            },
        ],
    }

    md = generate_markdown_report(evaluation, qa_data=qa_data)

    assert "### Question 1: How do you defend against fast followers?" in md
    assert "Jane Founder" in md
    assert "We patent our feature extraction" in md
    assert "Convincing IP protection and defensibility moat." in md

    assert "### Question 2: What is your gross margin profile?" in md
    assert "Alex CTO" in md
    assert "Our gross margin is 82% at current cloud volume." in md
    assert "Clear software-like margin economics." in md
