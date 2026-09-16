from typing import Any

from app.contracts import Finding, FollowUpQuestion, PrimaryQuestion
from app.providers.base import AnswerAssessment, ReportFeedbackResult
from app.providers.types import DocumentChunk


class FakeJudgeModelProvider:
    """Deterministic LLM judge provider for question generation and answer assessment."""

    async def generate_questions(
        self,
        transcript: str,
        rubric_id: str,
        document_chunks: list[DocumentChunk] | None = None,
        evidence_bundle: object = None,
    ) -> list[PrimaryQuestion]:
        speech_id = "ev_speech_001"
        doc_id = "ev_doc_slide_01"

        if hasattr(evidence_bundle, "items") and evidence_bundle.items:
            speech_ids = [i.evidence_id for i in evidence_bundle.items if i.source == "speech"]
            doc_ids = [i.evidence_id for i in evidence_bundle.items if i.source == "documents"]
            if speech_ids:
                speech_id = speech_ids[0]
            doc_id = doc_ids[0] if doc_ids else speech_id

        return [
            PrimaryQuestion(
                candidate_id="01JEXAMPLE000000000000001A",
                text="What is your projected customer acquisition cost at scale, "
                "and what channels drive that estimate?",
                reason="Validates financial feasibility and go-to-market model assumptions.",
                rubric_dimension="market_and_business_model",
                evidence_ids=[speech_id],
            ),
            PrimaryQuestion(
                candidate_id="01JEXAMPLE000000000000001B",
                text="How does your core proprietary technology maintain its defensive moat "
                "against incumbent fast-followers?",
                reason="Assesses technical defensibility and differentiation.",
                rubric_dimension="technology_and_moat",
                evidence_ids=[speech_id],
            ),
            PrimaryQuestion(
                candidate_id="01JEXAMPLE000000000000001C",
                text="What specific milestones must be achieved during the initial pilot phase "
                "to secure renewal commitments?",
                reason="Evaluates execution roadmap and early customer validation.",
                rubric_dimension="execution_and_milestones",
                evidence_ids=[doc_id],
            ),
        ]

    async def assess_answer(
        self,
        answer_transcript: str,
        question_text: str,
        rubric_dimension: str,
        *,
        remaining_follow_ups: int = 0,
    ) -> AnswerAssessment:
        """Deterministic assessment of a team member's answer."""
        assessment_text = (
            f"The team member's response directly addresses the question '{question_text}' "
            f"regarding dimension '{rubric_dimension}' with substantive reasoning and "
            "operational clarity."
        )
        evidence_ids: list[str] = []

        is_comprehensive = any(
            term in answer_transcript.lower()
            for term in ["comprehensive", "detailed", "thorough", "exhaustive", "complete"]
        )

        follow_up: FollowUpQuestion | None = None
        if remaining_follow_ups > 0 and answer_transcript.strip() and not is_comprehensive:
            follow_up = FollowUpQuestion(
                text=(
                    "Could you elaborate on the key assumptions underlying your approach "
                    f"to {rubric_dimension}?"
                ),
                reason=(
                    "Clarifies operational trade-offs identified in the answer regarding "
                    f"{rubric_dimension}."
                ),
                rubric_dimension=rubric_dimension,
                evidence_ids=[],
            )

        return AnswerAssessment(
            assessment_text=assessment_text,
            score=0.85 if is_comprehensive else 0.65,
            evidence_ids=evidence_ids,
            follow_up=follow_up,
        )

    async def generate_report_feedback(
        self,
        *,
        transcript_summary: str,
        qa_summary: str,
        speaker_profiles: dict[str, Any],
        rubric_id: str = "startup_pitch",
    ) -> ReportFeedbackResult:
        """Deterministic evaluation feedback for final report generation."""
        member_strengths: dict[str, list[Finding]] = {}
        member_improvements: dict[str, list[Finding]] = {}

        for spk in sorted(speaker_profiles.keys()):
            member_strengths[spk] = [
                Finding(
                    id=f"f_{spk.lower()}_s1",
                    kind="strength",
                    title="Grounded Pacing & Delivery",
                    detail="Maintained steady vocal delivery and structured pacing throughout active turns.",
                    recommendation="Continue using natural pauses to emphasize core points.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="delivery_and_body_language",
                    speaker_labels=[spk],
                )
            ]
            member_improvements[spk] = [
                Finding(
                    id=f"f_{spk.lower()}_i1",
                    kind="improvement",
                    title="Reduce Filler Words During Transitions",
                    detail="Occasional filler words detected during slide and topic transitions.",
                    recommendation="Pause intentionally for 1-2 seconds between ideas instead of using fillers.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="timing_and_speech_mechanics",
                    speaker_labels=[spk],
                )
            ]

        if not member_strengths:
            member_strengths["SPEAKER_00"] = [
                Finding(
                    id="f_spk0_s1",
                    kind="strength",
                    title="Clear Articulation",
                    detail="Spoke clearly with well-paced delivery across presentation sections.",
                    rubric_dimension="delivery_and_body_language",
                    speaker_labels=["SPEAKER_00"],
                )
            ]
            member_improvements["SPEAKER_00"] = [
                Finding(
                    id="f_spk0_i1",
                    kind="improvement",
                    title="Slide Transition Pauses",
                    detail="Transitions between topics were somewhat abrupt.",
                    recommendation="Use 2-second deliberate pauses when moving to new slides.",
                    rubric_dimension="timing_and_speech_mechanics",
                    speaker_labels=["SPEAKER_00"],
                )
            ]

        return ReportFeedbackResult(
            executive_summary=(
                "The team demonstrated solid problem-solution alignment and clear technical architecture "
                "during the pitch. Delivery was well-paced with minor opportunities to sharpen Q&A "
                "conciseness and visual engagement."
            ),
            dimension_scores={
                "pitch_content_and_evidence": 0.85,
                "business_and_problem_solution_reasoning": 0.80,
                "technical_feasibility": 0.82,
            },
            dimension_rationales={
                "pitch_content_and_evidence": "Clear articulation of market opportunity with evidence-backed claims.",
                "business_and_problem_solution_reasoning": "Substantiated market assumptions and business model viability.",
                "technical_feasibility": "Realistic architecture moat and clear scalability roadmap.",
            },
            team_strengths=[
                Finding(
                    id="f_team_s1",
                    kind="strength",
                    title="Strong Problem-Solution Articulation",
                    detail="The presentation clearly defined the customer pain point and demonstrated why the solution is uniquely defensible.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="pitch_content_and_evidence",
                ),
                Finding(
                    id="f_team_s2",
                    kind="strength",
                    title="Rigorous Q&A Objections Handling",
                    detail="The team provided concrete data points and unit economics when responding to challenging technical inquiries.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="qa_quality",
                ),
            ],
            team_improvements=[
                Finding(
                    id="f_team_i1",
                    kind="improvement",
                    title="Deepen Competitor Differentiation",
                    detail="The competitive landscape slide lacked granular differentiation against legacy incumbents.",
                    recommendation="Include a clear 2x2 matrix or feature comparison highlighting proprietary barriers.",
                    evidence_ids=["ev_speech_001"],
                    rubric_dimension="business_and_problem_solution_reasoning",
                )
            ],
            member_strengths=member_strengths,
            member_improvements=member_improvements,
            recommendations=[
                "Explicitly quantify market size and serviceable obtainable market (SOM) on Slide 3.",
                "Rehearse Q&A handoffs between founders to ensure immediate, concise responses.",
            ],
        )


__all__ = ["FakeJudgeModelProvider"]
