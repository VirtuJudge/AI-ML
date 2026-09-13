"""Fake judge model provider generating grounded evaluation questions and assessments."""

from app.contracts import FollowUpQuestion, PrimaryQuestion
from app.providers.base import AnswerAssessment
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
            evidence_ids=evidence_ids,
            follow_up=follow_up,
        )


__all__ = ["FakeJudgeModelProvider"]
