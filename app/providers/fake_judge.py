"""Fake judge model provider generating grounded evaluation questions."""

from app.contracts import PrimaryQuestion
from app.providers.types import DocumentChunk


class FakeJudgeModelProvider:
    """Deterministic LLM judge provider for question generation."""

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
            if doc_ids:
                doc_id = doc_ids[0]
            else:
                doc_id = speech_id

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


__all__ = ["FakeJudgeModelProvider"]
