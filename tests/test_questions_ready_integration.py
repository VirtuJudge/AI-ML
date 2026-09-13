"""End-to-end integration tests for AI-05 (Three Grounded Primary Questions).

Validates:
1. Exactly three primary questions returned for any valid presentation.
2. Every question has a valid rubric dimension.
3. Every question references real evidence from speech or documents.
4. Presentation-only pitch (no supporting documents) completes with 3 grounded questions.
5. Presentation with missing vision (audio-only) completes with limitations and 3 questions.
6. Question panel validation deduplicates and fills to maintain exactly 3 questions.
7. process_job(analyze_session) returns SessionAnalysisCompleted conforming to schema.
8. Document chunks persisted in document store while questions delivered via backend update.
"""

from datetime import UTC, datetime

import pytest
import ulid

from app.backend_client import FakeBackendClient
from app.contracts import (
    AnalyzeSessionPayload,
    AssetInput,
    JobType,
    PrimaryQuestion,
    QueueMessage,
    RubricRef,
    SessionAnalysisCompleted,
    UpdateStatus,
)
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.fake_vision import FakeVisionProvider
from app.stages.evidence import build_evidence_bundle
from app.stages.judge_panel import (
    JUDGE_PANEL,
    validate_panel_questions,
)
from app.worker import process_job


@pytest.fixture
def base_session_payload() -> AnalyzeSessionPayload:
    return AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTESTART000000000000000001",
            object_key="presentations/test.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
            duration_ms=15000,
        ),
        supporting_documents=[
            AssetInput(
                artifact_id="01JTESTDOC000000000000000001",
                object_key="tests/fixtures/documents/sample_2page.pdf",
                checksum="sha256:" + "b" * 64,
                media_type="application/pdf",
            )
        ],
        rubric=RubricRef(
            rubric_id="startup_pitch",
            version=1,
        ),
    )


@pytest.mark.asyncio
async def test_exactly_three_questions_with_valid_dimensions(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 1 & 2: Pipeline produces exactly 3 questions with standard dimensions."""
    pipeline = FakePipeline()
    result = await pipeline.analyze_session(base_session_payload)

    assert isinstance(result, SessionAnalysisCompleted)
    assert len(result.primary_questions) == 3

    dimensions = [q.rubric_dimension for q in result.primary_questions]
    assert len(set(dimensions)) == 3

    valid_dimensions = {
        "market_and_business_model",
        "technology_and_moat",
        "execution_and_milestones",
        "business_reasoning",
        "technical_feasibility",
        "pitch_content_and_evidence",
        "customer_retention",
    }
    for dim in dimensions:
        assert dim in valid_dimensions


@pytest.mark.asyncio
async def test_questions_reference_real_evidence_ids(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 3: Every question links to at least one valid evidence item."""
    pipeline = FakePipeline()
    result = await pipeline.analyze_session(base_session_payload)

    for q in result.primary_questions:
        assert len(q.evidence_ids) >= 1
        for ev_id in q.evidence_ids:
            assert ev_id.startswith("ev_speech_") or ev_id.startswith("ev_doc_")


@pytest.mark.asyncio
async def test_presentation_only_without_documents(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 4: Presentation-only pitch (0 documents) produces 3 grounded questions."""
    no_docs_payload = base_session_payload.model_copy(
        update={"supporting_documents": []}
    )
    pipeline = FakePipeline()
    result = await pipeline.analyze_session(no_docs_payload)

    assert len(result.primary_questions) == 3
    # All evidence linked should be speech
    for q in result.primary_questions:
        for ev_id in q.evidence_ids:
            assert ev_id.startswith("ev_speech_")


@pytest.mark.asyncio
async def test_presentation_with_missing_vision(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 5: Missing vision produces limitations and 3 valid primary questions."""
    class EmptyVisionProvider(FakeVisionProvider):
        async def analyze_video(self, video_path: object) -> list:
            return []

    pipeline = FakePipeline(vision_provider=EmptyVisionProvider())
    result = await pipeline.analyze_session(base_session_payload)

    assert len(result.primary_questions) == 3
    has_vision_lim = any(
        lim.scope == "vision" or "visual" in lim.message.lower() for lim in result.limitations
    )
    assert has_vision_lim


@pytest.mark.asyncio
async def test_duplicate_question_deduplication_and_filling() -> None:
    """Invariant 6: Duplicate judge questions are deduplicated and filled to exactly 3."""
    bundle = build_evidence_bundle(transcript="We sell software to hospitals.")
    dup_q1 = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is your sales cycle length?",
        reason="Market evaluation.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )
    dup_q2 = PrimaryQuestion(
        candidate_id=ulid.new().str,
        text="What is your sales cycle length?",
        reason="Market evaluation duplicate.",
        rubric_dimension="market_and_business_model",
        evidence_ids=["ev_speech_001"],
    )

    validated_questions, limitations = validate_panel_questions(
        [dup_q1, dup_q2],
        bundle,
        judges=JUDGE_PANEL,
    )

    assert len(validated_questions) == 3
    assert len({q.text for q in validated_questions}) == 3
    assert any(lim.code == "duplicate_question_filtered" for lim in limitations)


@pytest.mark.asyncio
async def test_process_job_e2e_with_backend_client(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 7: process_job dispatches STARTED and COMPLETED updates via backend client."""
    pipeline = FakePipeline()
    backend_client = FakeBackendClient()

    message = QueueMessage(
        schema_version=1,
        job_id="01JTESTJOB00000000000000001",
        job_type=JobType.ANALYZE_SESSION,
        practice_session_id="01JTESTSESSION000000000001",
        analysis_attempt=1,
        created_at=datetime.now(UTC),
        trace_id="01JTESTTRACE00000000000001",
        payload=base_session_payload.model_dump(mode="json"),
    )

    update = await process_job(message, pipeline, backend_client)

    assert update.status == UpdateStatus.COMPLETED
    assert update.sequence == 2
    assert isinstance(update.payload, SessionAnalysisCompleted)
    assert len(update.payload.primary_questions) == 3

    # Verify backend client received sequence=1 (STARTED) and sequence=2 (COMPLETED)
    assert len(backend_client.updates) == 2
    assert backend_client.updates[0].status == UpdateStatus.STARTED
    assert backend_client.updates[0].sequence == 1
    assert backend_client.updates[1].status == UpdateStatus.COMPLETED
    assert backend_client.updates[1].sequence == 2


@pytest.mark.asyncio
async def test_document_store_persists_chunks(
    base_session_payload: AnalyzeSessionPayload,
) -> None:
    """Invariant 8: Document store stores chunk vectors during session analysis."""
    doc_store = FakeDocumentStore()
    pipeline = FakePipeline(document_store=doc_store)

    result = await pipeline.analyze_session(base_session_payload)

    assert len(result.primary_questions) == 3
    session_id = base_session_payload.presentation.artifact_id
    stored_chunks = doc_store._store.get(session_id, [])
    assert len(stored_chunks) > 0
