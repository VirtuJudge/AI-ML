"""Unit tests for app.stages.reporting (Report Stage Orchestration)."""

from unittest.mock import AsyncMock

import pytest

from app.contracts import (
    ArtifactRef,
    Evaluation,
    SpeakerMapping,
    SpeakingInterval,
)
from app.providers.fake_judge import FakeJudgeModelProvider
from app.stages.report_loader import (
    ReportEvidenceBundle,
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
)
from app.stages.reporting import ReportStageResult, run_report_stage
from app.storage.base import ObjectStorageProtocol

FAKE_SHA256 = "sha256:" + "a" * 64


@pytest.mark.asyncio
async def test_run_report_stage_with_synthetic_defaults() -> None:
    """Verify end-to-end report stage execution with synthetic fallbacks."""
    analysis_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000001",
        object_key="artifacts/analysis.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000002",
        object_key="artifacts/qa.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )

    result = await run_report_stage(
        report_id="01JREPORT00000000000000001",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=None,
        judge_provider=FakeJudgeModelProvider(),
        session_title="VirtuJudge Synthetic Practice",
    )

    assert isinstance(result, ReportStageResult)
    eval_model = result.evaluation
    assert isinstance(eval_model, Evaluation)

    # Verify overall score and components
    assert eval_model.overall_score is not None
    assert 0.0 <= eval_model.overall_score <= 1.0
    assert len(eval_model.components) == 6

    # Verify effective weights sum to 1.0
    eff_weights = [
        c.effective_weight for c in eval_model.components if c.effective_weight is not None
    ]
    assert pytest.approx(sum(eff_weights), rel=1e-3) == 1.0

    # Verify Q&A quality component
    qa_comp = next((c for c in eval_model.components if c.dimension == "qa_quality"), None)
    assert qa_comp is not None
    assert qa_comp.configured_weight == 0.20
    assert qa_comp.status == "scored"
    assert qa_comp.display_score is not None

    # Verify every scored component has at least one evidence_id (Data-Contracts.md:330)
    scored_comps = [c for c in eval_model.components if c.status == "scored"]
    assert all(len(c.evidence_ids) >= 1 for c in scored_comps)

    # Verify reproducibility metadata has all required contract fields (Data-Contracts.md:506)
    required_repro_fields = [
        "pipeline_version",
        "rubric_id",
        "rubric_version",
        "prompt_versions",
        "contract_versions",
        "input_checksums",
        "stage_versions",
        "model_runs",
        "started_at",
        "completed_at",
        "duration_ms",
        "cost_summary",
    ]
    for rf in required_repro_fields:
        assert rf in eval_model.reproducibility

    # Verify member feedback
    assert len(eval_model.member_feedback) >= 1
    for mf in eval_model.member_feedback:
        assert len(mf.user_id) > 0
        assert len(mf.display_name) > 0
        assert len(mf.speaker_labels) >= 1
        assert mf.speaking_time_ms >= 0
        assert len(mf.delivery_components) == 2
        for dc in mf.delivery_components:
            if dc.status == "scored":
                assert len(dc.evidence_ids) >= 1
        assert len(mf.strengths) >= 1
        assert len(mf.improvements) >= 1
        for inv in mf.speaking_intervals:
            assert isinstance(inv, SpeakingInterval)
            assert " - " in inv.formatted

    # Verify user IDs
    assert len(result.member_feedback_user_ids) == len(eval_model.member_feedback)

    # Verify markdown report
    md = result.report_markdown
    assert "# VirtuJudge Synthetic Practice — Final Evaluation Report" in md
    assert "## Rubric Score Breakdown" in md
    assert "## Team Pitch Assessment" in md
    assert "## Individual Presenter Delivery Feedback" in md
    assert "## Q&A Session Deep Dive" in md
    assert "## Appendix" not in md


@pytest.mark.asyncio
async def test_run_report_stage_with_explicit_speaker_mappings() -> None:
    """Verify mapped speaker labels bind to user identities and display names."""
    analysis_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000003",
        object_key="artifacts/analysis.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000004",
        object_key="artifacts/qa.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )

    speaker_mappings = [
        SpeakerMapping(
            speaker_label="SPEAKER_00",
            user_id="user_founder_jane",
            display_name="Jane Founder",
        )
    ]

    result = await run_report_stage(
        report_id="01JREPORT00000000000000002",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        speaker_mappings=speaker_mappings,
        judge_provider=FakeJudgeModelProvider(),
    )

    assert len(result.evaluation.member_feedback) == 1
    mbr = result.evaluation.member_feedback[0]
    assert mbr.user_id == "user_founder_jane"
    assert mbr.display_name == "Jane Founder"
    assert mbr.speaker_labels == ["SPEAKER_00"]
    assert result.member_feedback_user_ids == ["user_founder_jane"]

    # In markdown
    assert "### Jane Founder (SPEAKER_00)" in result.report_markdown


@pytest.mark.asyncio
async def test_run_report_stage_with_skipped_questions() -> None:
    """Verify skipped Q&A questions score 0.0 and are formatted with warning."""
    bundle = ReportEvidenceBundle(
        session_id="session_skip_test",
        transcript_full_text="Test pitch presentation.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 30000,
                "intervals": [{"start_ms": 0, "end_ms": 30000, "formatted": "00:00 - 00:30"}],
                "windows": [],
            }
        },
        questions=[
            {"id": "q1", "text": "Question 1?", "rubric_dimension": "pitch_content_and_evidence"},
            {"id": "q2", "text": "Question 2?", "rubric_dimension": "technical_feasibility"},
        ],
        answers=[
            {"question_id": "q1", "status": "submitted", "transcript": "Detailed answer."},
            {"question_id": "q2", "status": "skipped"},
        ],
        assessments=[
            {"question_id": "q1", "score": 0.80, "assessment_text": "Good answer."},
        ],
    )

    analysis_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000005",
        object_key="artifacts/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JSYNTH000000000000000006",
        object_key="artifacts/qa.json",
        checksum=FAKE_SHA256,
    )

    result = await run_report_stage(
        report_id="01JREPORT_SKIP_01",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
    )

    qa_comp = next((c for c in result.evaluation.components if c.dimension == "qa_quality"), None)
    assert qa_comp is not None
    # 1 answered (0.80) + 1 skipped (0.0) -> average 0.40 -> display 40
    assert qa_comp.normalized_score == 0.40
    assert qa_comp.display_score == 40
    assert qa_comp.configured_weight == 0.20

    # Markdown checks
    assert "⚠️ *Skipped by team*" in result.report_markdown


@pytest.mark.asyncio
async def test_run_report_stage_with_mock_storage() -> None:
    """Verify report stage reads from object storage when available."""
    mock_storage = AsyncMock(spec=ObjectStorageProtocol)
    raw_analysis = build_synthetic_analysis_data(session_id="storage_sess_01")
    raw_qa = build_synthetic_qa_data(session_id="storage_sess_01")

    async def mock_read_json(key: str) -> dict:
        if "analysis" in key:
            return raw_analysis
        return raw_qa

    mock_storage.read_json.side_effect = mock_read_json

    analysis_ref = ArtifactRef(
        artifact_id="01JSTORAGE0000000000000001",
        object_key="artifacts/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JSTORAGE0000000000000002",
        object_key="artifacts/qa.json",
        checksum=FAKE_SHA256,
    )

    result = await run_report_stage(
        report_id="01JSTORAGE_REPORT",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=mock_storage,
        judge_provider=FakeJudgeModelProvider(),
    )

    assert result.evaluation.analysis_attempt_id == "storage_sess_01"
    assert mock_storage.read_json.call_count == 2


@pytest.mark.asyncio
async def test_run_report_stage_missing_modalities_produce_not_evaluated() -> None:
    """Verify missing audio/visual observations produce not_evaluated, not invented scores."""
    bundle = ReportEvidenceBundle(
        session_id="session_missing_modalities",
        transcript_full_text="Founder pitch without video or audio observations.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 30000,
                "intervals": [{"start_ms": 0, "end_ms": 30000, "formatted": "00:00 - 00:30"}],
                "windows": [],
            }
        },
        questions=[],
        answers=[],
        assessments=[],
    )

    analysis_ref = ArtifactRef(
        artifact_id="01JMISSING0000000000000001",
        object_key="artifacts/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JMISSING0000000000000002",
        object_key="artifacts/qa.json",
        checksum=FAKE_SHA256,
    )

    result = await run_report_stage(
        report_id="01JREPORT_MISSING_01",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
    )

    eval_model = result.evaluation
    deliv_comp = next(
        c for c in eval_model.components if c.dimension == "delivery_and_body_language"
    )
    timing_comp = next(
        c for c in eval_model.components if c.dimension == "timing_and_speech_mechanics"
    )
    qa_comp = next(c for c in eval_model.components if c.dimension == "qa_quality")

    assert deliv_comp.status == "not_evaluated"
    assert deliv_comp.normalized_score is None
    assert deliv_comp.effective_weight is None

    assert timing_comp.status == "not_evaluated"
    assert timing_comp.normalized_score is None
    assert timing_comp.effective_weight is None

    assert qa_comp.status == "not_evaluated"
    assert qa_comp.normalized_score is None
    assert qa_comp.effective_weight is None

    # Remaining scored dimensions are normalized to sum to 1.0
    scored_comps = [c for c in eval_model.components if c.status == "scored"]
    assert len(scored_comps) == 3
    eff_sum = sum(c.effective_weight for c in scored_comps if c.effective_weight is not None)
    assert pytest.approx(eff_sum, rel=1e-3) == 1.0


@pytest.mark.asyncio
async def test_run_report_stage_zero_turn_presenter_has_no_delivery_feedback() -> None:
    """Presenter coaching requires a recorded turn or delivery observation."""
    bundle = ReportEvidenceBundle(
        session_id="session_no_delivery_data",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 0,
                "intervals": [],
                "windows": [],
            }
        },
    )

    result = await run_report_stage(
        report_id="01JREPORT_NO_DELIVERY_01",
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
    )

    presenter = result.evaluation.member_feedback[0]
    assert presenter.strengths == []
    assert presenter.improvements == []
    assert "No delivery data was recorded" in presenter.summary
