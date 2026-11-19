"""Unit tests for app.stages.reporting (Report Stage Orchestration)."""

import hashlib
import json
from pathlib import Path
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
from app.stages.report_markdown import SCORE_LABEL_DISPLAY
from app.stages.reporting import ReportStageResult, run_report_stage
from app.stages.scoring import WEIGHTING_POLICY_VERSION, score_to_label
from app.storage.base import ObjectStorageProtocol
from app.storage.local import LocalDiskObjectStorage
from tests.conftest import load_json_fixture

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
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="01JSYNTHUSER00000000000001",
                display_name="Presenter One",
            )
        ],
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
        "weighting_policy_version",
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
    assert eval_model.reproducibility["weighting_policy_version"] == WEIGHTING_POLICY_VERSION

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
    """Verify report stage reads and checksum-verifies objects from object storage."""
    mock_storage = AsyncMock(spec=ObjectStorageProtocol)
    raw_analysis = build_synthetic_analysis_data(session_id="storage_sess_01")
    raw_qa = build_synthetic_qa_data(session_id="storage_sess_01")
    analysis_bytes = json.dumps(raw_analysis).encode("utf-8")
    qa_bytes = json.dumps(raw_qa).encode("utf-8")

    async def mock_download_file(key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(analysis_bytes if "analysis" in key else qa_bytes)
        return destination

    mock_storage.download_file.side_effect = mock_download_file

    analysis_ref = ArtifactRef(
        artifact_id="01JSTORAGE0000000000000001",
        object_key="artifacts/analysis.json",
        checksum="sha256:" + hashlib.sha256(analysis_bytes).hexdigest(),
    )
    qa_ref = ArtifactRef(
        artifact_id="01JSTORAGE0000000000000002",
        object_key="artifacts/qa.json",
        checksum="sha256:" + hashlib.sha256(qa_bytes).hexdigest(),
    )

    result = await run_report_stage(
        report_id="01JSTORAGE_REPORT",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=mock_storage,
        judge_provider=FakeJudgeModelProvider(),
    )

    assert result.evaluation.analysis_attempt_id == "storage_sess_01:attempt_1"
    assert result.evaluation.qa_round_id == "storage_sess_01:round_1"
    assert mock_storage.download_file.call_count == 2


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
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="01JUSER_NO_DELIVERY_01",
                display_name="Zero Turn Presenter",
            )
        ],
    )

    presenter = result.evaluation.member_feedback[0]
    assert presenter.strengths == []
    assert presenter.improvements == []
    assert "No delivery data was recorded" in presenter.summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing_modality", "windows", "omit_technical_feasibility", "expected_not_evaluated"),
    [
        (
            "vision",
            [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "acoustic": {
                        "speaking_rate_wpm": 140.0,
                        "pause_duration_ms": 200,
                        "filler_count": 0,
                    },
                }
            ],
            False,
            {"delivery_and_body_language"},
        ),
        (
            "audio",
            [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "visual": {
                        "gaze_direction": 1.0,
                        "posture_openness": 0.85,
                        "shoulder_symmetry_ratio": 0.96,
                    },
                }
            ],
            False,
            {"timing_and_speech_mechanics"},
        ),
        (
            "video_and_audio",
            [],
            False,
            {"delivery_and_body_language", "timing_and_speech_mechanics"},
        ),
        (
            "documents",
            [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "acoustic": {
                        "speaking_rate_wpm": 140.0,
                        "pause_duration_ms": 200,
                        "filler_count": 0,
                    },
                    "visual": {
                        "gaze_direction": 1.0,
                        "posture_openness": 0.85,
                        "shoulder_symmetry_ratio": 0.96,
                    },
                }
            ],
            True,
            {"technical_feasibility"},
        ),
    ],
)
async def test_run_report_stage_keeps_qa_at_20_percent_when_optional_modality_missing(
    missing_modality: str,
    windows: list[dict[str, object]],
    omit_technical_feasibility: bool,
    expected_not_evaluated: set[str],
) -> None:
    """Q&A effective weight stays 0.20 and remaining 0.80 redistributes for any missing modality."""

    class PartialDimensionJudge(FakeJudgeModelProvider):
        async def generate_report_feedback(self, **kwargs):  # type: ignore[override]
            res = await super().generate_report_feedback(**kwargs)
            if omit_technical_feasibility:
                res.dimension_scores = {
                    k: v for k, v in res.dimension_scores.items() if k != "technical_feasibility"
                }
            return res

    bundle = ReportEvidenceBundle(
        session_id=f"session_missing_{missing_modality}",
        transcript_full_text="Founder pitch with partial modalities.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 30000,
                "intervals": [{"start_ms": 0, "end_ms": 30000, "formatted": "00:00 - 00:30"}],
                "windows": windows,
            }
        },
        questions=[
            {"id": "q1", "text": "Question 1?", "rubric_dimension": "pitch_content_and_evidence"},
        ],
        answers=[
            {"question_id": "q1", "status": "submitted", "transcript": "Detailed answer."},
        ],
        assessments=[
            {"question_id": "q1", "score": 0.50, "assessment_text": "Adequate answer."},
        ],
    )

    result = await run_report_stage(
        report_id=f"01JREPORT_MODALITY_{missing_modality.upper()}",
        evidence_bundle_override=bundle,
        judge_provider=PartialDimensionJudge(),
    )

    eval_model = result.evaluation
    assert eval_model.reproducibility["weighting_policy_version"] == WEIGHTING_POLICY_VERSION

    by_dim = {c.dimension: c for c in eval_model.components}
    for dim in expected_not_evaluated:
        assert by_dim[dim].status == "not_evaluated"
        assert by_dim[dim].effective_weight is None

    qa_comp = by_dim["qa_quality"]
    assert qa_comp.status == "scored"
    assert qa_comp.effective_weight == 0.20

    non_qa_scored = [
        c
        for c in eval_model.components
        if c.status == "scored" and c.dimension != "qa_quality" and c.effective_weight is not None
    ]
    assert pytest.approx(sum(c.effective_weight or 0.0 for c in non_qa_scored), abs=1e-3) == 0.80
    assert (
        pytest.approx(
            sum(c.effective_weight or 0.0 for c in eval_model.components if c.status == "scored"),
            abs=1e-3,
        )
        == 1.0
    )


async def test_run_report_stage_backend_shaped_qa_artifact_scores_mean(tmp_path) -> None:
    """Backend-shaped qa.json with 3 valid assessments yields qa_quality equal to mean score."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    qa_payload = load_json_fixture("ai/qa_artifact_valid.json")
    session_id = qa_payload["practice_session_id"]

    analysis_ref = await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        build_synthetic_analysis_data(session_id=session_id),
        artifact_id="01JANALYSIS_BACKEND_QA_01",
    )
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/qa.json",
        qa_payload,
        artifact_id=qa_payload["artifact_id"],
    )

    scores = [0.85, 0.72, 0.68]  # mean = 0.75
    for idx, (ans, score) in enumerate(zip(qa_payload["answers"], scores, strict=True)):
        await storage.upload_json(
            f"ai/session/{session_id}/answers/{ans['id']}/assessment.json",
            {
                "artifact_id": ans["assessment_artifact_id"],
                "answer_id": ans["id"],
                "practice_session_id": session_id,
                "question_id": ans["question_id"],
                "kind": "answer_assessment",
                "schema_version": 1,
                "assessment": {
                    "text": qa_payload["assessments"][idx]["assessment_text"],
                    "score": score,
                    "evidence_ids": qa_payload["assessments"][idx]["evidence_ids"],
                },
                "limitations": [],
            },
            artifact_id=ans["assessment_artifact_id"],
        )

    result = await run_report_stage(
        report_id="01JREPORT_BACKEND_QA_01",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=storage,
        judge_provider=FakeJudgeModelProvider(),
    )

    qa_comp = next(c for c in result.evaluation.components if c.dimension == "qa_quality")
    assert qa_comp.status == "scored"
    assert qa_comp.normalized_score == 0.75
    assert qa_comp.display_score == 75
    assert qa_comp.label == "good"
    assert qa_comp.configured_weight == 0.20
    assert qa_comp.effective_weight == 0.20


@pytest.mark.asyncio
async def test_run_report_stage_unrecoverable_qa_scores_become_not_evaluated(tmp_path) -> None:
    """Answered questions with no recoverable score become not_evaluated with limitation."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    qa_payload = load_json_fixture("ai/qa_artifact_valid.json")
    session_id = qa_payload["practice_session_id"]

    analysis_ref = await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        build_synthetic_analysis_data(session_id=session_id),
        artifact_id="01JANALYSIS_MISSING_ASMT_01",
    )
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/qa.json",
        qa_payload,
        artifact_id=qa_payload["artifact_id"],
    )

    result = await run_report_stage(
        report_id="01JREPORT_MISSING_ASMT_01",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=storage,
        judge_provider=FakeJudgeModelProvider(),
    )

    qa_comp = next(c for c in result.evaluation.components if c.dimension == "qa_quality")
    assert qa_comp.status == "not_evaluated"
    assert qa_comp.normalized_score is None
    assert qa_comp.display_score is None
    assert qa_comp.effective_weight is None
    assert qa_comp.limitation_code == "qa_assessment_unavailable"
    assert any(
        lim.code == "qa_assessment_unavailable" and "qa_quality" in lim.affected_dimensions
        for lim in result.evaluation.limitations
    )
    assert "- **Assessment Score:** Not Evaluated" in result.report_markdown
    assert "0/100" not in result.report_markdown


@pytest.mark.asyncio
async def test_run_report_stage_markdown_and_evaluation_labels_match() -> None:
    """Verify evaluation.json and report.md use the same 40/60/80 score_to_label bands."""
    bundle = ReportEvidenceBundle(
        session_id="session_label_consistency",
        transcript_full_text="Pitch transcript.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 30000,
                "intervals": [{"start_ms": 0, "end_ms": 30000, "formatted": "00:00 - 00:30"}],
                "windows": [],
            }
        },
        questions=[
            {"id": "q1", "text": "Q1?", "rubric_dimension": "pitch_content_and_evidence"},
            {"id": "q2", "text": "Q2?", "rubric_dimension": "technical_feasibility"},
        ],
        answers=[
            {"question_id": "q1", "status": "submitted", "transcript": "Answer 1."},
            {"question_id": "q2", "status": "submitted", "transcript": "Answer 2."},
        ],
        assessments=[
            {"question_id": "q1", "score": 0.62, "assessment_text": "Solid answer."},
            {"question_id": "q2", "score": 0.82, "assessment_text": "Thorough answer."},
        ],
    )

    result = await run_report_stage(
        report_id="01JREPORT_LABELS_01",
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
    )

    eval_model = result.evaluation
    md = result.report_markdown

    # FakeJudgeModelProvider returns 0.85, 0.80, 0.82; Q&A = (0.62 + 0.82) / 2 = 0.72
    labels_by_dim = {c.dimension: c.label for c in eval_model.components if c.status == "scored"}
    assert labels_by_dim["pitch_content_and_evidence"] == "strong"
    assert labels_by_dim["business_and_problem_solution_reasoning"] == "strong"
    assert labels_by_dim["technical_feasibility"] == "strong"
    assert labels_by_dim["qa_quality"] == "good"

    for comp in eval_model.components:
        if comp.status == "scored" and comp.normalized_score is not None:
            expected_label = score_to_label(
                comp.normalized_score, rubric_id=eval_model.rubric.rubric_id
            )
            assert comp.label == expected_label
            assert SCORE_LABEL_DISPLAY[expected_label] in md

    assert "- **Assessment Score:** Good (62/100)" in md
    assert "- **Assessment Score:** Strong (82/100)" in md


@pytest.mark.asyncio
@pytest.mark.parametrize("mappings", [None, []])
async def test_run_report_stage_without_speaker_mappings_emits_limitation_and_no_members(
    mappings: list[SpeakerMapping] | None,
) -> None:
    """Missing speaker mappings must not invent user_speaker_XX members; emit a limitation."""
    bundle = ReportEvidenceBundle(
        session_id="session_unmapped_01",
        qa_round_id="01JUNMAPPEDROUND0000000001",
        analysis_attempt=2,
        transcript_full_text="Pitch without confirmed speaker mappings.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 15000,
                "intervals": [{"start_ms": 0, "end_ms": 15000, "formatted": "00:00 - 00:15"}],
                "windows": [],
            },
            "SPEAKER_01": {
                "speaking_time_ms": 10000,
                "intervals": [{"start_ms": 15000, "end_ms": 25000, "formatted": "00:15 - 00:25"}],
                "windows": [],
            },
        },
    )

    result = await run_report_stage(
        report_id="01JREPORT_UNMAPPED_01",
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
        speaker_mappings=mappings,
    )

    assert result.evaluation.member_feedback == []
    assert result.member_feedback_user_ids == []
    assert "user_speaker_" not in result.evaluation.model_dump_json()
    assert any(
        lim.code == "speaker_mappings_not_provided" and lim.scope == "member_feedback"
        for lim in result.limitations
    )
    assert any(lim.code == "speaker_mappings_not_provided" for lim in result.evaluation.limitations)
    assert "*No mapped individual presenter feedback available.*" in result.report_markdown


@pytest.mark.asyncio
async def test_run_report_stage_uses_real_attempt_and_qa_round_ids(tmp_path: Path) -> None:
    """Evaluation uses real qa_round_id from qa.json and attempt identity, never session ID."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    session_id = "109156be-c4fb-41ea-b1b4-efe1671c5836"
    qa_round_id = "22222222-3333-4444-5555-666666666666"
    attempt_id = "77777777-8888-9999-aaaa-bbbbbbbbbbbb"

    analysis_ref = await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        build_synthetic_analysis_data(session_id=session_id),
        artifact_id="01JANALYSIS_REAL_IDS_001",
    )
    qa_payload = build_synthetic_qa_data(session_id=session_id)
    qa_payload["qa_round_id"] = qa_round_id
    qa_payload["analysis_attempt"] = 2
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/qa.json",
        qa_payload,
        artifact_id="01JQA_REAL_IDS_000000001",
    )

    result_with_attempt_id = await run_report_stage(
        report_id="01JREPORT_REAL_IDS_00001",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=storage,
        judge_provider=FakeJudgeModelProvider(),
        analysis_attempt=2,
        analysis_attempt_id=attempt_id,
    )
    assert result_with_attempt_id.evaluation.analysis_attempt_id == attempt_id
    assert result_with_attempt_id.evaluation.qa_round_id == qa_round_id
    assert f"- **Analysis Attempt ID:** `{attempt_id}`" in result_with_attempt_id.report_markdown
    assert f"- **Q&A Round ID:** `{qa_round_id}`" in result_with_attempt_id.report_markdown

    result_without_attempt_uuid = await run_report_stage(
        report_id="01JREPORT_REAL_IDS_00002",
        analysis_ref=analysis_ref,
        qa_ref=qa_ref,
        storage=storage,
        judge_provider=FakeJudgeModelProvider(),
    )
    assert result_without_attempt_uuid.evaluation.analysis_attempt_id == f"{session_id}:attempt_2"
    assert result_without_attempt_uuid.evaluation.analysis_attempt_id != session_id
    assert result_without_attempt_uuid.evaluation.qa_round_id == qa_round_id


@pytest.mark.asyncio
async def test_run_report_stage_produces_qa_feedback_only_for_members_who_answered() -> None:
    """MemberFeedback.qa_feedback is set for members who submitted answers and None otherwise."""
    bundle = ReportEvidenceBundle(
        session_id="session_member_qa_01",
        qa_round_id="round_member_qa_01",
        transcript_full_text="Team pitch.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 20000,
                "intervals": [{"start_ms": 0, "end_ms": 20000, "formatted": "00:00 - 00:20"}],
                "windows": [],
            },
            "SPEAKER_01": {
                "speaking_time_ms": 15000,
                "intervals": [{"start_ms": 20000, "end_ms": 35000, "formatted": "00:20 - 00:35"}],
                "windows": [],
            },
        },
        questions=[
            {
                "id": "q1",
                "text": "What is your gross margin?",
                "rubric_dimension": "business_and_problem_solution_reasoning",
                "evidence_ids": ["ev_speech_001"],
            },
            {
                "id": "q2",
                "text": "How does failover work?",
                "rubric_dimension": "technical_feasibility",
                "evidence_ids": ["ev_doc_001"],
            },
            {
                "id": "q3",
                "text": "What is your retention rate?",
                "rubric_dimension": "pitch_content_and_evidence",
                "evidence_ids": ["ev_speech_002"],
            },
        ],
        answers=[
            {
                "id": "a1",
                "question_id": "q1",
                "answered_by": "user_alice",
                "status": "submitted",
                "transcript": "78% gross margin.",
            },
            {
                "id": "a2",
                "question_id": "q2",
                "answered_by": "user_alice",
                "status": "submitted",
                "transcript": "We have one node.",
            },
            {
                "id": "a3",
                "question_id": "q3",
                "answered_by": "user_bob",
                "status": "skipped",
            },
        ],
        assessments=[
            {
                "question_id": "q1",
                "answer_id": "a1",
                "score": 0.85,
                "assessment_text": "Concrete gross margin figure provided.",
                "evidence_ids": ["ev_ans_01"],
            },
            {
                "question_id": "q2",
                "answer_id": "a2",
                "score": 0.45,
                "assessment_text": "Single-node setup lacks redundancy details.",
                "evidence_ids": ["ev_ans_02"],
            },
        ],
    )

    result = await run_report_stage(
        report_id="01JREPORT_MEMBER_QA_01",
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="user_alice",
                display_name="Alice Founder",
            ),
            SpeakerMapping(
                speaker_label="SPEAKER_01",
                user_id="user_bob",
                display_name="Bob CTO",
            ),
        ],
    )

    by_user = {mf.user_id: mf for mf in result.evaluation.member_feedback}
    alice = by_user["user_alice"]
    bob = by_user["user_bob"]

    assert alice.qa_feedback is not None
    assert "65/100" in alice.qa_feedback.summary
    assert len(alice.qa_feedback.strengths) == 1
    assert alice.qa_feedback.strengths[0].evidence_ids == ["ev_ans_01", "ev_speech_001"]
    assert len(alice.qa_feedback.improvements) == 1
    assert alice.qa_feedback.improvements[0].evidence_ids == ["ev_ans_02", "ev_doc_001"]
    assert len(alice.qa_feedback.score_components) == 1
    assert alice.qa_feedback.score_components[0].dimension == "qa_quality"
    assert alice.qa_feedback.score_components[0].status == "scored"
    assert alice.qa_feedback.score_components[0].normalized_score == 0.65
    assert alice.qa_feedback.score_components[0].label == "good"
    assert "#### Presenter Q&A Feedback" in result.report_markdown

    # Bob only skipped a question and submitted no answers -> qa_feedback must be None
    assert bob.qa_feedback is None


@pytest.mark.asyncio
async def test_run_report_stage_unscored_member_qa_emits_limitation() -> None:
    """Member who answered without recoverable assessment scores gets not_evaluated qa_feedback."""
    bundle = ReportEvidenceBundle(
        session_id="session_member_qa_unscored",
        qa_round_id="round_member_qa_unscored",
        transcript_full_text="Team pitch.",
        by_speaker={
            "SPEAKER_00": {
                "speaking_time_ms": 0,
                "intervals": [],
                "windows": [],
            }
        },
        questions=[
            {
                "id": "q1",
                "text": "What is your burn rate?",
                "rubric_dimension": "business_and_problem_solution_reasoning",
            }
        ],
        answers=[
            {
                "id": "a1",
                "question_id": "q1",
                "answered_by": "user_alice",
                "status": "submitted",
                "transcript": "Our burn rate is $12k/month.",
            }
        ],
        assessments=[],
    )

    result = await run_report_stage(
        report_id="01JREPORT_MEMBER_QA_UNSCORED",
        evidence_bundle_override=bundle,
        judge_provider=FakeJudgeModelProvider(),
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="user_alice",
                display_name="Alice Founder",
            )
        ],
    )

    alice = result.evaluation.member_feedback[0]
    assert alice.qa_feedback is not None
    assert alice.qa_feedback.strengths == []
    assert alice.qa_feedback.improvements == []
    assert alice.qa_feedback.score_components[0].status == "not_evaluated"
    assert alice.qa_feedback.score_components[0].limitation_code == "qa_assessment_unavailable"
    assert any(lim.code == "qa_assessment_unavailable" for lim in alice.qa_feedback.limitations)
    assert "#### Presenter Q&A Feedback" in result.report_markdown


@pytest.mark.asyncio
async def test_run_report_stage_all_unscored_emits_explicit_limitation() -> None:
    """When zero dimensions can be scored, overall_score=None must emit an explicit limitation."""

    class AllUnscoredJudge(FakeJudgeModelProvider):
        async def generate_report_feedback(self, **kwargs):  # type: ignore[override]
            res = await super().generate_report_feedback(**kwargs)
            res.dimension_scores = {}
            return res

    bundle = ReportEvidenceBundle(
        session_id="session_all_unscored",
        qa_round_id="round_all_unscored",
        transcript_full_text="",
        by_speaker={},
        questions=[],
        answers=[],
        assessments=[],
    )

    result = await run_report_stage(
        report_id="01JREPORT_ALL_UNSCORED",
        evidence_bundle_override=bundle,
        judge_provider=AllUnscoredJudge(),
        speaker_mappings=[],
    )

    assert result.evaluation.overall_score is None
    assert all(comp.status == "not_evaluated" for comp in result.evaluation.components)
    assert any(
        lim.code == "all_dimensions_not_evaluated" and lim.scope == "evaluation"
        for lim in result.limitations
    )
    assert any(lim.code == "all_dimensions_not_evaluated" for lim in result.evaluation.limitations)
    assert any(
        lim.code == "all_dimensions_not_evaluated"
        for lim in result.evaluation.team_feedback.limitations
    )
