"""Unit tests for report evidence and Q&A loader (app.stages.report_loader)."""

import logging
from pathlib import Path

import pytest

from app.contracts import ArtifactRef
from app.stages.report_loader import (
    ReportEvidenceBundle,
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
    load_report_evidence,
)
from app.storage.local import LocalDiskObjectStorage
from tests.conftest import load_json_fixture

FAKE_SHA256 = "sha256:" + "0" * 64


def test_build_synthetic_analysis_data() -> None:
    """Verify synthetic analysis payload contains required schema keys."""
    data = build_synthetic_analysis_data("01JTEST0001")
    assert data["session_id"] == "01JTEST0001"
    assert "transcript" in data
    assert "by_speaker" in data
    assert "SPEAKER_00" in data["by_speaker"]
    assert "SPEAKER_01" in data["by_speaker"]
    assert len(data["by_speaker"]["SPEAKER_00"]["intervals"]) >= 1


def test_build_synthetic_qa_data() -> None:
    """Verify synthetic Q&A payload contains questions, answers, and assessments."""
    data = build_synthetic_qa_data("01JTEST0001")
    assert "questions" in data
    assert len(data["questions"]) == 3
    assert "answers" in data
    assert len(data["answers"]) == 3
    assert "assessments" in data
    assert len(data["assessments"]) == 3


@pytest.mark.asyncio
async def test_load_report_evidence_from_storage(tmp_path: Path) -> None:
    """Verify load_report_evidence correctly fetches and unpacks artifacts from storage."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)

    # 1. Upload sample analysis.json
    analysis_key = "ai/session/session_100/analysis.json"
    analysis_payload = {
        "session_id": "session_100",
        "practice_session_id": "session_100",
        "transcript": {
            "text": "Hello world from founders.",
            "segments": [{"start_ms": 0, "end_ms": 5000, "text": "Hello world from founders."}],
        },
        "by_speaker": {
            "SPEAKER_00": {
                "speaking_time_ms": 5000,
                "intervals": [{"start_ms": 0, "end_ms": 5000, "formatted": "00:00 - 00:05"}],
                "windows": [],
            }
        },
        "limitations": [
            {
                "code": "audio_warning",
                "scope": "audio",
                "message": "Low SNR",
                "affected_dimensions": [],
            }
        ],
    }
    analysis_ref = await storage.upload_json(
        analysis_key, analysis_payload, artifact_id="01JANALYSIS00000000000001"
    )

    # 2. Upload sample qa.json
    qa_key = "ai/session/session_100/qa.json"
    qa_payload = {
        "questions": [
            {"id": "q1", "text": "What is the moat?", "rubric_dimension": "technical_feasibility"}
        ],
        "answers": [
            {"question_id": "q1", "status": "submitted", "transcript": "Patented algorithms"}
        ],
        "assessments": [{"question_id": "q1", "score": 0.88, "evidence_ids": ["ev_1"]}],
    }
    qa_ref = await storage.upload_json(
        qa_key, qa_payload, artifact_id="01JQA0000000000000000000001"
    )

    # 3. Load evidence bundle
    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=storage)

    assert isinstance(bundle, ReportEvidenceBundle)
    assert bundle.session_id == "session_100"
    assert bundle.transcript_full_text == "Hello world from founders."
    assert "SPEAKER_00" in bundle.by_speaker
    assert len(bundle.limitations) == 1
    assert bundle.limitations[0].code == "audio_warning"
    assert len(bundle.questions) == 1
    assert bundle.questions[0]["id"] == "q1"
    assert len(bundle.answers) == 1
    assert len(bundle.assessments) == 1


@pytest.mark.asyncio
async def test_load_report_evidence_flattens_worker_assessment_score(tmp_path: Path) -> None:
    """The report scorer consumes the nested answer-assessment artifact format."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    analysis_ref = await storage.upload_json(
        "analysis.json",
        {"session_id": "session_101", "transcript": {"text": "Pitch", "segments": []}},
        artifact_id="01JANALYSIS00000000000002",
    )
    qa_ref = await storage.upload_json(
        "qa.json",
        {
            "questions": [{"id": "q1", "text": "What is the CAC?"}],
            "answers": [{"question_id": "q1", "status": "submitted", "transcript": "120 USD"}],
            "assessments": [
                {
                    "question_id": "q1",
                    "assessment": {
                        "text": "The response lacks a payback explanation.",
                        "score": 0.45,
                        "evidence_ids": ["ev_ans_01"],
                    },
                }
            ],
        },
        artifact_id="01JQA0000000000000000000002",
    )

    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=storage)

    assert bundle.assessments == [
        {
            "question_id": "q1",
            "assessment": {
                "text": "The response lacks a payback explanation.",
                "score": 0.45,
                "evidence_ids": ["ev_ans_01"],
            },
            "assessment_text": "The response lacks a payback explanation.",
            "score": 0.45,
            "evidence_ids": ["ev_ans_01"],
        }
    ]


@pytest.mark.asyncio
async def test_load_report_evidence_synthetic_fallback_on_missing_storage(tmp_path: Path) -> None:
    """Verify load_report_evidence falls back to synthetic data when objects are missing."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)

    missing_analysis_ref = ArtifactRef(
        artifact_id="01JMISSING000000000000000001",
        object_key="artifacts/non_existent_analysis.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )
    missing_qa_ref = ArtifactRef(
        artifact_id="01JMISSING000000000000000002",
        object_key="artifacts/non_existent_qa.json",
        checksum=FAKE_SHA256,
        schema_version=1,
    )

    bundle = await load_report_evidence(missing_analysis_ref, missing_qa_ref, storage=storage)

    assert isinstance(bundle, ReportEvidenceBundle)
    assert bundle.session_id == "01JMISSING000000000000000001"
    assert "SPEAKER_00" in bundle.by_speaker
    assert len(bundle.questions) == 3
    assert len(bundle.answers) == 3
    assert len(bundle.assessments) == 3


@pytest.mark.asyncio
async def test_load_report_evidence_none_storage() -> None:
    """Verify load_report_evidence works when storage is None."""
    analysis_ref = ArtifactRef(
        artifact_id="01JTEST00000000000000000001",
        object_key="analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JTEST00000000000000000002",
        object_key="qa.json",
        checksum=FAKE_SHA256,
    )

    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=None)

    assert isinstance(bundle, ReportEvidenceBundle)
    assert bundle.session_id == "01JTEST00000000000000000001"
    assert len(bundle.questions) == 3


@pytest.mark.asyncio
async def test_load_report_evidence_recovers_scores_from_assessment_artifacts(
    tmp_path: Path,
) -> None:
    """Backend-shaped qa.json without scores recovers verified scores from assessment.json."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    qa_payload = load_json_fixture("ai/qa_artifact_valid.json")
    session_id = qa_payload["practice_session_id"]

    analysis_ref = await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        build_synthetic_analysis_data(session_id=session_id),
        artifact_id="01JANALYSIS00000000000010",
    )
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/qa.json",
        qa_payload,
        artifact_id=qa_payload["artifact_id"],
    )

    expected_scores = [0.85, 0.72, 0.68]
    for idx, (ans, score) in enumerate(zip(qa_payload["answers"], expected_scores, strict=True)):
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

    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=storage)

    assert [a["score"] for a in bundle.assessments] == expected_scores
    assert all(a["status"] == "scored" for a in bundle.assessments)
    assert bundle.limitations == []


@pytest.mark.asyncio
async def test_load_report_evidence_rejects_mismatched_assessment_artifact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verification failure on assessment.json marks answer not_evaluated with safe limitation."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    qa_payload = load_json_fixture("ai/qa_artifact_valid.json")
    session_id = qa_payload["practice_session_id"]

    analysis_ref = await storage.upload_json(
        f"ai/session/{session_id}/analysis.json",
        build_synthetic_analysis_data(session_id=session_id),
        artifact_id="01JANALYSIS00000000000011",
    )
    qa_ref = await storage.upload_json(
        f"ai/session/{session_id}/qa.json",
        qa_payload,
        artifact_id=qa_payload["artifact_id"],
    )

    first_ans = qa_payload["answers"][0]
    secret_key = f"ai/session/{session_id}/answers/{first_ans['id']}/assessment.json"
    await storage.upload_json(
        secret_key,
        {
            "artifact_id": "01JWRONGARTIFACT000000000",
            "answer_id": first_ans["id"],
            "practice_session_id": session_id,
            "question_id": first_ans["question_id"],
            "kind": "answer_assessment",
            "schema_version": 1,
            "assessment": {"text": "TAMPERED_SECRET_TEXT", "score": 0.99, "evidence_ids": []},
        },
    )

    caplog.set_level(logging.WARNING)
    bundle = await load_report_evidence(analysis_ref, qa_ref, storage=storage)

    assert all(a["status"] == "not_evaluated" for a in bundle.assessments)
    assert all(a["score"] is None for a in bundle.assessments)
    codes = {lim.code for lim in bundle.limitations}
    assert "qa_assessment_verification_failed" in codes
    assert "qa_assessment_unavailable" in codes
    assert secret_key not in caplog.text
    assert "TAMPERED_SECRET_TEXT" not in caplog.text
