"""Unit tests for report evidence and Q&A loader (app.stages.report_loader)."""

from pathlib import Path

import pytest

from app.contracts import ArtifactRef, Limitation
from app.stages.report_loader import (
    ReportEvidenceBundle,
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
    load_report_evidence,
)
from app.storage.local import LocalDiskObjectStorage

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
            {"code": "audio_warning", "scope": "audio", "message": "Low SNR", "affected_dimensions": []}
        ],
    }
    analysis_ref = await storage.upload_json(analysis_key, analysis_payload, artifact_id="01JANALYSIS00000000000001")

    # 2. Upload sample qa.json
    qa_key = "ai/session/session_100/qa.json"
    qa_payload = {
        "questions": [
            {"id": "q1", "text": "What is the moat?", "rubric_dimension": "technical_feasibility"}
        ],
        "answers": [
            {"question_id": "q1", "status": "submitted", "transcript": "Patented algorithms"}
        ],
        "assessments": [
            {"question_id": "q1", "score": 0.88, "evidence_ids": ["ev_1"]}
        ],
    }
    qa_ref = await storage.upload_json(qa_key, qa_payload, artifact_id="01JQA0000000000000000000001")

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
