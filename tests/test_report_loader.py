"""Unit tests for report evidence and Q&A loader (app.stages.report_loader)."""

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import app.__main__ as app_main
from app.contracts import ArtifactRef, ChecksumMismatchError, GenerateReportPayload, SpeakerMapping
from app.document_store import FakeDocumentStore
from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.fake_documents import FakeDocumentProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.fake_vision import FakeVisionProvider
from app.runtime_mode import resolve_runtime_mode
from app.stages.report_loader import (
    ReportEvidenceBundle,
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
    load_report_evidence,
)
from app.storage.base import ObjectNotFoundError, ObjectStorageError, ObjectStorageProtocol
from app.storage.local import LocalDiskObjectStorage
from tests.conftest import load_json_fixture

LIVE_MODE_VARIANTS = ["live", "LIVE", " live ", "Live", "groq", "GROQ", " groq ", "Groq"]

FAKE_SHA256 = "sha256:" + "0" * 64


def test_build_synthetic_analysis_data() -> None:
    """Verify synthetic analysis payload contains required schema keys and synthetic markers."""
    data = build_synthetic_analysis_data("01JTEST0001")
    assert data["session_id"] == "01JTEST0001"
    assert "transcript" in data
    assert "by_speaker" in data
    assert "SPEAKER_00" in data["by_speaker"]
    assert "SPEAKER_01" in data["by_speaker"]
    assert len(data["by_speaker"]["SPEAKER_00"]["intervals"]) >= 1
    assert data["metadata"]["is_synthetic_fixture"] is True
    assert any(lim["code"] == "synthetic_analysis_fixture" for lim in data["limitations"])


def test_build_synthetic_qa_data() -> None:
    """Verify synthetic Q&A payload contains questions, answers, assessments, and markers."""
    data = build_synthetic_qa_data("01JTEST0001")
    assert "questions" in data
    assert len(data["questions"]) == 3
    assert "answers" in data
    assert len(data["answers"]) == 3
    assert "assessments" in data
    assert len(data["assessments"]) == 3
    assert data["metadata"]["is_synthetic_fixture"] is True
    assert any(lim["code"] == "synthetic_qa_fixture" for lim in data["limitations"])


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
    assert bundle.metadata["is_synthetic_fixture"] is True
    assert bundle.metadata["synthetic_sources"] == ["analysis", "qa"]
    lim_codes = {lim.code for lim in bundle.limitations}
    assert {"synthetic_analysis_fixture", "synthetic_qa_fixture"}.issubset(lim_codes)


@pytest.mark.asyncio
async def test_load_report_evidence_none_storage() -> None:
    """Verify load_report_evidence works when storage is None in test/fake mode."""
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
    assert bundle.metadata["is_synthetic_fixture"] is True
    lim_codes = {lim.code for lim in bundle.limitations}
    assert {"synthetic_analysis_fixture", "synthetic_qa_fixture"}.issubset(lim_codes)


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
    # analysis.json is not stored in this test, so only the marked synthetic-analysis
    # limitation may appear; every Q&A score must be recovered without a Q&A limitation.
    assert [lim.code for lim in bundle.limitations] == ["synthetic_analysis_fixture"]


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


@pytest.mark.parametrize("raw_mode", LIVE_MODE_VARIANTS)
def test_resolve_runtime_mode_live_aliases(monkeypatch: pytest.MonkeyPatch, raw_mode: str) -> None:
    """Every accepted live mode spelling/alias normalizes to live and forbids synthetic fixtures."""
    monkeypatch.setenv("AI_PROVIDER_MODE", raw_mode)
    policy = resolve_runtime_mode()
    assert policy.is_live is True
    assert policy.canonical_mode == "live"
    assert policy.allow_synthetic_report_fixtures is False
    with pytest.raises(ValueError, match="Synthetic fixtures cannot be enabled"):
        policy.with_synthetic_fixtures(True)


def test_resolve_runtime_mode_mixed_prod_and_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic fixtures are disabled in mixed, production, and unconfigured non-test modes."""
    monkeypatch.setenv("AI_PROVIDER_MODE", " MIXED ")
    mixed_policy = resolve_runtime_mode()
    assert mixed_policy.is_mixed is True
    assert mixed_policy.allow_synthetic_report_fixtures is False

    monkeypatch.setenv("AI_PROVIDER_MODE", "fake")
    monkeypatch.setenv("APP_ENV", "production")
    prod_policy = resolve_runtime_mode()
    assert prod_policy.is_production is True
    assert prod_policy.allow_synthetic_report_fixtures is False

    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("SPACE_ID", "virtujudge/space")
    space_policy = resolve_runtime_mode()
    assert space_policy.is_production is True
    assert space_policy.allow_synthetic_report_fixtures is False

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("AI_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    unconfigured = resolve_runtime_mode()
    assert unconfigured.is_fake is True
    assert unconfigured.provider_mode_explicit is False
    assert unconfigured.allow_synthetic_report_fixtures is False

    monkeypatch.setenv("AI_PROVIDER_MODE", "FAKE")
    explicit_fake = resolve_runtime_mode()
    assert explicit_fake.provider_mode == "fake"
    assert explicit_fake.allow_synthetic_report_fixtures is True

    monkeypatch.setenv("AI_PROVIDER_MODE", "unknown_mode")
    with pytest.raises(ValueError, match="AI_PROVIDER_MODE"):
        resolve_runtime_mode()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_mode", LIVE_MODE_VARIANTS)
async def test_load_report_evidence_fails_closed_on_missing_objects_in_live_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_mode: str
) -> None:
    """Missing analysis/Q&A objects fail closed in every supported live mode spelling/alias."""
    monkeypatch.setenv("AI_PROVIDER_MODE", raw_mode)
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    analysis_ref = ArtifactRef(
        artifact_id="01JLIVE0000000000000000001",
        object_key="ai/session/missing/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JLIVE0000000000000000002",
        object_key="ai/session/missing/qa.json",
        checksum=FAKE_SHA256,
    )

    with pytest.raises(ObjectNotFoundError):
        await load_report_evidence(analysis_ref, qa_ref, storage=storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_mode", LIVE_MODE_VARIANTS)
async def test_load_report_evidence_fails_closed_on_missing_qa_with_valid_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_mode: str
) -> None:
    """Missing qa.json must fail closed even when analysis.json exists in live modes."""
    monkeypatch.setenv("AI_PROVIDER_MODE", raw_mode)
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    analysis_ref = await storage.upload_json(
        "ai/session/live_01/analysis.json",
        {"session_id": "live_01", "transcript": {"text": "Real pitch", "segments": []}},
        artifact_id="01JLIVE0000000000000000003",
    )
    missing_qa_ref = ArtifactRef(
        artifact_id="01JLIVE0000000000000000004",
        object_key="ai/session/live_01/qa.json",
        checksum=FAKE_SHA256,
    )

    with pytest.raises(ObjectNotFoundError):
        await load_report_evidence(analysis_ref, missing_qa_ref, storage=storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_mode", LIVE_MODE_VARIANTS)
async def test_load_report_evidence_fails_closed_on_storage_outage_corrupt_and_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, raw_mode: str
) -> None:
    """Storage errors, corrupt JSON, invalid payloads, and None storage fail closed in live mode."""
    monkeypatch.setenv("AI_PROVIDER_MODE", raw_mode)
    analysis_ref = ArtifactRef(
        artifact_id="01JLIVE0000000000000000005",
        object_key="ai/session/live_02/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JLIVE0000000000000000006",
        object_key="ai/session/live_02/qa.json",
        checksum=FAKE_SHA256,
    )

    # 1. storage=None fails closed
    with pytest.raises(ObjectStorageError):
        await load_report_evidence(analysis_ref, qa_ref, storage=None)

    # 2. Storage outage (ObjectStorageError) fails closed without leaking signed URLs/secrets
    outage_storage = AsyncMock(spec=ObjectStorageProtocol)
    outage_storage.download_file.side_effect = ObjectStorageError(
        "https://bucket.example/analysis.json?X-Amz-Signature=SECRET123 timeout"
    )
    caplog.set_level(logging.ERROR)
    with pytest.raises(ObjectStorageError):
        await load_report_evidence(analysis_ref, qa_ref, storage=outage_storage)
    assert "SECRET123" not in caplog.text
    assert "X-Amz-Signature" not in caplog.text

    disk_storage = LocalDiskObjectStorage(base_dir=tmp_path)
    analysis_path = tmp_path / "ai" / "session" / "live_02" / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    def _ref_for(text: str) -> ArtifactRef:
        # Matching checksum so the payload, not the integrity check, is what fails.
        analysis_path.write_text(text, encoding="utf-8")
        return analysis_ref.model_copy(
            update={"checksum": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()}
        )

    # 3. Empty payload with a valid checksum fails closed
    with pytest.raises(ObjectStorageError):
        await load_report_evidence(_ref_for("{}"), qa_ref, storage=disk_storage)

    # 4. Corrupt JSON with a valid checksum fails closed
    with pytest.raises(json.JSONDecodeError):
        await load_report_evidence(_ref_for("{not valid json"), qa_ref, storage=disk_storage)

    # 5. A checksum mismatch always fails closed and never falls back to a fixture
    analysis_path.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ChecksumMismatchError):
        await load_report_evidence(analysis_ref, qa_ref, storage=disk_storage)


@pytest.mark.asyncio
async def test_load_report_evidence_rejects_synthetic_fallback_without_test_or_fake_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing objects fail closed when neither explicit fake mode nor test env is configured."""
    monkeypatch.delenv("AI_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("APP_ENV", "development")

    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    analysis_ref = ArtifactRef(
        artifact_id="01JDEV00000000000000000001",
        object_key="missing/analysis.json",
        checksum=FAKE_SHA256,
    )
    qa_ref = ArtifactRef(
        artifact_id="01JDEV00000000000000000002",
        object_key="missing/qa.json",
        checksum=FAKE_SHA256,
    )

    with pytest.raises(ObjectNotFoundError):
        await load_report_evidence(analysis_ref, qa_ref, storage=storage)
    with pytest.raises(ObjectStorageError):
        await load_report_evidence(analysis_ref, qa_ref, storage=None)

    # Explicit production env also overrides AI_PROVIDER_MODE=fake
    monkeypatch.setenv("AI_PROVIDER_MODE", "fake")
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ObjectNotFoundError):
        await load_report_evidence(analysis_ref, qa_ref, storage=storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_mode", LIVE_MODE_VARIANTS)
async def test_build_pipeline_shares_normalized_live_policy_with_generate_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_mode: str
) -> None:
    """build_pipeline resolves mode once and shares that live fail-closed policy with reporting."""
    import app.providers.librosa_audio as librosa_mod
    import app.providers.mediapipe_vision as mediapipe_mod
    import app.providers.pyannote_diarization as pyannote_mod
    import app.providers.pymupdf_documents as pymupdf_mod

    monkeypatch.setenv("AI_PROVIDER_MODE", raw_mode)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")

    monkeypatch.setattr(app_main, "GroqSpeechProvider", FakeSpeechProvider)
    monkeypatch.setattr(app_main, "GroqJudgeModelProvider", FakeJudgeModelProvider)
    monkeypatch.setattr(pyannote_mod, "PyannoteDiarizationProvider", FakeDiarizationProvider)
    monkeypatch.setattr(mediapipe_mod, "MediaPipeVisionProvider", FakeVisionProvider)
    monkeypatch.setattr(librosa_mod, "LibrosaAudioProvider", FakeAudioMetricsProvider)
    monkeypatch.setattr(pymupdf_mod, "PyMuPDFDocumentProvider", FakeDocumentProvider)
    monkeypatch.setattr(
        app_main, "create_document_store", AsyncMock(return_value=FakeDocumentStore())
    )
    empty_storage = LocalDiskObjectStorage(base_dir=tmp_path)
    monkeypatch.setattr(app_main, "create_object_storage", lambda: empty_storage)

    pipeline = await app_main.build_pipeline()
    assert pipeline.mode_policy is not None
    assert pipeline.mode_policy.is_live is True
    assert pipeline.mode_policy.allow_synthetic_report_fixtures is False

    # Mutating env later must not downgrade the already-resolved pipeline policy.
    monkeypatch.setenv("AI_PROVIDER_MODE", "fake")

    report_payload = GenerateReportPayload(
        report_id="01JREPORT_LIVE_FAIL_000001",
        analysis_artifact=ArtifactRef(
            artifact_id="01JANALYSIS_MISSING_000001",
            object_key="ai/session/missing/analysis.json",
            checksum=FAKE_SHA256,
        ),
        qa_artifact=ArtifactRef(
            artifact_id="01JQA_MISSING_000000000001",
            object_key="ai/session/missing/qa.json",
            checksum=FAKE_SHA256,
        ),
        speaker_mappings=[
            SpeakerMapping(
                speaker_label="SPEAKER_00",
                user_id="user_01",
                display_name="Founder",
            )
        ],
    )

    with pytest.raises(ObjectNotFoundError):
        await pipeline.generate_report(report_payload)
