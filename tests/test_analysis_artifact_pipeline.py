"""Integration test for session analysis artifact persistence in object storage."""

from pathlib import Path

import pytest

from app.contracts import AnalyzeSessionPayload, AssetInput, RubricRef
from app.pipeline import FakePipeline
from app.storage.local import LocalDiskObjectStorage


@pytest.mark.asyncio
async def test_analyze_session_persists_artifact_to_storage(tmp_path: Path) -> None:
    """Verify analyze_session uploads a valid analysis.json artifact and returns
    a real ArtifactRef.
    """
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    pipeline = FakePipeline(object_storage=storage)

    payload = AnalyzeSessionPayload(
        presentation=AssetInput(
            artifact_id="01JTESTSESSION0000000000001",
            object_key="uploads/presentation.mp4",
            checksum="sha256:" + "a" * 64,
            media_type="video/mp4",
        ),
        supporting_documents=[],
        rubric=RubricRef(rubric_id="startup_pitch", version=1),
        requested_capabilities=["speech", "questions"],
    )

    result = await pipeline.analyze_session(payload)

    # 1. Assert ArtifactRef structure
    assert result.analysis_artifact is not None
    assert len(result.analysis_artifact.artifact_id) == 26
    assert (
        result.analysis_artifact.object_key
        == "ai/session/01JTESTSESSION0000000000001/analysis.json"
    )
    assert result.analysis_artifact.checksum.startswith("sha256:")
    assert len(result.analysis_artifact.checksum) == 7 + 64

    # 2. Assert stored JSON payload
    stored_artifact = await storage.read_json(result.analysis_artifact.object_key)
    assert stored_artifact["artifact_id"] == result.analysis_artifact.artifact_id
    assert stored_artifact["session_id"] == "01JTESTSESSION0000000000001"
    assert "transcript" in stored_artifact
    assert "observations" in stored_artifact
    assert "evidence_bundle" in stored_artifact
    assert "by_speaker" in stored_artifact
    by_speaker = stored_artifact["by_speaker"]
    assert "SPEAKER_00" in by_speaker
    assert "SPEAKER_01" in by_speaker
    assert by_speaker["SPEAKER_00"]["speaking_time_ms"] > 0
    assert len(by_speaker["SPEAKER_00"]["intervals"]) >= 1
    assert "formatted" in by_speaker["SPEAKER_00"]["intervals"][0]
    assert len(stored_artifact["primary_questions"]) == 3
