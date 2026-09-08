"""Unit tests for app.providers.pyannote_diarization (pyannote Community-1 adapter)."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.contracts import Limitation
from app.providers.base import DiarizationProvider
from app.providers.pyannote_diarization import (
    PyannoteDiarizationProvider,
    create_diarization_uncertainty_limitation,
)
from app.providers.types import DiarizationResult, SpeakerSegment


@dataclass
class FakeTurn:
    start: float
    end: float


class FakeAnnotation:
    """Mock pyannote.core.Annotation object."""

    def __init__(self, tracks: list[tuple[FakeTurn, str, str | None]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = True):
        yield from self.tracks


def _create_mock_pyannote_pipeline(tracks: list[tuple[FakeTurn, str, str | None]]):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = FakeAnnotation(tracks)
    return mock_pipeline


def test_pyannote_provider_satisfies_protocol() -> None:
    """Verify PyannoteDiarizationProvider conforms to DiarizationProvider Protocol."""
    mock_pipeline = _create_mock_pyannote_pipeline([])
    provider: DiarizationProvider = PyannoteDiarizationProvider(pipeline_instance=mock_pipeline)
    assert hasattr(provider, "diarize")
    assert callable(provider.diarize)


def test_pyannote_provider_missing_dependency() -> None:
    """Verify ImportError is raised at instantiation time when pyannote.audio is not installed."""
    with (
        patch.dict("sys.modules", {"pyannote.audio": None}),
        pytest.raises(
            ImportError,
            match=r"pyannote\.audio is required for PyannoteDiarizationProvider",
        ),
    ):
        PyannoteDiarizationProvider(pipeline_instance=None)


@pytest.mark.asyncio
async def test_pyannote_multi_speaker_fixture(tmp_path: Path) -> None:
    """Verify multi-speaker fixtures produce expected speaker count and anonymous labels."""
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")

    tracks = [
        (FakeTurn(0.0, 3.5), "track_0", "Speaker_A"),
        (FakeTurn(3.8, 7.2), "track_1", "Speaker_B"),
        (FakeTurn(7.5, 10.0), "track_2", "Speaker_A"),
        (FakeTurn(10.2, 13.0), "track_3", "Speaker_C"),
    ]
    mock_pipeline = _create_mock_pyannote_pipeline(tracks)
    provider = PyannoteDiarizationProvider(
        model_name="pyannote/speaker-diarization-community-1",
        min_speakers=2,
        max_speakers=4,
        pipeline_instance=mock_pipeline,
    )

    result = await provider.diarize(audio_path)

    assert isinstance(result, DiarizationResult)
    # Expected 3 distinct speakers mapped anonymously in order of first appearance
    assert len(result.speaker_labels) == 3
    assert result.speaker_labels == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    assert len(result.segments) == 4
    s0 = result.segments[0]
    assert isinstance(s0, SpeakerSegment)
    assert s0.speaker_label == "SPEAKER_00"
    assert s0.start_ms == 0
    assert s0.end_ms == 3500

    s1 = result.segments[1]
    assert s1.speaker_label == "SPEAKER_01"
    assert s1.start_ms == 3800
    assert s1.end_ms == 7200

    s2 = result.segments[2]
    assert s2.speaker_label == "SPEAKER_00"

    s3 = result.segments[3]
    assert s3.speaker_label == "SPEAKER_02"


@pytest.mark.asyncio
async def test_pyannote_handles_uncertain_missing_speakers(tmp_path: Path) -> None:
    """Verify missing/uncertain speaker assignments are handled and flagged."""
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")

    tracks = [
        (FakeTurn(0.0, 2.0), "track_0", "Speaker_1"),
        (FakeTurn(2.1, 4.0), "track_1", "UNKNOWN"),
        (FakeTurn(4.2, 6.0), "track_2", None),
    ]
    mock_pipeline = _create_mock_pyannote_pipeline(tracks)
    provider = PyannoteDiarizationProvider(pipeline_instance=mock_pipeline)

    result = await provider.diarize(audio_path)

    assert result.metadata["uncertain_turns_count"] == 2
    assert result.segments[1].speaker_label == "SPEAKER_UNKNOWN"
    assert result.segments[2].speaker_label == "SPEAKER_UNKNOWN"

    limitation = create_diarization_uncertainty_limitation(result.metadata["uncertain_turns_count"])
    assert isinstance(limitation, Limitation)
    assert limitation.code == "uncertain_speaker_assignment"
    assert limitation.scope == "diarization"
    assert "2 unassigned" in limitation.message


@pytest.mark.asyncio
async def test_pyannote_records_metadata(tmp_path: Path) -> None:
    """Verify model version, configuration, and detected speaker counts are recorded."""
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")

    tracks = [(FakeTurn(0.0, 1.0), "track_0", "Speaker_1")]
    mock_pipeline = _create_mock_pyannote_pipeline(tracks)
    provider = PyannoteDiarizationProvider(
        model_name="pyannote/speaker-diarization-community-1",
        device="cpu",
        min_speakers=1,
        max_speakers=2,
        pipeline_instance=mock_pipeline,
    )

    result = await provider.diarize(audio_path)

    assert result.metadata["provider"] == "pyannote.audio"
    assert result.metadata["model_name"] == "pyannote/speaker-diarization-community-1"
    assert result.metadata["device"] == "cpu"
    assert result.metadata["min_speakers"] == 1
    assert result.metadata["max_speakers"] == 2
    assert result.metadata["detected_speaker_count"] == 1
    assert result.metadata["uncertain_turns_count"] == 0


@pytest.mark.asyncio
async def test_pyannote_missing_audio(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when audio file is not found."""
    mock_pipeline = _create_mock_pyannote_pipeline([])
    provider = PyannoteDiarizationProvider(pipeline_instance=mock_pipeline)

    missing = tmp_path / "non_existent.wav"
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        await provider.diarize(missing)
