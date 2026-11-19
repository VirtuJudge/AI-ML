"""Unit tests for app.providers.pyannote_diarization (pyannote speaker diarization adapter)."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.contracts import Limitation
from app.providers.base import DiarizationProvider
from app.providers.pyannote_diarization import (
    CANONICAL_DIARIZATION_3_1,
    CANONICAL_DIARIZATION_COMMUNITY_1,
    PyannoteDiarizationProvider,
    create_diarization_uncertainty_limitation,
)
from app.providers.types import DiarizationResult, SpeakerSegment
from app.stages.checkpoint import extract_provider_fingerprint


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


@pytest.mark.asyncio
async def test_pyannote_device_assignment(tmp_path: Path) -> None:
    """Verify provider stores device and handles waveform tensor placement."""
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")

    mock_pipeline = _create_mock_pyannote_pipeline([(FakeTurn(0.0, 1.0), "track_0", "Spk_1")])
    mock_pipeline.device = "cpu"
    provider = PyannoteDiarizationProvider(
        pipeline_instance=mock_pipeline,
        device="cpu",
    )
    assert provider._device == "cpu"
    result = await provider.diarize(audio_path)
    assert len(result.segments) == 1


class _RecordingPipelineStub:
    """Fake loaded pyannote Pipeline returned by a recording from_pretrained loader."""

    def __init__(
        self,
        *,
        tracks: list[tuple[FakeTurn, str, str | None]] | None = None,
        otel_origin: str | None = None,
        revision: str | None = None,
    ) -> None:
        self._annotation = FakeAnnotation(tracks or [])
        self._otel_origin = otel_origin
        if revision is not None:
            self.revision = revision
        self.device: Any = None

    def to(self, dev: Any) -> "_RecordingPipelineStub":
        self.device = dev
        return self

    def __call__(self, *_args: Any, **_kwargs: Any) -> FakeAnnotation:
        return self._annotation


def _install_recording_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "4.0.7",
    resolved_revision: str | None = None,
    override_origin: str | None = None,
    legacy_use_auth_token_only: bool = False,
    accept_revision: bool = True,
) -> list[dict[str, Any]]:
    """Install a fake pyannote.audio module with a recording Pipeline.from_pretrained."""
    calls: list[dict[str, Any]] = []

    class _FakePipeline:
        pass

    if legacy_use_auth_token_only:

        def _from_pretrained(
            model_id: str,
            use_auth_token: str | None = None,
            revision: str | None = None,
        ) -> _RecordingPipelineStub:
            calls.append(
                {
                    "model_id": model_id,
                    "use_auth_token": use_auth_token,
                    "revision": revision,
                }
            )
            return _RecordingPipelineStub(
                tracks=[(FakeTurn(0.0, 1.0), "track_0", "Speaker_1")],
                otel_origin=override_origin or model_id,
                revision=resolved_revision or revision,
            )

    elif not accept_revision:

        def _from_pretrained(
            model_id: str,
            token: str | None = None,
        ) -> _RecordingPipelineStub:
            calls.append({"model_id": model_id, "token": token})
            return _RecordingPipelineStub(
                tracks=[(FakeTurn(0.0, 1.0), "track_0", "Speaker_1")],
                otel_origin=override_origin or model_id,
                revision=resolved_revision,
            )

    else:

        def _from_pretrained(
            model_id: str,
            token: str | None = None,
            revision: str | None = None,
        ) -> _RecordingPipelineStub:
            calls.append({"model_id": model_id, "token": token, "revision": revision})
            return _RecordingPipelineStub(
                tracks=[(FakeTurn(0.0, 1.0), "track_0", "Speaker_1")],
                otel_origin=override_origin or model_id,
                revision=resolved_revision or revision,
            )

    _FakePipeline.from_pretrained = staticmethod(_from_pretrained)  # type: ignore[attr-defined]
    fake_mod = SimpleNamespace(Pipeline=_FakePipeline, __version__=version)
    monkeypatch.setitem(__import__("sys").modules, "pyannote.audio", fake_mod)
    return calls


@pytest.mark.parametrize(
    ("requested", "expected_canonical", "expected_alias"),
    [
        (CANONICAL_DIARIZATION_3_1, CANONICAL_DIARIZATION_3_1, None),
        ("pyannote-3.1", CANONICAL_DIARIZATION_3_1, "pyannote-3.1"),
        ("speaker-diarization-3.1", CANONICAL_DIARIZATION_3_1, "speaker-diarization-3.1"),
        ("3.1", CANONICAL_DIARIZATION_3_1, "3.1"),
        (CANONICAL_DIARIZATION_COMMUNITY_1, CANONICAL_DIARIZATION_COMMUNITY_1, None),
        ("pyannote-community-1", CANONICAL_DIARIZATION_COMMUNITY_1, "pyannote-community-1"),
        ("community-1", CANONICAL_DIARIZATION_COMMUNITY_1, "community-1"),
        (
            "speaker-diarization-community-1",
            CANONICAL_DIARIZATION_COMMUNITY_1,
            "speaker-diarization-community-1",
        ),
    ],
)
def test_pyannote_resolves_canonical_and_aliases_without_cross_substitution(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    expected_canonical: str,
    expected_alias: str | None,
) -> None:
    """Canonical IDs and aliases pass the actual canonical model to Pipeline.from_pretrained."""
    monkeypatch.delenv("DIARIZATION_MODEL", raising=False)
    calls = _install_recording_loader(monkeypatch, version="4.0.7")

    provider = PyannoteDiarizationProvider(
        model_name=requested,
        auth_token="hf_test_token",
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["model_id"] == expected_canonical
    assert calls[0]["token"] == "hf_test_token"
    assert calls[0]["revision"] is None
    assert provider.loaded_model == expected_canonical
    assert provider.model_name == expected_canonical
    assert provider.requested_model == requested
    assert provider.model_alias == expected_alias


@pytest.mark.asyncio
async def test_pyannote_records_loaded_model_and_revision_separately_from_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Loaded canonical model and resolved revision are recorded separately from user alias."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"dummy")

    calls = _install_recording_loader(
        monkeypatch,
        version="4.0.7",
        resolved_revision="commit_sha_9f8e7d6c",
    )

    provider = PyannoteDiarizationProvider(
        model_name="community-1",
        revision="main",
        auth_token="hf_test_token",
        device="cpu",
    )

    assert calls == [
        {
            "model_id": CANONICAL_DIARIZATION_COMMUNITY_1,
            "token": "hf_test_token",
            "revision": "main",
        }
    ]
    assert provider.requested_model == "community-1"
    assert provider.model_alias == "community-1"
    assert provider.loaded_model == CANONICAL_DIARIZATION_COMMUNITY_1
    assert provider.model_name == CANONICAL_DIARIZATION_COMMUNITY_1
    assert provider.requested_revision == "main"
    assert provider.loaded_revision == "commit_sha_9f8e7d6c"

    result = await provider.diarize(audio_path)
    assert result.metadata["model_name"] == CANONICAL_DIARIZATION_COMMUNITY_1
    assert result.metadata["loaded_model"] == CANONICAL_DIARIZATION_COMMUNITY_1
    assert result.metadata["loaded_revision"] == "commit_sha_9f8e7d6c"
    assert result.metadata["requested_model"] == "community-1"
    assert result.metadata["model_alias"] == "community-1"
    assert result.metadata["requested_revision"] == "main"


def test_pyannote_env_var_alias_resolves_to_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DIARIZATION_MODEL env var alias resolves to its canonical model without substitution."""
    monkeypatch.setenv("DIARIZATION_MODEL", "pyannote-community-1")
    calls = _install_recording_loader(monkeypatch, version="4.0.7")

    provider = PyannoteDiarizationProvider(auth_token="hf_test_token", device="cpu")

    assert len(calls) == 1
    assert calls[0]["model_id"] == CANONICAL_DIARIZATION_COMMUNITY_1
    assert provider.requested_model == "pyannote-community-1"
    assert provider.model_alias == "pyannote-community-1"
    assert provider.loaded_model == CANONICAL_DIARIZATION_COMMUNITY_1
    assert provider.model_name == CANONICAL_DIARIZATION_COMMUNITY_1


def test_pyannote_default_model_remains_3_1_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default model remains pyannote/speaker-diarization-3.1 when env/arg are unset."""
    monkeypatch.delenv("DIARIZATION_MODEL", raising=False)
    calls = _install_recording_loader(monkeypatch, version="4.0.7")

    provider = PyannoteDiarizationProvider(auth_token="hf_test_token", device="cpu")

    assert calls[0]["model_id"] == CANONICAL_DIARIZATION_3_1
    assert provider.loaded_model == CANONICAL_DIARIZATION_3_1
    assert provider.model_alias is None


@pytest.mark.parametrize(
    "bad_model",
    [
        "",
        "   ",
        "unknown-model",
        "pyannote/speaker-diarization-2.1",
        "pyannote/speaker-diarization-3.0",
        "pyannote/speaker-diarization-community-1@main",
        "community-1@v1",
    ],
)
def test_pyannote_rejects_unsupported_model_names(
    monkeypatch: pytest.MonkeyPatch,
    bad_model: str,
) -> None:
    """Unsupported models and inline @revision specs are rejected before loading."""
    calls = _install_recording_loader(monkeypatch, version="4.0.7")

    with pytest.raises(ValueError, match=r"Unsupported|non-empty"):
        PyannoteDiarizationProvider(model_name=bad_model)

    assert calls == []


def test_pyannote_rejects_community_1_on_pyannote_3x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requesting Community-1 on pyannote.audio 3.x must fail instead of loading 3.1."""
    calls = _install_recording_loader(monkeypatch, version="3.1.1")

    with pytest.raises(ValueError, match=r"requires pyannote\.audio>=4\.0"):
        PyannoteDiarizationProvider(model_name="community-1")

    assert calls == []


def test_pyannote_rejects_3_1_on_pyannote_3_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requesting 3.1 on pyannote.audio < 3.1 must fail instead of loading anyway."""
    calls = _install_recording_loader(monkeypatch, version="3.0.0")

    with pytest.raises(ValueError, match=r"requires pyannote\.audio>=3\.1"):
        PyannoteDiarizationProvider(model_name="3.1")

    assert calls == []


def test_pyannote_rejects_revision_when_loader_lacks_revision_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning revision against a loader without revision support fails rather than dropping it."""
    calls = _install_recording_loader(monkeypatch, version="4.0.7", accept_revision=False)

    with pytest.raises(ValueError, match="revision"):
        PyannoteDiarizationProvider(model_name="3.1", revision="v1.0")

    assert calls == []


def test_pyannote_rejects_empty_revision_and_invalid_speaker_bounds() -> None:
    """Reject empty revision strings and inverted or non-positive speaker bounds."""
    mock_pipeline = _create_mock_pyannote_pipeline([])

    with pytest.raises(ValueError, match="revision"):
        PyannoteDiarizationProvider(revision="  ", pipeline_instance=mock_pipeline)

    with pytest.raises(ValueError, match="min_speakers"):
        PyannoteDiarizationProvider(min_speakers=0, pipeline_instance=mock_pipeline)

    with pytest.raises(ValueError, match="max_speakers"):
        PyannoteDiarizationProvider(max_speakers=0, pipeline_instance=mock_pipeline)

    with pytest.raises(ValueError, match="min_speakers"):
        PyannoteDiarizationProvider(min_speakers=4, max_speakers=2, pipeline_instance=mock_pipeline)


def test_pyannote_rejects_loader_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the underlying loader reports a different model than requested, fail closed."""
    _install_recording_loader(
        monkeypatch,
        version="4.0.7",
        override_origin=CANONICAL_DIARIZATION_3_1,
    )

    with pytest.raises(RuntimeError, match="does not match requested canonical model"):
        PyannoteDiarizationProvider(model_name="community-1")


def test_pyannote_legacy_use_auth_token_fallback_preserves_model_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older Pipeline.from_pretrained(use_auth_token=...) receives canonical model and revision."""
    calls = _install_recording_loader(
        monkeypatch,
        version="3.1.0",
        legacy_use_auth_token_only=True,
    )

    provider = PyannoteDiarizationProvider(
        model_name="3.1",
        revision="rev-31",
        auth_token="hf_legacy",
        device="cpu",
    )

    assert calls == [
        {
            "model_id": CANONICAL_DIARIZATION_3_1,
            "use_auth_token": "hf_legacy",
            "revision": "rev-31",
        }
    ]
    assert provider.loaded_model == CANONICAL_DIARIZATION_3_1
    assert provider.loaded_revision == "rev-31"


def test_pyannote_checkpoint_fingerprint_distinguishes_models_and_revisions() -> None:
    """Stage checkpoint fingerprints distinguish 3.1 vs community-1 and revisions."""
    p_3_1 = PyannoteDiarizationProvider(
        model_name="3.1",
        pipeline_instance=_create_mock_pyannote_pipeline([]),
    )
    p_comm = PyannoteDiarizationProvider(
        model_name="community-1",
        pipeline_instance=_create_mock_pyannote_pipeline([]),
    )
    p_comm_canonical = PyannoteDiarizationProvider(
        model_name=CANONICAL_DIARIZATION_COMMUNITY_1,
        pipeline_instance=_create_mock_pyannote_pipeline([]),
    )
    p_comm_rev = PyannoteDiarizationProvider(
        model_name="community-1",
        revision="rev-2",
        pipeline_instance=_create_mock_pyannote_pipeline([]),
    )

    fp_3_1 = extract_provider_fingerprint(p_3_1)
    fp_comm = extract_provider_fingerprint(p_comm)
    fp_comm_canonical = extract_provider_fingerprint(p_comm_canonical)
    fp_comm_rev = extract_provider_fingerprint(p_comm_rev)

    assert fp_3_1 != fp_comm
    assert fp_comm == fp_comm_canonical
    assert fp_comm != fp_comm_rev
