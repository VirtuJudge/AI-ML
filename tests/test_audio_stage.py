"""Unit tests for the audio acoustic measurement stage (app.stages.audio)."""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.types import AudioObservation
from app.stages.audio import (
    AudioStageResult,
    run_audio_stage,
    validate_audio_observations,
)


class FailingAudioProvider:
    """Provider that simulates an unexpected failure/crash."""

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]:
        raise RuntimeError("Audio acoustic feature extraction simulation error")


class EmptyAudioProvider:
    """Provider that returns an empty observation list."""

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]:
        return []


class OutOfBoundsAudioProvider:
    """Provider returning observations with timestamps exceeding duration or inverted."""

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]:
        return [
            AudioObservation(
                start_ms=0,
                end_ms=25000,
                metric="speaking_rate_wpm",
                value=140.0,
                unit="words_per_minute",
                algorithm_version="test/1.0",
            ),
            AudioObservation(
                start_ms=8000,
                end_ms=5000,
                metric="pitch_mean_hz",
                value=150.0,
                unit="hertz",
                algorithm_version="test/1.0",
            ),
        ]


@pytest.fixture
def dummy_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid 16kHz mono WAV file for testing."""
    wav_path = tmp_path / "presentation.wav"
    sample_rate = 16000
    duration_s = 1.0
    n_samples = int(sample_rate * duration_s)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write 1 second of 440 Hz sine wave
        frames = bytearray()
        for i in range(n_samples):
            val = int(10000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)

    return wav_path


@pytest.mark.asyncio
async def test_audio_stage_happy_path(dummy_audio_file: Path) -> None:
    """Verify happy path execution with FakeAudioMetricsProvider."""
    provider = FakeAudioMetricsProvider(duration_ms=15000)
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=15000)

    assert isinstance(result, AudioStageResult)
    assert len(result.observations) == 9
    assert len(result.limitations) == 0
    assert result.metadata["observation_count"] == 9
    assert result.metadata["provider"] == "FakeAudioMetricsProvider"
    assert result.metadata["media_duration_ms"] == 15000


@pytest.mark.asyncio
async def test_audio_stage_timestamps_within_duration(dummy_audio_file: Path) -> None:
    """Verify that all observations respect media duration bounds."""
    duration_ms = 10000
    provider = FakeAudioMetricsProvider(duration_ms=duration_ms)
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=duration_ms)

    assert len(result.observations) > 0
    for obs in result.observations:
        assert obs.start_ms >= 0
        assert obs.end_ms <= duration_ms
        assert obs.start_ms <= obs.end_ms


@pytest.mark.asyncio
async def test_audio_stage_observations_have_required_fields(dummy_audio_file: Path) -> None:
    """Verify all audio observations satisfy contract fields."""
    provider = FakeAudioMetricsProvider()
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=15000)

    for obs in result.observations:
        assert isinstance(obs.start_ms, int)
        assert isinstance(obs.end_ms, int)
        assert isinstance(obs.metric, str) and len(obs.metric) > 0
        assert isinstance(obs.value, float)
        assert isinstance(obs.unit, str) and len(obs.unit) > 0
        assert 0.0 <= obs.confidence <= 1.0
        assert isinstance(obs.algorithm_version, str) and len(obs.algorithm_version) > 0


@pytest.mark.asyncio
async def test_audio_stage_no_audio_stream() -> None:
    """Verify graceful handling when audio_path is None."""
    provider = FakeAudioMetricsProvider()
    result = await run_audio_stage(None, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "no_audio_stream"
    assert lim.scope == "audio_features"
    assert "delivery_pace" in lim.affected_dimensions
    assert "acoustic_features" in lim.affected_dimensions


@pytest.mark.asyncio
async def test_audio_stage_missing_audio_file(tmp_path: Path) -> None:
    """Verify graceful handling when audio file path does not exist on disk."""
    non_existent = tmp_path / "missing_audio.wav"
    provider = FakeAudioMetricsProvider()
    result = await run_audio_stage(non_existent, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "audio_file_missing"
    assert lim.scope == "audio_features"


@pytest.mark.asyncio
async def test_audio_stage_provider_failure(dummy_audio_file: Path) -> None:
    """Verify provider exception is caught and converted to a safe limitation."""
    provider = FailingAudioProvider()
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "audio_extraction_failed"
    assert lim.scope == "audio_features"


@pytest.mark.asyncio
async def test_audio_stage_empty_observations(dummy_audio_file: Path) -> None:
    """Verify empty observations list produces no_audio_observations limitation."""
    provider = EmptyAudioProvider()
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "no_audio_observations"
    assert lim.scope == "audio_features"


@pytest.mark.asyncio
async def test_audio_stage_timestamps_clamped(dummy_audio_file: Path) -> None:
    """Verify observations exceeding media duration are clamped and report limitation."""
    provider = OutOfBoundsAudioProvider()
    media_duration_ms = 10000
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=media_duration_ms)

    lim_codes = [lim.code for lim in result.limitations]
    assert "audio_observation_timestamp_clamped" in lim_codes

    for obs in result.observations:
        assert obs.start_ms >= 0
        assert obs.end_ms <= media_duration_ms
        assert obs.start_ms <= obs.end_ms


@pytest.mark.asyncio
async def test_audio_stage_observable_language(dummy_audio_file: Path) -> None:
    """Verify no emotion/confidence/anxiety/personality claims in observations or limitations."""
    prohibited_terms = {
        "emotion",
        "emotional",
        "happy",
        "sad",
        "angry",
        "nervous",
        "nervousness",
        "anxiety",
        "anxious",
        "honesty",
        "honest",
        "deceptive",
        "personality",
        "mental state",
        "stress",
    }

    provider = FakeAudioMetricsProvider()
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=15000)

    for obs in result.observations:
        metric_lower = obs.metric.lower()
        for term in prohibited_terms:
            assert term not in metric_lower, (
                f"Prohibited term '{term}' found in metric '{obs.metric}'"
            )

    for lim in result.limitations:
        msg_lower = lim.message.lower()
        for term in prohibited_terms:
            assert term not in msg_lower, (
                f"Prohibited term '{term}' found in limitation '{lim.message}'"
            )


@pytest.mark.asyncio
async def test_audio_stage_no_private_data_in_metadata(dummy_audio_file: Path) -> None:
    """Verify metadata does not leak waveform, audio bytes, or private info."""
    provider = FakeAudioMetricsProvider()
    result = await run_audio_stage(dummy_audio_file, provider, media_duration_ms=15000)

    meta = result.metadata
    for sensitive_key in ["waveform", "samples", "audio_data", "raw_audio", "file_bytes"]:
        assert sensitive_key not in meta

    assert meta["stage"] == "audio_features"
    assert meta["observation_count"] == len(result.observations)


def test_validate_audio_observations_function() -> None:
    """Directly test validate_audio_observations clamping helper."""
    raw = [
        AudioObservation(
            start_ms=12000,
            end_ms=15000,
            metric="pitch_mean_hz",
            value=120.0,
            unit="hertz",
            algorithm_version="test",
        ),
        AudioObservation(
            start_ms=5000,
            end_ms=4000,
            metric="pause_duration_ms",
            value=200.0,
            unit="milliseconds",
            algorithm_version="test",
        ),
    ]
    validated, limitations = validate_audio_observations(raw, media_duration_ms=10000)

    assert len(validated) == 2
    assert validated[0].start_ms == 10000
    assert validated[0].end_ms == 10000
    assert validated[1].start_ms == 4000
    assert validated[1].end_ms == 5000
    assert len(limitations) == 1
    assert limitations[0].code == "audio_observation_timestamp_clamped"
