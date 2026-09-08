"""Unit tests for app.providers.librosa_audio."""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.providers.base import AudioMetricsProvider
from app.providers.librosa_audio import (
    ALGORITHM_VERSION,
    LibrosaAudioProvider,
)
from app.providers.types import AudioObservation


@pytest.fixture
def synthetic_sine_wav(tmp_path: Path) -> Path:
    """Create a 5-second 220Hz synthetic WAV audio file."""
    wav_path = tmp_path / "sine_220hz.wav"
    sample_rate = 16000
    duration_s = 5.0
    freq_hz = 220.0
    n_samples = int(sample_rate * duration_s)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            sample = int(16000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(frames)

    return wav_path


def test_librosa_provider_satisfies_protocol() -> None:
    """Verify LibrosaAudioProvider satisfies AudioMetricsProvider Protocol."""
    provider: AudioMetricsProvider = LibrosaAudioProvider()
    assert hasattr(provider, "extract_metrics")
    assert callable(provider.extract_metrics)


@pytest.mark.asyncio
async def test_librosa_provider_invalid_file(tmp_path: Path) -> None:
    """Verify RuntimeError is raised when audio file cannot be loaded."""
    provider = LibrosaAudioProvider()
    non_existent = tmp_path / "does_not_exist.wav"
    with pytest.raises(RuntimeError, match="Failed to load audio file"):
        await provider.extract_metrics(non_existent)


@pytest.mark.asyncio
async def test_librosa_provider_synthetic_sine_wave(synthetic_sine_wav: Path) -> None:
    """Verify pitch, pause, and speaking rate extraction on synthetic audio."""
    provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=5.0)
    observations = await provider.extract_metrics(synthetic_sine_wav)

    assert isinstance(observations, list)
    assert len(observations) > 0
    assert all(isinstance(obs, AudioObservation) for obs in observations)

    metrics = {obs.metric: obs for obs in observations}
    assert "speaking_rate_wpm" in metrics
    assert "pitch_mean_hz" in metrics
    assert "pitch_std_hz" in metrics
    assert "pause_duration_ms" in metrics
    assert "pause_count" in metrics

    # A pure 220Hz sine wave should yield mean pitch near 220Hz (+/- 10Hz)
    pitch_obs = metrics["pitch_mean_hz"]
    assert 210.0 <= pitch_obs.value <= 230.0

    for obs in observations:
        assert obs.algorithm_version == ALGORITHM_VERSION
        assert obs.start_ms <= obs.end_ms
        assert 0.0 <= obs.confidence <= 1.0


@pytest.mark.asyncio
async def test_librosa_provider_real_audio() -> None:
    """Verify provider runs on real speech audio if sample file is available."""
    real_audio_path = Path("D:/MyData/Victories-5/test/test_audio.wav")
    if not real_audio_path.is_file():
        pytest.skip("Sample test audio not found on disk")

    provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=5.0)
    observations = await provider.extract_metrics(real_audio_path)

    assert len(observations) > 0
    metrics = {obs.metric for obs in observations}
    assert "pitch_mean_hz" in metrics
    assert "speaking_rate_wpm" in metrics
