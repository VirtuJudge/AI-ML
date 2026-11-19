"""Unit tests for app.providers.librosa_audio."""

import math
import struct
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import combine_timed_evidence
from app.providers.base import AudioMetricsProvider
from app.providers.librosa_audio import (
    ALGORITHM_VERSION,
    LibrosaAudioProvider,
)
from app.providers.types import AudioObservation, DiarizationResult, SpeakerSegment
from app.stages.aggregation import aggregate_speaker_observations
from app.stages.scoring import calculate_individual_delivery_scores


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


@pytest.fixture
def synthetic_silence_wav(tmp_path: Path) -> Path:
    """Create a 2-second silent WAV audio file."""
    wav_path = tmp_path / "silence_2s.wav"
    sample_rate = 16000
    duration_s = 2.0
    n_samples = int(sample_rate * duration_s)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # All zeros
        wf.writeframes(bytearray(n_samples * 2))

    return wav_path


@pytest.fixture
def synthetic_filler_wav(tmp_path: Path) -> Path:
    """Create synthetic audio containing a vocalized hesitation (400ms 150Hz tone with silence)."""
    wav_path = tmp_path / "filler_hesitation.wav"
    sample_rate = 16000

    # 500ms silence + 400ms 150Hz tone + 500ms silence
    frames = bytearray()
    silence_500ms = int(sample_rate * 0.5)
    frames.extend(b"\x00\x00" * silence_500ms)

    tone_400ms = int(sample_rate * 0.4)
    for i in range(tone_400ms):
        sample = int(12000 * math.sin(2 * math.pi * 150.0 * i / sample_rate))
        frames.extend(struct.pack("<h", sample))

    frames.extend(b"\x00\x00" * silence_500ms)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
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
    assert "filler_count" in metrics

    # A pure 220Hz sine wave should yield mean pitch near 220Hz (+/- 10Hz)
    pitch_obs = metrics["pitch_mean_hz"]
    assert 210.0 <= pitch_obs.value <= 230.0

    for obs in observations:
        assert obs.algorithm_version == ALGORITHM_VERSION
        assert obs.start_ms <= obs.end_ms
        assert 0.0 <= obs.confidence <= 1.0


@pytest.mark.asyncio
async def test_librosa_provider_silence_produces_no_pitch_observations(
    synthetic_silence_wav: Path,
) -> None:
    """Verify that unvoiced/silent audio omits pitch metrics rather than inventing 0.0 Hz."""
    provider = LibrosaAudioProvider(window_sec=2.0, hop_sec=2.0)
    observations = await provider.extract_metrics(synthetic_silence_wav)

    assert len(observations) > 0
    metrics = {obs.metric: obs for obs in observations}

    # Crucial contract check: unvoiced silence MUST NOT emit pitch_mean_hz = 0.0
    assert "pitch_mean_hz" not in metrics
    assert "pitch_std_hz" not in metrics

    # Silence metrics must conform to expected tolerances (near 2000ms duration)
    assert "pause_duration_ms" in metrics
    pause_ms = metrics["pause_duration_ms"].value
    assert 1850.0 <= pause_ms <= 2150.0


@pytest.mark.asyncio
async def test_librosa_provider_synthetic_filler_detection(
    synthetic_filler_wav: Path,
) -> None:
    """Verify that vocalized hesitation (filler vowel) is detected within numeric tolerance."""
    provider = LibrosaAudioProvider(window_sec=2.0, hop_sec=2.0)
    observations = await provider.extract_metrics(synthetic_filler_wav)

    assert len(observations) > 0
    metrics = {obs.metric: obs for obs in observations}

    assert "filler_count" in metrics
    assert metrics["filler_count"].value >= 1.0

    # The 150 Hz tone should be captured near 150 Hz
    # (+/- 30 Hz tolerance accounting for window boundary)
    assert "pitch_mean_hz" in metrics
    assert 120.0 <= metrics["pitch_mean_hz"].value <= 165.0


@pytest.mark.asyncio
async def test_librosa_provider_real_audio() -> None:
    """Verify provider runs on real speech audio if sample file path is configured."""
    import os

    sample_env = os.getenv("TEST_SAMPLE_AUDIO")
    if not sample_env or not Path(sample_env).is_file():
        pytest.skip("No sample test audio path configured via TEST_SAMPLE_AUDIO")

    provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=5.0)
    observations = await provider.extract_metrics(Path(sample_env))

    assert len(observations) > 0
    metrics = {obs.metric for obs in observations}
    assert "pitch_mean_hz" in metrics
    assert "speaking_rate_wpm" in metrics


def _write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write normalized float samples [-1, 1] as 16-bit PCM mono WAV."""
    pcm = np.clip(samples * 16000.0, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _sine_segment(freq_hz: float, dur_s: float, *, amp: float = 0.8, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def _noise_segment(
    dur_s: float, *, seed: int = 42, amp: float = 0.45, sr: int = 16000
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * np.diff(rng.standard_normal(int(dur_s * sr) + 1))


@pytest.mark.asyncio
async def test_librosa_provider_no_double_counting_in_overlap(tmp_path: Path) -> None:
    """One pause and one filler inside [2500, 5000] overlap must each be counted once."""
    sr = 16000
    samples = np.concatenate(
        [
            _sine_segment(220.0, 2.7, sr=sr),
            np.zeros(int(1.0 * sr)),  # 2700-3700ms pause in overlap
            _sine_segment(150.0, 0.50, sr=sr),  # 3700-4200ms steady filler in overlap
            _noise_segment(0.45, seed=12, sr=sr),
            _sine_segment(220.0, 5.35, sr=sr),
        ]
    )
    wav_path = tmp_path / "overlap_event.wav"
    _write_mono_wav(wav_path, samples, sample_rate=sr)

    provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=2.5)
    observations = await provider.extract_metrics(wav_path)

    for metric_name in ("pause_duration_ms", "pause_count", "filler_count"):
        metric_obs = sorted(
            (o for o in observations if o.metric == metric_name),
            key=lambda o: (o.start_ms, o.end_ms),
        )
        assert len(metric_obs) >= 2
        for a, b in pairwise(metric_obs):
            assert a.end_ms <= b.start_ms, f"{metric_name} counting regions must be disjoint"

    pause_dur_obs = [o for o in observations if o.metric == "pause_duration_ms"]
    for o in pause_dur_obs:
        assert 0.0 <= o.value <= float(o.end_ms - o.start_ms)

    total_pause_ms = sum(o.value for o in pause_dur_obs)
    total_pause_count = sum(o.value for o in observations if o.metric == "pause_count")
    total_filler_count = sum(o.value for o in observations if o.metric == "filler_count")

    assert 750.0 <= total_pause_ms <= 1050.0
    assert total_pause_count == 1.0
    assert total_filler_count == 1.0


@pytest.mark.asyncio
async def test_librosa_provider_events_at_window_boundaries_and_hop_invariance(
    tmp_path: Path,
) -> None:
    """Events straddling 2500ms/5000ms boundaries must not be lost or double-counted across hops."""
    sr = 16000
    samples = np.concatenate(
        [
            _sine_segment(220.0, 2.025, sr=sr),
            _noise_segment(0.25, seed=21, sr=sr),
            _sine_segment(150.0, 0.45, sr=sr),  # 2275-2725ms filler straddling 2500ms
            _noise_segment(0.25, seed=22, sr=sr),
            _sine_segment(220.0, 1.8, sr=sr),
            np.zeros(int(0.45 * sr)),  # 4775-5225ms pause straddling 5000ms
            _sine_segment(220.0, 4.775, sr=sr),
        ]
    )
    wav_path = tmp_path / "boundary_events.wav"
    _write_mono_wav(wav_path, samples, sample_rate=sr)

    totals_by_hop: dict[float, tuple[float, float, float]] = {}
    for hop_sec in (1.0, 2.5, 5.0):
        provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=hop_sec)
        obs = await provider.extract_metrics(wav_path)
        totals_by_hop[hop_sec] = (
            sum(o.value for o in obs if o.metric == "pause_duration_ms"),
            sum(o.value for o in obs if o.metric == "pause_count"),
            sum(o.value for o in obs if o.metric == "filler_count"),
        )

    for hop_sec, (p_ms, p_cnt, f_cnt) in totals_by_hop.items():
        assert 250.0 <= p_ms <= 500.0, f"Unexpected pause_ms={p_ms} at hop={hop_sec}"
        assert p_cnt == 1.0, f"Unexpected pause_count={p_cnt} at hop={hop_sec}"
        assert f_cnt == 1.0, f"Unexpected filler_count={f_cnt} at hop={hop_sec}"

    # All hop sizes must agree identically on extensive event counts and durations
    reference = totals_by_hop[2.5]
    assert totals_by_hop[1.0] == reference
    assert totals_by_hop[5.0] == reference


@pytest.mark.asyncio
async def test_librosa_benchmark_against_synthetic_known_timings(tmp_path: Path) -> None:
    """Benchmark end-to-end audio -> aggregation -> delivery scoring against known timings."""
    sr = 16000
    # 15-second deterministic synthetic timeline:
    # - Pause 1: 2000-3000ms (1000ms)
    # - Filler 1: 3000-3500ms (500ms steady 150Hz, followed by 220ms <250ms gap)
    # - Pause 2: 8000-9500ms (1500ms)
    # - Filler 2: 11220-11720ms (500ms steady 150Hz, flanked by 220ms <250ms gaps)
    # Known ground truth: 2 pauses totaling 2500ms, 2 fillers.
    gap_220ms = np.zeros(int(0.22 * sr))
    samples = np.concatenate(
        [
            _sine_segment(220.0, 2.0, sr=sr),
            np.zeros(int(1.0 * sr)),  # Pause 1: 1000ms
            _sine_segment(150.0, 0.50, sr=sr),  # Filler 1: 500ms
            gap_220ms,
            _sine_segment(220.0, 4.28, sr=sr),
            np.zeros(int(1.5 * sr)),  # Pause 2: 1500ms
            _sine_segment(220.0, 1.50, sr=sr),
            gap_220ms,
            _sine_segment(150.0, 0.50, sr=sr),  # Filler 2: 500ms
            gap_220ms,
            _sine_segment(220.0, 3.06, sr=sr),
        ]
    )
    assert len(samples) == 15 * sr
    wav_path = tmp_path / "synthetic_benchmark_15s.wav"
    _write_mono_wav(wav_path, samples, sample_rate=sr)

    diarization = DiarizationResult(
        segments=[SpeakerSegment(start_ms=0, end_ms=15000, speaker_label="SPEAKER_00")],
        speaker_labels=["SPEAKER_00"],
    )

    scores_by_hop: dict[float, float] = {}
    for hop_sec in (1.0, 2.5, 5.0):
        provider = LibrosaAudioProvider(window_sec=5.0, hop_sec=hop_sec)
        raw_obs = await provider.extract_metrics(wav_path)
        _, attributed_audio, _, _ = combine_timed_evidence(diarization, [], raw_obs)
        by_speaker = aggregate_speaker_observations([], attributed_audio, diarization)

        profile = by_speaker["SPEAKER_00"]
        summary = profile["metrics_summary"]
        assert summary["total_pause_count"] == 2
        assert summary["total_filler_count"] == 2
        # Frame-length=2048 (128ms) boundary smoothing trims ~128ms per pause edge
        assert 2100.0 <= summary["total_pause_ms"] <= 2600.0
        assert summary["total_pause_ms"] <= profile["speaking_time_ms"]

        delivery_comps = calculate_individual_delivery_scores(profile)
        timing = next(c for c in delivery_comps if c.dimension == "timing_and_speech_mechanics")
        assert timing.status == "scored"
        assert timing.normalized_score is not None
        scores_by_hop[hop_sec] = timing.normalized_score

    # Timing scores across hop sizes must stay consistent (not inflated by hop overlap)
    assert scores_by_hop[1.0] == scores_by_hop[2.5] == scores_by_hop[5.0]
