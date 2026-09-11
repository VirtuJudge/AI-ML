"""Fake acoustic and speech prosody metrics provider adapter."""

from pathlib import Path

from app.providers.types import AudioObservation

ALGORITHM_VERSION = "fake_librosa/0.0.0"


class FakeAudioMetricsProvider:
    """Deterministic audio prosody and acoustic feature provider for testing."""

    def __init__(self, *, duration_ms: int = 15000) -> None:
        self._duration_ms = duration_ms

    async def extract_metrics(
        self, audio_path: Path, *, source_artifact_id: str = ""
    ) -> list[AudioObservation]:
        observations: list[AudioObservation] = []
        artifact_id = source_artifact_id or audio_path.name
        sample_points = [
            (0, 5000, "speaking_rate_wpm", 138.0, "words_per_minute"),
            (0, 5000, "pitch_mean_hz", 145.2, "hertz"),
            (0, 5000, "pause_duration_ms", 250.0, "milliseconds"),
            (0, 5000, "filler_count", 0.0, "count"),
            (5000, 10000, "speaking_rate_wpm", 142.5, "words_per_minute"),
            (5000, 10000, "pitch_mean_hz", 152.0, "hertz"),
            (5000, 10000, "pause_duration_ms", 300.0, "milliseconds"),
            (5000, 10000, "filler_count", 1.0, "count"),
            (10000, 15000, "speaking_rate_wpm", 135.0, "words_per_minute"),
            (10000, 15000, "pitch_mean_hz", 148.5, "hertz"),
            (10000, 15000, "pause_duration_ms", 200.0, "milliseconds"),
            (10000, 15000, "filler_count", 0.0, "count"),
        ]
        for start, end, metric, value, unit in sample_points:
            if end <= self._duration_ms:
                observations.append(
                    AudioObservation(
                        start_ms=start,
                        end_ms=min(end, self._duration_ms),
                        metric=metric,
                        value=value,
                        unit=unit,
                        source_artifact_id=artifact_id,
                        algorithm_version=ALGORITHM_VERSION,
                    )
                )
        return observations


__all__ = ["ALGORITHM_VERSION", "FakeAudioMetricsProvider"]
