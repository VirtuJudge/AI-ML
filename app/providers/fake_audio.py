"""Fake acoustic and speech prosody metrics provider adapter."""

from pathlib import Path

from app.providers.types import AudioObservation


class FakeAudioMetricsProvider:
    """Deterministic audio prosody and acoustic feature provider."""

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]:
        return [
            AudioObservation(
                timestamp_ms=1000,
                pitch_hz=145.2,
                speaking_rate_wpm=138.0,
                pause_duration_ms=250,
            ),
            AudioObservation(
                timestamp_ms=5000,
                pitch_hz=152.0,
                speaking_rate_wpm=142.5,
                pause_duration_ms=300,
            ),
            AudioObservation(
                timestamp_ms=10000,
                pitch_hz=148.5,
                speaking_rate_wpm=135.0,
                pause_duration_ms=200,
            ),
        ]


__all__ = ["FakeAudioMetricsProvider"]
