"""Fake computer vision provider adapter."""

from pathlib import Path

from app.providers.types import VisualObservation


class FakeVisionProvider:
    """Deterministic visual observation provider."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        return [
            VisualObservation(
                timestamp_ms=1000,
                gaze_direction="camera",
                posture="open",
                confidence=0.95,
            ),
            VisualObservation(
                timestamp_ms=5000,
                gaze_direction="slides",
                posture="neutral",
                confidence=0.92,
            ),
            VisualObservation(
                timestamp_ms=10000,
                gaze_direction="camera",
                posture="open",
                confidence=0.96,
            ),
        ]


__all__ = ["FakeVisionProvider"]
