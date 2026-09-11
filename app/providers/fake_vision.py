"""Fake computer vision provider adapter."""

from pathlib import Path

from app.providers.types import VisualObservation

ALGORITHM_VERSION = "fake_mediapipe_pose/0.0.0"


class FakeVisionProvider:
    """Deterministic visual observation provider for testing."""

    def __init__(self, *, duration_ms: int = 15000) -> None:
        self._duration_ms = duration_ms

    async def analyze_video(
        self, video_path: Path, *, source_artifact_id: str = ""
    ) -> list[VisualObservation]:
        # Generate observations at 0s, 5s, 10s if they fit within duration
        observations: list[VisualObservation] = []
        artifact_id = source_artifact_id or video_path.name
        sample_points = [
            (0, 1000, "gaze_direction", 1.0, "categorical_index", 0.95),  # 1 = camera
            (0, 1000, "head_pitch_degrees", -2.5, "degrees", 0.92),
            (0, 1000, "shoulder_symmetry_ratio", 0.96, "ratio", 0.90),
            (5000, 6000, "gaze_direction", 2.0, "categorical_index", 0.92),  # 2 = slides
            (5000, 6000, "head_pitch_degrees", 15.3, "degrees", 0.88),
            (5000, 6000, "shoulder_symmetry_ratio", 0.91, "ratio", 0.87),
            (10000, 11000, "gaze_direction", 1.0, "categorical_index", 0.96),
            (10000, 11000, "head_pitch_degrees", -1.2, "degrees", 0.94),
            (10000, 11000, "shoulder_symmetry_ratio", 0.98, "ratio", 0.93),
        ]
        for start, end, metric, value, unit, conf in sample_points:
            if end <= self._duration_ms:
                observations.append(
                    VisualObservation(
                        start_ms=start,
                        end_ms=min(end, self._duration_ms),
                        metric=metric,
                        value=value,
                        unit=unit,
                        confidence=conf,
                        source_artifact_id=artifact_id,
                        algorithm_version=ALGORITHM_VERSION,
                    )
                )
        return observations


__all__ = ["ALGORITHM_VERSION", "FakeVisionProvider"]
