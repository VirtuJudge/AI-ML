"""Vision measurement stage for VirtuJudge AI-ML.

Uses MediaPipe (or a fake adapter) to extract timed visual measurements:
gaze direction, head orientation, posture, and movement.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import Limitation
from app.providers.base import VisionProvider
from app.providers.types import VisualObservation

logger = logging.getLogger(__name__)


class VisionStageResult(BaseModel):
    """Result of the vision measurement stage."""

    observations: list[VisualObservation] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_visual_observations(
    observations: list[VisualObservation],
    media_duration_ms: int,
) -> tuple[list[VisualObservation], list[Limitation]]:
    """Clamp observation timestamps and report limitations if out of bounds."""
    valid: list[VisualObservation] = []
    limitations: list[Limitation] = []
    had_out_of_bounds = False

    for obs in observations:
        start_ms = obs.start_ms
        end_ms = obs.end_ms
        clamped = False

        if end_ms > media_duration_ms:
            end_ms = media_duration_ms
            clamped = True
        if start_ms > media_duration_ms:
            start_ms = media_duration_ms
            clamped = True
        if start_ms > end_ms:
            start_ms, end_ms = end_ms, start_ms
            clamped = True

        if clamped:
            had_out_of_bounds = True
            valid.append(obs.model_copy(update={"start_ms": start_ms, "end_ms": end_ms}))
        else:
            valid.append(obs)

    if had_out_of_bounds:
        limitations.append(
            Limitation(
                code="vision_observation_timestamp_clamped",
                scope="vision",
                message=(
                    "One or more visual observations had timestamps exceeding "
                    "media duration and were clamped."
                ),
                affected_dimensions=["visual_delivery"],
            )
        )
    return valid, limitations


def check_coverage_limitations(
    observations: list[VisualObservation],
    media_duration_ms: int,
) -> list[Limitation]:
    """Check if face/body observations cover enough of the video."""
    limitations: list[Limitation] = []

    face_metrics = {"gaze_direction", "head_pitch_degrees", "head_yaw_degrees"}
    body_metrics = {"shoulder_symmetry_ratio", "posture_openness"}

    has_face = any(obs.metric in face_metrics for obs in observations)
    has_body = any(obs.metric in body_metrics for obs in observations)

    if not has_face and media_duration_ms > 0:
        limitations.append(
            Limitation(
                code="no_face_detected",
                scope="vision",
                message=(
                    "No face landmarks could be detected in the video. "
                    "Face-based measurements are unavailable."
                ),
                affected_dimensions=["visual_delivery", "eye_contact"],
            )
        )

    if not has_body and media_duration_ms > 0:
        limitations.append(
            Limitation(
                code="no_body_detected",
                scope="vision",
                message=(
                    "No body pose could be detected in the video. "
                    "Posture measurements are unavailable."
                ),
                affected_dimensions=["visual_delivery", "body_language"],
            )
        )

    return limitations


async def run_vision_stage(
    video_path: Path | str | None,
    vision_provider: VisionProvider,
    *,
    media_duration_ms: int,
) -> VisionStageResult:
    """Run the vision measurement stage.

    Args:
        video_path: Path to video file (from split_media). None if audio-only.
        vision_provider: VisionProvider adapter (real or fake).
        media_duration_ms: Total media duration for timestamp validation.

    Returns:
        VisionStageResult with timed observations and any limitations.
    """
    limitations: list[Limitation] = []

    if video_path is None:
        limitations.append(
            Limitation(
                code="no_video_stream",
                scope="vision",
                message="Input media has no video stream. Visual measurements are unavailable.",
                affected_dimensions=["visual_delivery", "eye_contact", "body_language"],
            )
        )
        return VisionStageResult(limitations=limitations)

    video_file = Path(video_path)
    if not video_file.is_file():
        logger.warning("Video file not found for vision analysis: %s", video_file)
        limitations.append(
            Limitation(
                code="video_file_missing",
                scope="vision",
                message="Video file was not available for visual measurement.",
                affected_dimensions=["visual_delivery", "eye_contact", "body_language"],
            )
        )
        return VisionStageResult(limitations=limitations)

    try:
        raw_observations = await vision_provider.analyze_video(video_file)
    except Exception:
        logger.warning("Vision analysis failed.", exc_info=True)
        limitations.append(
            Limitation(
                code="vision_analysis_failed",
                scope="vision",
                message="Visual analysis failed; vision measurements are unavailable.",
                affected_dimensions=["visual_delivery", "eye_contact", "body_language"],
            )
        )
        return VisionStageResult(limitations=limitations)

    if not raw_observations:
        limitations.append(
            Limitation(
                code="no_visual_observations",
                scope="vision",
                message="No visual measurements could be extracted from the video.",
                affected_dimensions=["visual_delivery", "eye_contact", "body_language"],
            )
        )
        return VisionStageResult(limitations=limitations)

    validated, ts_limitations = validate_visual_observations(raw_observations, media_duration_ms)
    limitations.extend(ts_limitations)

    coverage_limitations = check_coverage_limitations(validated, media_duration_ms)
    limitations.extend(coverage_limitations)

    metadata: dict[str, Any] = {
        "stage": "vision",
        "provider": vision_provider.__class__.__name__,
        "observation_count": len(validated),
        "media_duration_ms": media_duration_ms,
    }

    return VisionStageResult(
        observations=validated,
        limitations=limitations,
        metadata=metadata,
    )


__all__ = [
    "VisionStageResult",
    "check_coverage_limitations",
    "run_vision_stage",
    "validate_visual_observations",
]
