"""Audio acoustic measurement stage for VirtuJudge AI-ML.

Uses librosa (or a fake adapter) to extract timed acoustic measurements:
speaking rate, pause detection, filler words, and pitch variation.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import Limitation
from app.providers.base import AudioMetricsProvider
from app.providers.types import AudioObservation

from app.stages.common import validate_observation_timestamps

logger = logging.getLogger(__name__)


class AudioStageResult(BaseModel):
    """Result of the audio acoustic measurement stage."""

    observations: list[AudioObservation] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_audio_observations(
    observations: list[AudioObservation],
    media_duration_ms: int,
    *,
    source_artifact_id: str = "",
) -> tuple[list[AudioObservation], list[Limitation]]:
    """Clamp observation timestamps and report limitations if out of bounds."""
    return validate_observation_timestamps(
        observations,
        media_duration_ms,
        scope="audio_features",
        clamped_code="audio_observation_timestamp_clamped",
        clamped_message=(
            "One or more audio observations had timestamps exceeding "
            "media duration and were clamped."
        ),
        affected_dimensions=["delivery_pace", "acoustic_features"],
        source_artifact_id=source_artifact_id,
    )


async def run_audio_stage(
    audio_path: Path | str | None,
    audio_provider: AudioMetricsProvider,
    *,
    media_duration_ms: int,
    source_artifact_id: str = "",
    skip_file_check: bool = False,
) -> AudioStageResult:
    """Run the audio acoustic measurement stage.

    Args:
        audio_path: Path to normalized 16kHz mono WAV. None if audio missing.
        audio_provider: AudioMetricsProvider adapter (real or fake).
        media_duration_ms: Total media duration for timestamp validation.
        source_artifact_id: Identifier of the source audio artifact.
        skip_file_check: If True, bypass file existence check (useful for fake providers).

    Returns:
        AudioStageResult with timed observations and any limitations.
    """
    limitations: list[Limitation] = []

    if audio_path is None:
        limitations.append(
            Limitation(
                code="no_audio_stream",
                scope="audio_features",
                message="Input media has no audio stream. Acoustic measurements are unavailable.",
                affected_dimensions=["delivery_pace", "acoustic_features"],
            )
        )
        return AudioStageResult(limitations=limitations)

    audio_file = Path(audio_path)
    if not skip_file_check and not audio_file.is_file():
        logger.warning("Audio file not found for acoustic analysis: %s", audio_file)
        limitations.append(
            Limitation(
                code="audio_file_missing",
                scope="audio_features",
                message="Audio file was not available for acoustic measurement.",
                affected_dimensions=["delivery_pace", "acoustic_features"],
            )
        )
        return AudioStageResult(limitations=limitations)

    try:
        raw_observations = await audio_provider.extract_metrics(audio_file)
    except Exception:
        logger.warning("Audio metrics extraction failed.", exc_info=True)
        limitations.append(
            Limitation(
                code="audio_extraction_failed",
                scope="audio_features",
                message="Acoustic feature extraction failed; audio measurements are unavailable.",
                affected_dimensions=["delivery_pace", "acoustic_features"],
            )
        )
        return AudioStageResult(limitations=limitations)

    if not raw_observations:
        limitations.append(
            Limitation(
                code="no_audio_observations",
                scope="audio_features",
                message="No acoustic measurements could be extracted from the audio.",
                affected_dimensions=["delivery_pace", "acoustic_features"],
            )
        )
        return AudioStageResult(limitations=limitations)

    resolved_artifact_id = source_artifact_id or audio_file.name
    validated, ts_limitations = validate_audio_observations(
        raw_observations, media_duration_ms, source_artifact_id=resolved_artifact_id
    )
    limitations.extend(ts_limitations)

    metadata: dict[str, Any] = {
        "stage": "audio_features",
        "provider": audio_provider.__class__.__name__,
        "observation_count": len(validated),
        "media_duration_ms": media_duration_ms,
    }

    return AudioStageResult(
        observations=validated,
        limitations=limitations,
        metadata=metadata,
    )


__all__ = [
    "AudioStageResult",
    "run_audio_stage",
    "validate_audio_observations",
]
