"""Shared utilities and validators for VirtuJudge AI-ML pipeline stages."""

from typing import Any, TypeVar

from app.contracts import Limitation
from app.providers.types import AudioObservation, VisualObservation

T = TypeVar("T", VisualObservation, AudioObservation)


def validate_observation_timestamps(
    observations: list[T],
    media_duration_ms: int,
    *,
    scope: str,
    clamped_code: str,
    clamped_message: str,
    affected_dimensions: list[str],
    source_artifact_id: str = "",
) -> tuple[list[T], list[Limitation]]:
    """Validate observation timestamps, clamping or discarding out-of-bounds intervals.

    Guarantees the contract invariant: 0 <= start_ms < end_ms <= media_duration_ms.
    Any observation with start_ms >= media_duration_ms, or where start_ms >= end_ms
    (after boundary correction), is discarded rather than collapsed into a degenerate
    zero-duration interval.

    Also ensures source_artifact_id is populated if provided or absent.
    """
    valid: list[T] = []
    limitations: list[Limitation] = []
    had_clamping = False

    for obs in observations:
        start_ms = obs.start_ms
        end_ms = obs.end_ms
        clamped = False

        # Inverted interval correction (start > end)
        if start_ms > end_ms:
            start_ms, end_ms = end_ms, start_ms
            clamped = True

        # Clamp negative start to 0
        if start_ms < 0:
            start_ms = 0
            clamped = True

        # Discard observations that start at or beyond media duration
        if media_duration_ms > 0 and start_ms >= media_duration_ms:
            had_clamping = True
            continue

        # Clamp end to media duration
        if media_duration_ms > 0 and end_ms > media_duration_ms:
            end_ms = media_duration_ms
            clamped = True

        # Strict contract invariant: start_ms must be strictly less than end_ms
        if start_ms >= end_ms:
            had_clamping = True
            continue

        updates: dict[str, Any] = {}
        if clamped:
            had_clamping = True
            updates["start_ms"] = start_ms
            updates["end_ms"] = end_ms

        if source_artifact_id and not obs.source_artifact_id:
            updates["source_artifact_id"] = source_artifact_id

        if updates:
            valid.append(obs.model_copy(update=updates))
        else:
            valid.append(obs)

    if had_clamping:
        limitations.append(
            Limitation(
                code=clamped_code,
                scope=scope,
                message=clamped_message,
                affected_dimensions=affected_dimensions,
            )
        )

    return valid, limitations


__all__ = ["validate_observation_timestamps"]
