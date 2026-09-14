"""Shared utilities and validators for VirtuJudge AI-ML pipeline stages."""

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from app.contracts import Limitation
from app.providers.types import AudioObservation, VisualObservation

logger = logging.getLogger(__name__)

T = TypeVar("T", VisualObservation, AudioObservation)
R = TypeVar("R")


class StageTransientError(Exception):
    """Raised when an external provider or I/O operation experiences a transient failure."""


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


async def with_transient_retries(
    operation: Callable[[], Coroutine[Any, Any, R]],
    *,
    stage_name: str,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        StageTransientError,
        TimeoutError,
        ConnectionError,
    ),
) -> R:
    """Execute an async operation with exponential backoff and jitter on retryable exceptions.

    Performs up to max_attempts attempts. On catch of retryable_exceptions:
    if attempt < max_attempts, sleeps (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.1)
    and retries. If attempt == max_attempts, re-raises the exception.
    Non-retryable exceptions are raised immediately without delay.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except retryable_exceptions as exc:
            if attempt >= max_attempts:
                logger.warning(
                    "Stage '%s' exhausted all %d retry attempts; re-raising %s: %s",
                    stage_name,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.1)
            logger.info(
                "Stage '%s' transient error on attempt %d/%d (%s: %s); retrying in %.2fs",
                stage_name,
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")


__all__ = [
    "StageTransientError",
    "validate_observation_timestamps",
    "with_transient_retries",
]
