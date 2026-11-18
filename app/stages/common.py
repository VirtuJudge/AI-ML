"""Shared utilities and validators for VirtuJudge AI-ML pipeline stages."""

import asyncio
import logging
import random
import re
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import httpx

from app.contracts import MAX_QUESTION_TEXT_LENGTH, ErrorCode, Limitation
from app.providers.groq_pool import (
    AllKeysExhaustedError,
    GroqPoolError,
    GroqRateLimitError,
)
from app.providers.groq_speech import ProviderError
from app.providers.types import AudioObservation, VisualObservation

logger = logging.getLogger(__name__)

T = TypeVar("T", VisualObservation, AudioObservation)
R = TypeVar("R")

_STATUS_CODE_RE = re.compile(
    r"(?:http\s*|status\s*(?:code)?\s*[:=]?\s*|code\s*[:=]?\s*)\b(\d{3})\b"
    r"|\b(400|401|403|404|408|413|422|429|500|502|503|504|529)\b"
)
_TIMEOUT_KEYWORDS = (
    "timeout",
    "timed out",
    "deadline exceeded",
    "readtimeout",
    "connecttimeout",
    "pooltimeout",
    "writetimeout",
)
_TRANSIENT_KEYWORDS = (
    "exhausted",
    "rate limit",
    "ratelimit",
    "too many requests",
    "service unavailable",
    "temporarily unavailable",
    "bad gateway",
    "network error",
    "connection",
    "temporarily",
    "try again",
    "overloaded",
)


class StageError(Exception):
    """Base error raised by a pipeline stage carrying safe reporting metadata."""

    def __init__(
        self,
        message: str = "",
        *,
        stage: str | None = None,
        code: ErrorCode = ErrorCode.PROVIDER_ERROR,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self._stage = stage
        self.code = code
        self.retryable = retryable

    @property
    def stage(self) -> str:
        return self._stage or "speech"

    @stage.setter
    def stage(self, value: str) -> None:
        self._stage = value

    @property
    def has_explicit_stage(self) -> bool:
        return self._stage is not None


class StageTransientError(StageError):
    """Raised when an external provider or I/O operation experiences a transient failure."""

    def __init__(
        self,
        message: str = "",
        *,
        stage: str | None = None,
        code: ErrorCode = ErrorCode.PROVIDER_ERROR,
    ) -> None:
        super().__init__(message, stage=stage, code=code, retryable=True)


class StageProviderError(StageError):
    """Raised when an external provider fails with a non-retryable error."""

    def __init__(
        self,
        message: str = "",
        *,
        stage: str | None = None,
        code: ErrorCode = ErrorCode.PROVIDER_ERROR,
    ) -> None:
        super().__init__(message, stage=stage, code=code, retryable=False)


def _stage_error_from_status(
    status_code: int,
    *,
    stage_name: str,
    exc: BaseException,
) -> StageError:
    if status_code in (408, 504):
        return StageTransientError(
            f"Stage '{stage_name}' provider HTTP {status_code} timeout ({type(exc).__name__}).",
            stage=stage_name,
            code=ErrorCode.PROVIDER_TIMEOUT,
        )
    if status_code == 429 or 500 <= status_code < 600:
        return StageTransientError(
            f"Stage '{stage_name}' transient HTTP {status_code} failure ({type(exc).__name__}).",
            stage=stage_name,
            code=ErrorCode.PROVIDER_ERROR,
        )
    return StageProviderError(
        f"Stage '{stage_name}' non-retryable HTTP {status_code} failure ({type(exc).__name__}).",
        stage=stage_name,
        code=ErrorCode.PROVIDER_ERROR,
    )


def classify_stage_error(
    exc: BaseException,
    *,
    stage_name: str,
    _depth: int = 0,
) -> StageError | None:
    """Map provider, timeout, network, and pool failures to typed StageError instances."""
    if _depth > 5:
        return None

    if isinstance(exc, StageError):
        if not exc.has_explicit_stage:
            exc.stage = stage_name
        return exc

    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return StageTransientError(
            f"Stage '{stage_name}' timed out ({type(exc).__name__}).",
            stage=stage_name,
            code=ErrorCode.PROVIDER_TIMEOUT,
        )

    if isinstance(exc, (ConnectionError, httpx.TransportError, httpx.RequestError)):
        return StageTransientError(
            f"Stage '{stage_name}' network failure ({type(exc).__name__}).",
            stage=stage_name,
            code=ErrorCode.PROVIDER_ERROR,
        )

    if isinstance(exc, httpx.HTTPStatusError):
        return _stage_error_from_status(
            exc.response.status_code,
            stage_name=stage_name,
            exc=exc,
        )

    if isinstance(exc, (ProviderError, GroqPoolError)):
        if exc.is_timeout:
            return StageTransientError(
                f"Stage '{stage_name}' provider timeout ({type(exc).__name__}).",
                stage=stage_name,
                code=ErrorCode.PROVIDER_TIMEOUT,
            )
        if exc.retryable is True:
            code = (
                ErrorCode.PROVIDER_TIMEOUT
                if exc.status_code in (408, 504)
                else ErrorCode.PROVIDER_ERROR
            )
            return StageTransientError(
                f"Stage '{stage_name}' transient provider failure ({type(exc).__name__}).",
                stage=stage_name,
                code=code,
            )
        if exc.retryable is False:
            return StageProviderError(
                f"Stage '{stage_name}' non-retryable provider failure ({type(exc).__name__}).",
                stage=stage_name,
                code=ErrorCode.PROVIDER_ERROR,
            )
        if exc.status_code is not None:
            return _stage_error_from_status(
                exc.status_code,
                stage_name=stage_name,
                exc=exc,
            )
        if exc.__cause__ is not None:
            cause_mapped = classify_stage_error(
                exc.__cause__,
                stage_name=stage_name,
                _depth=_depth + 1,
            )
            if cause_mapped is not None:
                return cause_mapped

        msg = str(exc).lower()
        codes = [int(a or b) for a, b in _STATUS_CODE_RE.findall(msg)]
        if any(kw in msg for kw in _TIMEOUT_KEYWORDS) or any(c in (408, 504) for c in codes):
            return StageTransientError(
                f"Stage '{stage_name}' provider timeout ({type(exc).__name__}).",
                stage=stage_name,
                code=ErrorCode.PROVIDER_TIMEOUT,
            )
        if codes:
            for status_code in codes:
                if status_code == 429 or 500 <= status_code < 600:
                    return _stage_error_from_status(
                        status_code,
                        stage_name=stage_name,
                        exc=exc,
                    )
            return _stage_error_from_status(
                codes[-1],
                stage_name=stage_name,
                exc=exc,
            )
        if any(kw in msg for kw in _TRANSIENT_KEYWORDS):
            return StageTransientError(
                f"Stage '{stage_name}' transient provider failure ({type(exc).__name__}).",
                stage=stage_name,
                code=ErrorCode.PROVIDER_ERROR,
            )
        if isinstance(exc, (GroqRateLimitError, AllKeysExhaustedError)):
            return StageTransientError(
                f"Stage '{stage_name}' provider pool exhausted ({type(exc).__name__}).",
                stage=stage_name,
                code=ErrorCode.PROVIDER_ERROR,
            )
        return StageProviderError(
            f"Stage '{stage_name}' non-retryable provider failure ({type(exc).__name__}).",
            stage=stage_name,
            code=ErrorCode.PROVIDER_ERROR,
        )

    return None


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
        except Exception as exc:
            if isinstance(exc, StageProviderError):
                if not exc.has_explicit_stage:
                    exc.stage = stage_name
                raise

            mapped = classify_stage_error(exc, stage_name=stage_name)
            is_retryable = isinstance(exc, retryable_exceptions) or (
                mapped is not None and mapped.retryable
            )
            if not is_retryable:
                if mapped is not None:
                    raise mapped from exc
                raise

            if attempt >= max_attempts:
                logger.warning(
                    "Stage '%s' exhausted all %d retry attempts (%s)",
                    stage_name,
                    max_attempts,
                    type(exc).__name__,
                )
                if isinstance(exc, StageTransientError):
                    if not exc.has_explicit_stage:
                        exc.stage = stage_name
                    raise
                if mapped is not None:
                    raise mapped from exc
                raise

            delay = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.1)
            logger.info(
                "Stage '%s' transient error on attempt %d/%d (%s); retrying in %.2fs",
                stage_name,
                attempt,
                max_attempts,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")


def build_over_length_follow_up_limitation(rubric_dimension: str | None = None) -> Limitation:
    """Build a log-safe limitation when a generated follow-up exceeds the text length cap."""
    dim = rubric_dimension.strip() if rubric_dimension else ""
    return Limitation(
        code="over_length_follow_up_dropped",
        scope="answer_assessment",
        message=(
            f"Generated follow-up question exceeded {MAX_QUESTION_TEXT_LENGTH} "
            "characters and was dropped."
        ),
        affected_dimensions=[dim] if dim else [],
    )


__all__ = [
    "StageError",
    "StageProviderError",
    "StageTransientError",
    "build_over_length_follow_up_limitation",
    "classify_stage_error",
    "validate_observation_timestamps",
    "with_transient_retries",
]
