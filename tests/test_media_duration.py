"""Media duration policy: probed duration wins, no invented default, explicit limitations."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.providers.fake_speech import FakeSpeechProvider
from app.stages.answers import run_answer_speech_stage
from app.stages.common import StageValidationError
from app.stages.media import DURATION_MISMATCH_TOLERANCE_MS, resolve_media_duration


def test_probed_duration_is_authoritative_without_limitation() -> None:
    duration, limitations = resolve_media_duration(
        probed_duration_ms=42_000, declared_duration_ms=None
    )
    assert duration == 42_000
    assert limitations == []


def test_declared_duration_used_when_probe_unavailable() -> None:
    duration, limitations = resolve_media_duration(
        probed_duration_ms=None, declared_duration_ms=30_000
    )
    assert duration == 30_000
    assert limitations == []


def test_missing_duration_stays_missing_with_limitation() -> None:
    duration, limitations = resolve_media_duration(
        probed_duration_ms=None, declared_duration_ms=None
    )
    assert duration is None
    assert [lim.code for lim in limitations] == ["media_duration_unavailable"]


@pytest.mark.parametrize(
    ("probed", "declared"),
    [(0, None), (0, 30_000), (-5, None), (None, 0), (None, -1)],
)
def test_zero_or_negative_duration_fails_validation(
    probed: int | None, declared: int | None
) -> None:
    with pytest.raises(StageValidationError):
        resolve_media_duration(probed_duration_ms=probed, declared_duration_ms=declared)


def test_mismatched_declared_duration_uses_probe_and_emits_limitation() -> None:
    duration, limitations = resolve_media_duration(
        probed_duration_ms=60_000,
        declared_duration_ms=60_000 + DURATION_MISMATCH_TOLERANCE_MS + 1,
    )
    assert duration == 60_000
    assert [lim.code for lim in limitations] == ["media_duration_mismatch"]


def test_declared_duration_within_tolerance_is_not_a_mismatch() -> None:
    duration, limitations = resolve_media_duration(
        probed_duration_ms=60_000,
        declared_duration_ms=60_000 + DURATION_MISMATCH_TOLERANCE_MS,
    )
    assert duration == 60_000
    assert limitations == []


async def test_answer_stage_without_any_duration_keeps_timestamps_and_flags_it(
    tmp_path: Path,
) -> None:
    """No declared duration and an unprobe-able file: no 15 s default, no clamping to zero."""
    audio = tmp_path / "answer.webm"
    audio.write_bytes(b"not a wav file")

    result = await run_answer_speech_stage(
        audio,
        FakeSpeechProvider(),
        media_duration_ms=None,
        skip_normalization=True,
        strict_timestamps=False,
    )

    assert [lim.code for lim in result.limitations] == ["media_duration_unavailable"]
    assert result.metadata["media_duration_ms"] is None
    segments = result.transcript.segments
    assert segments and segments[0].end_ms == 4500  # not clamped to an invented bound


async def test_answer_stage_rejects_zero_declared_duration(tmp_path: Path) -> None:
    audio = tmp_path / "answer.webm"
    audio.write_bytes(b"not a wav file")

    with pytest.raises(StageValidationError):
        await run_answer_speech_stage(
            audio,
            FakeSpeechProvider(),
            media_duration_ms=0,
            skip_normalization=True,
        )
