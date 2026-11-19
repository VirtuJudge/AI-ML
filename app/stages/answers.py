"""Answer speech transcription stage for VirtuJudge AI-ML.

Orchestrates audio normalization and speech transcription for single-speaker
Q&A answers, validating word timestamps and producing AnswerSpeechResult.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import AssetInput, Limitation
from app.providers.base import SpeechProvider
from app.providers.types import TranscriptionResult
from app.stages.media import (
    get_media_temp_root,
    get_wav_metadata,
    normalize_media,
    resolve_media_duration,
)
from app.stages.speech import transcript_confidence_limitations, validate_word_timestamps

logger = logging.getLogger(__name__)


class AnswerSpeechResult(BaseModel):
    """Result of answer speech transcription stage."""

    transcript: TranscriptionResult = Field(
        default_factory=lambda: TranscriptionResult(full_text="", segments=[])
    )
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def run_answer_speech_stage(
    audio_path: Path | str | AssetInput | None,
    speech_provider: SpeechProvider,
    *,
    media_duration_ms: int | None = None,
    skip_normalization: bool = False,
    strict_timestamps: bool = True,
    output_wav_path: Path | str | None = None,
) -> AnswerSpeechResult:
    """Run answer speech stage: optionally normalize audio, transcribe, validate timestamps.

    Args:
        audio_path: Media path, AssetInput, or None (if answer skipped).
        speech_provider: Speech transcription provider.
        media_duration_ms: Known duration in ms. If omitted, extracted from audio file.
        skip_normalization: If True, skips ffmpeg normalization and uses audio_path directly.
        strict_timestamps: If True, raises ValueError when timestamps exceed media duration.
        output_wav_path: Optional destination path for normalized WAV.

    Returns:
        AnswerSpeechResult with transcription, limitations, and metadata.
    """
    if audio_path is None:
        return AnswerSpeechResult(
            transcript=TranscriptionResult(full_text="", segments=[]),
            limitations=[],
            metadata={"stage": "answer_speech", "skipped": True},
        )

    expected_duration: int | None = media_duration_ms
    if isinstance(audio_path, AssetInput):
        input_path = Path(audio_path.object_key)
        expected_duration = audio_path.duration_ms or expected_duration
    else:
        input_path = Path(audio_path)

    # 1. Normalize audio to 16kHz mono WAV (unless skipped)
    owned_temp_wav: Path | None = None
    probed_duration_ms: int | None = None
    if not skip_normalization:
        norm_result = await normalize_media(input_path, output_path=output_wav_path)
        audio_for_provider = norm_result.output_path
        probed_duration_ms = norm_result.duration_ms or None
        if output_wav_path is None and audio_for_provider.resolve() != input_path.resolve():
            owned_temp_wav = audio_for_provider
    else:
        audio_for_provider = input_path
        if expected_duration is None:
            try:
                probed_duration_ms = get_wav_metadata(input_path)[0] or None
            except Exception:
                probed_duration_ms = None

    # Probed duration wins; declared duration is the fallback; when neither is known the
    # duration stays unknown with an explicit limitation instead of an invented length.
    duration_ms, limitations = resolve_media_duration(
        probed_duration_ms=probed_duration_ms,
        declared_duration_ms=expected_duration,
        media_role="Answer",
        stage="speech",
    )
    try:
        # 2. Transcribe via speech provider (single speaker, no diarization)
        transcription = await speech_provider.transcribe(audio_for_provider)

        # 3. Validate word timestamps stay within media duration
        ts_limitations = validate_word_timestamps(
            transcription, duration_ms, strict=strict_timestamps
        )
        limitations.extend(ts_limitations)
        limitations.extend(transcript_confidence_limitations(transcription))
    finally:
        if owned_temp_wav is not None:
            owned_temp_wav.unlink(missing_ok=True)
            temp_root = get_media_temp_root()
            if owned_temp_wav.parent == temp_root:
                with contextlib.suppress(OSError):
                    if temp_root.is_dir() and not any(temp_root.iterdir()):
                        temp_root.rmdir()

    metadata: dict[str, Any] = {
        "stage": "answer_speech",
        "speech_provider": speech_provider.__class__.__name__,
        "speech_model": getattr(speech_provider, "model_name", "unknown"),
        "speech_metadata": getattr(transcription, "metadata", {}),
        "media_duration_ms": duration_ms,
    }

    return AnswerSpeechResult(
        transcript=transcription,
        limitations=limitations,
        metadata=metadata,
    )


__all__ = [
    "AnswerSpeechResult",
    "run_answer_speech_stage",
]
