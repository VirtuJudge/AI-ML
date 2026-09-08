"""Speech transcription and diarization stage for VirtuJudge AI-ML.

Orchestrates audio normalization, parallel Whisper transcription, and pyannote
diarization, validating word timestamps and combining results into SpeechStageResult.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import AnalyzeSessionPayload, AssetInput, Limitation
from app.providers.base import DiarizationProvider, SpeechProvider
from app.providers.types import DiarizationResult, SpeakerSegment, TranscriptionResult
from app.stages.media import get_wav_metadata, normalize_media

logger = logging.getLogger(__name__)


class SpeechStageResult(BaseModel):
    """Result of speech transcription and diarization stage."""

    transcription: TranscriptionResult
    diarization: DiarizationResult
    speaker_labels: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_word_timestamps(
    transcription: TranscriptionResult,
    media_duration_ms: int,
    *,
    strict: bool = True,
) -> list[Limitation]:
    """Validate that word timestamps and segment bounds stay within media duration and start <= end.

    Args:
        transcription: Transcription result containing segments and word timestamps.
        media_duration_ms: Total media duration in milliseconds.
        strict: If True, raises ValueError if a timestamp exceeds duration or is inverted.
                If False, clamps timestamps to media duration, corrects inverted intervals,
                and produces Limitations.

    Returns:
        List of limitations if any timestamps were out of bounds or inverted in non-strict mode.

    Raises:
        ValueError: If strict is True and any timestamp is negative, exceeds media_duration_ms,
                    or has start_ms > end_ms.
    """
    limitations: list[Limitation] = []
    has_out_of_bounds = False
    has_inverted = False

    for seg in transcription.segments:
        if seg.start_ms < 0:
            if strict:
                raise ValueError(f"Segment start timestamp ({seg.start_ms}ms) cannot be negative.")
            seg.start_ms = 0
            has_out_of_bounds = True

        if seg.end_ms > media_duration_ms:
            if strict:
                raise ValueError(
                    f"Segment end timestamp ({seg.end_ms}ms) exceeds "
                    f"media duration ({media_duration_ms}ms)."
                )
            seg.end_ms = media_duration_ms
            has_out_of_bounds = True

        if seg.start_ms > media_duration_ms:
            if strict:
                raise ValueError(
                    f"Segment start timestamp ({seg.start_ms}ms) exceeds "
                    f"media duration ({media_duration_ms}ms)."
                )
            seg.start_ms = media_duration_ms
            has_out_of_bounds = True

        if seg.start_ms > seg.end_ms:
            if strict:
                raise ValueError(
                    f"Segment start timestamp ({seg.start_ms}ms) exceeds "
                    f"end timestamp ({seg.end_ms}ms)."
                )
            seg.start_ms, seg.end_ms = (
                min(seg.start_ms, seg.end_ms),
                max(seg.start_ms, seg.end_ms),
            )
            has_inverted = True

        for word in seg.words:
            if word.start_ms < 0:
                if strict:
                    raise ValueError(
                        f"Word '{word.word}' start timestamp ({word.start_ms}ms) "
                        "cannot be negative."
                    )
                word.start_ms = 0
                has_out_of_bounds = True

            if word.end_ms > media_duration_ms:
                if strict:
                    raise ValueError(
                        f"Word '{word.word}' end timestamp ({word.end_ms}ms) "
                        f"exceeds media duration ({media_duration_ms}ms)."
                    )
                word.end_ms = media_duration_ms
                has_out_of_bounds = True

            if word.start_ms > media_duration_ms:
                if strict:
                    raise ValueError(
                        f"Word '{word.word}' start timestamp ({word.start_ms}ms) "
                        f"exceeds media duration ({media_duration_ms}ms)."
                    )
                word.start_ms = media_duration_ms
                has_out_of_bounds = True

            if word.start_ms > word.end_ms:
                if strict:
                    raise ValueError(
                        f"Word '{word.word}' start timestamp ({word.start_ms}ms) "
                        f"exceeds end timestamp ({word.end_ms}ms)."
                    )
                word.start_ms, word.end_ms = (
                    min(word.start_ms, word.end_ms),
                    max(word.start_ms, word.end_ms),
                )
                has_inverted = True

    if has_out_of_bounds and not strict:
        limitations.append(
            Limitation(
                code="timestamp_out_of_bounds",
                scope="speech",
                message="Word or segment timestamps exceeded media duration and were clamped.",
                affected_dimensions=["delivery_pace"],
            )
        )

    if has_inverted and not strict:
        limitations.append(
            Limitation(
                code="inverted_timestamps",
                scope="speech",
                message=(
                    "Word or segment start timestamps exceeded end timestamps and were corrected."
                ),
                affected_dimensions=["delivery_pace"],
            )
        )

    return limitations


async def run_speech_stage(
    target: Path | str | AnalyzeSessionPayload | AssetInput,
    speech_provider: SpeechProvider,
    diarization_provider: DiarizationProvider,
    *,
    output_wav_path: Path | str | None = None,
    media_duration_ms: int | None = None,
    strict_timestamps: bool = True,
    skip_normalization: bool = False,
) -> SpeechStageResult:
    """Run speech stage: normalize audio, transcribe & diarize in parallel, combine results.

    Args:
        target: Media path, AnalyzeSessionPayload, or AssetInput.
        speech_provider: SpeechProvider (e.g. WhisperSpeechProvider or FakeSpeechProvider).
        diarization_provider: DiarizationProvider (e.g. PyannoteDiarizationProvider).
        output_wav_path: Optional output path for normalized audio.
        media_duration_ms: Known duration in ms. If omitted, extracted from normalized WAV.
        strict_timestamps: If True, raises ValueError when timestamps exceed duration.
        skip_normalization: If True, skips ffmpeg normalization and uses target path directly.

    Returns:
        SpeechStageResult with combined transcription, diarization, speaker labels, limitations.
    """
    expected_duration: int | None = media_duration_ms
    if isinstance(target, AnalyzeSessionPayload):
        input_path = Path(target.presentation.object_key)
        expected_duration = target.presentation.duration_ms or expected_duration
    elif isinstance(target, AssetInput):
        input_path = Path(target.object_key)
        expected_duration = target.duration_ms or expected_duration
    else:
        input_path = Path(target)

    # 1. Normalize audio to 16kHz mono WAV (unless skipped)
    if not skip_normalization:
        norm_result = await normalize_media(input_path, output_path=output_wav_path)
        audio_for_providers = norm_result.output_path
        duration_ms = norm_result.duration_ms
    else:
        audio_for_providers = input_path
        if expected_duration is not None:
            duration_ms = expected_duration
        else:
            try:
                duration_ms, _, _ = get_wav_metadata(input_path)
            except Exception:
                duration_ms = 0

    if expected_duration is not None and duration_ms == 0:
        duration_ms = expected_duration

    # 2. Transcribe and diarize in parallel (with graceful diarization fallback)
    async def _safe_diarize() -> tuple[DiarizationResult | None, bool]:
        try:
            res = await diarization_provider.diarize(audio_for_providers)
            return res, False
        except Exception:
            logger.warning(
                "Speaker diarization failed; falling back to single speaker attribution."
            )
            return None, True

    transcription, (diarization_res, diarization_failed) = await asyncio.gather(
        speech_provider.transcribe(audio_for_providers),
        _safe_diarize(),
    )

    limitations: list[Limitation] = []

    # 3. Validate word timestamps stay within media duration
    if duration_ms > 0:
        ts_limitations = validate_word_timestamps(
            transcription, duration_ms, strict=strict_timestamps
        )
        limitations.extend(ts_limitations)

    if diarization_failed or diarization_res is None:
        # Fallback: create a single SPEAKER_00 label assigned to all transcript segments
        fallback_segments = [
            SpeakerSegment(
                start_ms=t_seg.start_ms,
                end_ms=t_seg.end_ms,
                speaker_label="SPEAKER_00",
            )
            for t_seg in transcription.segments
        ]
        speaker_labels = ["SPEAKER_00"]
        diarization = DiarizationResult(
            segments=fallback_segments,
            speaker_labels=speaker_labels,
            metadata={"fallback": True},
        )
        valid_speaker_labels = speaker_labels
        limitations.append(
            Limitation(
                code="diarization_unavailable",
                scope="diarization",
                message=(
                    "Speaker diarization was unavailable; "
                    "all speech attributed to a single speaker."
                ),
                affected_dimensions=["individual_feedback", "speaker_attribution"],
            )
        )
    else:
        diarization = diarization_res

        # 4. Check for missing/uncertain speaker assignments
        uncertain_in_diarization = diarization.metadata.get("uncertain_turns_count", 0)
        has_unknown_speaker_segment = any(
            s.speaker_label in ("SPEAKER_UNKNOWN", "", "UNKNOWN") for s in diarization.segments
        )

        # Align speaker turns with transcript segments to identify unassigned speech
        unassigned_transcript_segments = 0
        for t_seg in transcription.segments:
            best_overlap = 0
            best_speaker = None
            for d_seg in diarization.segments:
                overlap = max(
                    0,
                    min(t_seg.end_ms, d_seg.end_ms) - max(t_seg.start_ms, d_seg.start_ms),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d_seg.speaker_label
            if best_overlap == 0 or best_speaker in (None, "SPEAKER_UNKNOWN", "", "UNKNOWN"):
                unassigned_transcript_segments += 1

        if (
            uncertain_in_diarization > 0
            or has_unknown_speaker_segment
            or (unassigned_transcript_segments > 0 and len(diarization.segments) > 0)
        ):
            limitations.append(
                Limitation(
                    code="uncertain_speaker_assignment",
                    scope="diarization",
                    message=(
                        "Diarization produced unassigned or ambiguous speaker turns. "
                        "Individual speaker attribution may be incomplete."
                    ),
                    affected_dimensions=["individual_feedback", "speaker_attribution"],
                )
            )

        # Filter speaker labels to valid anonymous identifiers
        valid_speaker_labels = [
            label
            for label in diarization.speaker_labels
            if label not in ("SPEAKER_UNKNOWN", "UNKNOWN", "")
        ]

        if not valid_speaker_labels and len(transcription.segments) > 0:
            limitations.append(
                Limitation(
                    code="no_speakers_detected",
                    scope="diarization",
                    message="No distinct speaker labels could be assigned to speech segments.",
                    affected_dimensions=["individual_feedback"],
                )
            )

    metadata: dict[str, Any] = {
        "stage": "speech",
        "speech_provider": speech_provider.__class__.__name__,
        "speech_model": getattr(speech_provider, "model_name", "unknown"),
        "speech_metadata": transcription.metadata,
        "diarization_provider": diarization_provider.__class__.__name__,
        "diarization_model": getattr(diarization_provider, "model_name", "unknown"),
        "diarization_metadata": diarization.metadata,
        "media_duration_ms": duration_ms,
    }

    return SpeechStageResult(
        transcription=transcription,
        diarization=diarization,
        speaker_labels=valid_speaker_labels,
        limitations=limitations,
        metadata=metadata,
    )


__all__ = [
    "SpeechStageResult",
    "run_speech_stage",
    "validate_word_timestamps",
]
