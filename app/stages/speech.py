"""Speech transcription and diarization stage for VirtuJudge AI-ML.

Orchestrates audio normalization, parallel Whisper transcription, and pyannote
diarization, validating word timestamps and diarization speaker turns against
actual media duration (`0 <= start < end <= media_duration_ms`) and combining
results into SpeechStageResult.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import AnalyzeSessionPayload, AssetInput, Limitation
from app.providers.base import DiarizationProvider, SpeechProvider
from app.providers.types import DiarizationResult, SpeakerSegment, TranscriptionResult
from app.stages.media import get_wav_metadata, normalize_media

logger = logging.getLogger(__name__)

SPEECH_STAGE_VERSION = "1.1.0"

_INVALID_SPEAKER_LABELS = frozenset({"SPEAKER_UNKNOWN", "UNKNOWN", ""})


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


def _reconcile_speaker_labels(
    existing_labels: Sequence[str],
    segments: Sequence[SpeakerSegment],
) -> list[str]:
    """Reconcile speaker_labels to valid anonymous labels present in surviving turns."""
    present_labels: list[str] = []
    seen_present: set[str] = set()
    for seg in segments:
        label = seg.speaker_label.strip()
        if label and label not in _INVALID_SPEAKER_LABELS and label not in seen_present:
            seen_present.add(label)
            present_labels.append(label)

    reconciled: list[str] = []
    seen_reconciled: set[str] = set()
    for raw_label in existing_labels:
        label = raw_label.strip()
        if label in seen_present and label not in seen_reconciled:
            seen_reconciled.add(label)
            reconciled.append(label)

    for label in present_labels:
        if label not in seen_reconciled:
            seen_reconciled.add(label)
            reconciled.append(label)

    return reconciled


def validate_diarization_timestamps(
    diarization: DiarizationResult,
    media_duration_ms: int | None,
    *,
    strict: bool = True,
) -> tuple[DiarizationResult, list[Limitation]]:
    """Validate speaker turns against `0 <= start_ms < end_ms <= media_duration_ms`.

    Policy:
        Every diarization speaker turn must represent a strictly positive interval
        within the actual media duration (`0 <= start_ms < end_ms <= media_duration_ms`,
        or `0 <= start_ms < end_ms` when media duration is unavailable).

        - In strict mode (`strict=True`), any negative endpoint, inverted interval
          (`start_ms > end_ms`), zero-length interval (`start_ms == end_ms`), or
          interval starting at/beyond or ending beyond `media_duration_ms` raises
          `ValueError`.
        - In permissive mode (`strict=False`), inverted endpoints are swapped before
          bounds checking, endpoints are clamped to `[0, media_duration_ms]`,
          originally zero-length turns or turns collapsing to `start_ms >= end_ms`
          (such as turns starting at/beyond `media_duration_ms` or boundary-rounded
          to zero length) are dropped, and explicit log-safe `Limitation` records are
          emitted with `scope="diarization"`.
        - After validation, `speaker_labels` is reconciled to only valid anonymous
          speaker labels that appear in surviving turns.

    Args:
        diarization: Raw diarization result from a DiarizationProvider.
        media_duration_ms: Actual media duration in milliseconds, or None if unknown.
        strict: If True, reject invalid intervals by raising ValueError.

    Returns:
        Tuple of (validated DiarizationResult with reconciled speaker_labels, limitations).

    Raises:
        ValueError: If `media_duration_ms` is negative, or if `strict=True` and any
            turn violates `0 <= start_ms < end_ms <= media_duration_ms`.
    """
    if media_duration_ms is not None and media_duration_ms < 0:
        raise ValueError(f"media_duration_ms ({media_duration_ms}ms) cannot be negative.")

    if strict:
        for seg in diarization.segments:
            if seg.start_ms < 0:
                raise ValueError(
                    f"Diarization turn start timestamp ({seg.start_ms}ms) cannot be negative."
                )
            if seg.end_ms < 0:
                raise ValueError(
                    f"Diarization turn end timestamp ({seg.end_ms}ms) cannot be negative."
                )
            if seg.start_ms > seg.end_ms:
                raise ValueError(
                    f"Diarization turn start timestamp ({seg.start_ms}ms) exceeds "
                    f"end timestamp ({seg.end_ms}ms)."
                )
            if seg.start_ms == seg.end_ms:
                raise ValueError(
                    f"Diarization turn must have positive duration "
                    f"(start={seg.start_ms}ms, end={seg.end_ms}ms)."
                )
            if media_duration_ms is not None:
                if seg.start_ms >= media_duration_ms:
                    raise ValueError(
                        f"Diarization turn start timestamp ({seg.start_ms}ms) must be "
                        f"less than media duration ({media_duration_ms}ms)."
                    )
                if seg.end_ms > media_duration_ms:
                    raise ValueError(
                        f"Diarization turn end timestamp ({seg.end_ms}ms) exceeds "
                        f"media duration ({media_duration_ms}ms)."
                    )

        reconciled_labels = _reconcile_speaker_labels(
            diarization.speaker_labels, diarization.segments
        )
        updated_meta = dict(diarization.metadata)
        if "detected_speaker_count" in updated_meta:
            updated_meta["detected_speaker_count"] = len(reconciled_labels)
        validated = diarization.model_copy(
            update={
                "segments": list(diarization.segments),
                "speaker_labels": reconciled_labels,
                "metadata": updated_meta,
            }
        )
        return validated, []

    valid_segments: list[SpeakerSegment] = []
    limitations: list[Limitation] = []
    has_out_of_bounds = False
    has_inverted = False
    has_zero_length = False

    for seg in diarization.segments:
        start_ms = seg.start_ms
        end_ms = seg.end_ms

        if start_ms > end_ms:
            start_ms, end_ms = end_ms, start_ms
            has_inverted = True

        if start_ms == end_ms:
            if start_ms < 0 or (media_duration_ms is not None and start_ms >= media_duration_ms):
                has_out_of_bounds = True
            else:
                has_zero_length = True
            continue

        if start_ms < 0:
            start_ms = 0
            has_out_of_bounds = True

        if end_ms < 0:
            end_ms = 0
            has_out_of_bounds = True

        if media_duration_ms is not None:
            if start_ms >= media_duration_ms:
                has_out_of_bounds = True
                continue
            if end_ms > media_duration_ms:
                end_ms = media_duration_ms
                has_out_of_bounds = True

        if start_ms >= end_ms:
            has_out_of_bounds = True
            continue

        if start_ms != seg.start_ms or end_ms != seg.end_ms:
            valid_segments.append(seg.model_copy(update={"start_ms": start_ms, "end_ms": end_ms}))
        else:
            valid_segments.append(seg)

    if has_out_of_bounds:
        limitations.append(
            Limitation(
                code="diarization_timestamp_out_of_bounds",
                scope="diarization",
                message=(
                    "One or more diarization speaker turns fell outside media duration "
                    "bounds and were clamped or dropped."
                ),
                affected_dimensions=["individual_feedback", "speaker_attribution"],
            )
        )

    if has_inverted:
        limitations.append(
            Limitation(
                code="diarization_inverted_timestamps",
                scope="diarization",
                message=(
                    "One or more diarization speaker turns had start timestamps "
                    "exceeding end timestamps and were corrected."
                ),
                affected_dimensions=["individual_feedback", "speaker_attribution"],
            )
        )

    if has_zero_length:
        limitations.append(
            Limitation(
                code="diarization_zero_length_turns",
                scope="diarization",
                message=(
                    "One or more diarization speaker turns had zero duration and were dropped."
                ),
                affected_dimensions=["individual_feedback", "speaker_attribution"],
            )
        )

    reconciled_labels = _reconcile_speaker_labels(diarization.speaker_labels, valid_segments)
    updated_meta = dict(diarization.metadata)
    if "detected_speaker_count" in updated_meta:
        updated_meta["detected_speaker_count"] = len(reconciled_labels)

    validated = diarization.model_copy(
        update={
            "segments": valid_segments,
            "speaker_labels": reconciled_labels,
            "metadata": updated_meta,
        }
    )
    return validated, limitations


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
    duration_known = True
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
                duration_known = False

    if expected_duration is not None and duration_ms == 0:
        duration_ms = expected_duration
        duration_known = True

    known_duration_ms: int | None = duration_ms if duration_known else None

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
        # Fallback: create a single SPEAKER_00 label assigned to valid transcript segments
        fallback_segments = [
            SpeakerSegment(
                start_ms=t_seg.start_ms,
                end_ms=t_seg.end_ms,
                speaker_label="SPEAKER_00",
            )
            for t_seg in transcription.segments
            if 0 <= t_seg.start_ms < t_seg.end_ms
            and (known_duration_ms is None or t_seg.end_ms <= known_duration_ms)
        ]
        speaker_labels = ["SPEAKER_00"] if fallback_segments else []
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
        diarization, diar_limitations = validate_diarization_timestamps(
            diarization_res,
            known_duration_ms,
            strict=strict_timestamps,
        )
        limitations.extend(diar_limitations)

        # 4. Check for missing/uncertain speaker assignments
        uncertain_in_diarization = diarization.metadata.get("uncertain_turns_count", 0)
        has_unknown_speaker_segment = any(
            s.speaker_label in _INVALID_SPEAKER_LABELS for s in diarization.segments
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
            if best_overlap == 0 or best_speaker is None or best_speaker in _INVALID_SPEAKER_LABELS:
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

        # Reconciled valid anonymous speaker labels
        valid_speaker_labels = list(diarization.speaker_labels)

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
    "SPEECH_STAGE_VERSION",
    "SpeechStageResult",
    "run_speech_stage",
    "validate_diarization_timestamps",
    "validate_word_timestamps",
]
