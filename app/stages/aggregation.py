"""Deterministic speaker pre-aggregation and observation filtering stage.

Pre-aggregates acoustic and visual observations into deterministic 10-second
windowed profiles per presenter, filtering out passive non-speaking timestamps
and cleaning acoustic micro-gaps before any downstream evaluation or LLM invocation.
"""

from collections.abc import Sequence
import logging
from typing import Any, NamedTuple, TypeVar

from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    SpeakerSegment,
    VisualObservation,
)

logger = logging.getLogger(__name__)

ObsT = TypeVar("ObsT", VisualObservation, AudioObservation)

ACOUSTIC_MEAN_METRICS = {
    "speaking_rate_wpm",
    "pitch_mean_hz",
    "pitch_std_hz",
}

ACOUSTIC_SUM_METRICS = {
    "pause_duration_ms",
    "pause_count",
    "filler_count",
}

VISUAL_MEAN_METRICS = {
    "gaze_direction",
    "head_pitch_degrees",
    "head_yaw_degrees",
    "posture_openness",
    "shoulder_symmetry_ratio",
    "upper_body_movement_px",
}


def format_timestamp_ms(ms: int) -> str:
    """Format millisecond timestamp to MM:SS (or HH:MM:SS if >= 1 hour)."""
    if ms < 0:
        ms = 0
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_duration_ms(ms: int) -> str:
    """Format total duration in milliseconds to human-readable string (e.g. '3m 00s' or '45s')."""
    if ms < 0:
        ms = 0
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_interval(start_ms: int, end_ms: int) -> str:
    """Format interval to 'MM:SS - MM:SS'."""
    return f"{format_timestamp_ms(start_ms)} - {format_timestamp_ms(end_ms)}"


class SpeakerIntervalResult(NamedTuple):
    """Result of interval extraction for a presenter."""

    intervals: list[dict[str, Any]]
    speaking_time_ms: int

    @property
    def formatted_intervals(self) -> list[str]:
        """Return human-readable strings for each speaking turn interval."""
        return [item["formatted"] for item in self.intervals]


def clean_diarization_turns(
    diarization_segments: Sequence[SpeakerSegment] | DiarizationResult | None,
    *,
    min_duration_ms: int = 100,
    merge_gap_ms: int = 300,
) -> list[SpeakerSegment]:
    """De-noise and consolidate diarization turns per speaker.

    - Eliminates micro-gaps <= merge_gap_ms between consecutive turns of the same speaker.
    - Merges overlapping turns from the same speaker.
    - Discards isolated micro-segments shorter than min_duration_ms as acoustic noise.
    - Returns a chronologically sorted list of clean SpeakerSegments.
    """
    if diarization_segments is None:
        return []

    if isinstance(diarization_segments, DiarizationResult):
        raw_segments = list(diarization_segments.segments)
    else:
        raw_segments = list(diarization_segments)

    if not raw_segments:
        return []

    # Group segments by speaker label
    by_speaker: dict[str, list[SpeakerSegment]] = {}
    for seg in raw_segments:
        if not seg.speaker_label or seg.speaker_label in ("SPEAKER_UNKNOWN", "UNKNOWN"):
            continue
        # Ensure start_ms <= end_ms
        s_ms = min(seg.start_ms, seg.end_ms)
        e_ms = max(seg.start_ms, seg.end_ms)
        by_speaker.setdefault(seg.speaker_label, []).append(
            SpeakerSegment(start_ms=s_ms, end_ms=e_ms, speaker_label=seg.speaker_label)
        )

    cleaned: list[SpeakerSegment] = []

    for spk, segs in by_speaker.items():
        # Sort chronologically
        sorted_segs = sorted(segs, key=lambda s: (s.start_ms, s.end_ms))

        # Merge adjacent turns with gap <= merge_gap_ms (or overlapping)
        merged: list[SpeakerSegment] = []
        for seg in sorted_segs:
            if not merged:
                merged.append(SpeakerSegment(start_ms=seg.start_ms, end_ms=seg.end_ms, speaker_label=spk))
                continue

            last = merged[-1]
            if seg.start_ms <= last.end_ms + merge_gap_ms:
                # Merge / extend
                last.end_ms = max(last.end_ms, seg.end_ms)
            else:
                merged.append(SpeakerSegment(start_ms=seg.start_ms, end_ms=seg.end_ms, speaker_label=spk))

        # Prune isolated micro-segments shorter than min_duration_ms
        for m_seg in merged:
            if (m_seg.end_ms - m_seg.start_ms) >= min_duration_ms:
                cleaned.append(m_seg)

    # Sort all cleaned segments across all speakers chronologically
    cleaned.sort(key=lambda s: (s.start_ms, s.end_ms, s.speaker_label))
    return cleaned


def filter_active_speaker_observations(
    observations: Sequence[ObsT],
    cleaned_segments: Sequence[SpeakerSegment] | DiarizationResult,
) -> list[ObsT]:
    """Filter observations, keeping only those temporally overlapping an active speaking turn.

    Discards observations attributed to a speaker if the speaker was not actively
    presenting during that observation's interval (e.g. passive audience presence,
    ambient noise, or silent intervals).
    """
    if isinstance(cleaned_segments, DiarizationResult):
        segments = cleaned_segments.segments
    else:
        segments = list(cleaned_segments)

    if not observations or not segments:
        return []

    # Map cleaned segments by speaker label
    segments_by_speaker: dict[str, list[SpeakerSegment]] = {}
    for seg in segments:
        segments_by_speaker.setdefault(seg.speaker_label, []).append(seg)

    filtered: list[ObsT] = []
    for obs in observations:
        spk = obs.speaker_label
        if not spk or spk not in segments_by_speaker:
            continue

        obs_start = min(obs.start_ms, obs.end_ms)
        obs_end = max(obs.start_ms, obs.end_ms)

        # Check for temporal overlap > 0 with any cleaned turn of this speaker
        has_overlap = False
        for seg in segments_by_speaker[spk]:
            overlap = min(obs_end, seg.end_ms) - max(obs_start, seg.start_ms)
            if overlap > 0:
                has_overlap = True
                break

        if has_overlap:
            filtered.append(obs)

    return filtered


def extract_speaker_intervals(
    cleaned_segments: Sequence[SpeakerSegment] | DiarizationResult,
    speaker_label: str,
) -> SpeakerIntervalResult:
    """Extract active speaking turns and total duration for a given speaker.

    Returns:
        SpeakerIntervalResult with:
        - intervals: list of dicts with start_ms, end_ms, and formatted ('MM:SS - MM:SS')
        - speaking_time_ms: total active presentation time in milliseconds
    """
    if isinstance(cleaned_segments, DiarizationResult):
        segments = cleaned_segments.segments
    else:
        segments = list(cleaned_segments)

    spk_turns = [
        seg for seg in segments
        if seg.speaker_label == speaker_label
    ]
    spk_turns.sort(key=lambda s: (s.start_ms, s.end_ms))

    intervals: list[dict[str, Any]] = []
    total_time_ms = 0

    for seg in spk_turns:
        dur = max(0, seg.end_ms - seg.start_ms)
        total_time_ms += dur
        intervals.append(
            {
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "formatted": format_interval(seg.start_ms, seg.end_ms),
            }
        )

    return SpeakerIntervalResult(intervals=intervals, speaking_time_ms=total_time_ms)


def aggregate_speaker_observations(
    visual_obs: Sequence[VisualObservation],
    audio_obs: Sequence[AudioObservation],
    diarization: DiarizationResult | Sequence[SpeakerSegment],
    *,
    window_size_ms: int = 10000,
) -> dict[str, Any]:
    """Aggregate acoustic and visual observations into 10-second windowed profiles per presenter.

    1. De-noises and merges diarization segments via clean_diarization_turns.
    2. Filters observations to only those with active speaker overlap.
    3. Groups observations by canonical speaker label.
    4. Buckets observations into 10-second windows without zero-filling silent gaps.
    5. Computes mathematically verified acoustic and visual statistics.

    Returns:
        Dictionary mapping speaker labels (e.g. 'SPEAKER_00') to speaker profiles:
        {
            "SPEAKER_00": {
                "speaking_time_ms": 45000,
                "intervals": [{"start_ms": 0, "end_ms": 45000, "formatted": "00:00 - 00:45"}],
                "windows": [
                    {
                        "start_ms": 0,
                        "end_ms": 10000,
                        "acoustic": { ... },
                        "visual": { ... }
                    }
                ]
            }
        }
    """
    # 1. Clean diarization
    cleaned_segs = clean_diarization_turns(diarization)

    # 2. Filter active observations
    filtered_visual = filter_active_speaker_observations(visual_obs, cleaned_segs)
    filtered_audio = filter_active_speaker_observations(audio_obs, cleaned_segs)

    # 3. Identify all participating active speakers
    speaker_set: set[str] = set()
    if isinstance(diarization, DiarizationResult):
        for s in diarization.speaker_labels:
            if s and s not in ("SPEAKER_UNKNOWN", "UNKNOWN"):
                speaker_set.add(s)
    for seg in cleaned_segs:
        speaker_set.add(seg.speaker_label)
    for obs in filtered_visual:
        if obs.speaker_label:
            speaker_set.add(obs.speaker_label)
    for obs in filtered_audio:
        if obs.speaker_label:
            speaker_set.add(obs.speaker_label)

    # Index filtered observations by speaker
    visual_by_speaker: dict[str, list[VisualObservation]] = {}
    for v in filtered_visual:
        if v.speaker_label:
            visual_by_speaker.setdefault(v.speaker_label, []).append(v)

    audio_by_speaker: dict[str, list[AudioObservation]] = {}
    for a in filtered_audio:
        if a.speaker_label:
            audio_by_speaker.setdefault(a.speaker_label, []).append(a)

    by_speaker: dict[str, Any] = {}

    for spk in sorted(speaker_set):
        spk_interval_res = extract_speaker_intervals(cleaned_segs, spk)
        spk_visual = visual_by_speaker.get(spk, [])
        spk_audio = audio_by_speaker.get(spk, [])

        # Group observations by window index
        window_indices: set[int] = set()
        audio_by_window: dict[int, list[AudioObservation]] = {}
        for a in spk_audio:
            mid = (a.start_ms + a.end_ms) / 2
            w_idx = int(mid // window_size_ms)
            window_indices.add(w_idx)
            audio_by_window.setdefault(w_idx, []).append(a)

        visual_by_window: dict[int, list[VisualObservation]] = {}
        for v in spk_visual:
            mid = (v.start_ms + v.end_ms) / 2
            w_idx = int(mid // window_size_ms)
            window_indices.add(w_idx)
            visual_by_window.setdefault(w_idx, []).append(v)

        windows: list[dict[str, Any]] = []

        for w_idx in sorted(window_indices):
            w_start = w_idx * window_size_ms
            w_end = (w_idx + 1) * window_size_ms

            w_audio = audio_by_window.get(w_idx, [])
            w_visual = visual_by_window.get(w_idx, [])

            # Aggregate acoustic metrics
            acoustic_dict: dict[str, Any] = {}
            if w_audio:
                audio_metric_values: dict[str, list[float]] = {}
                for a in w_audio:
                    audio_metric_values.setdefault(a.metric, []).append(float(a.value))

                for m in ("speaking_rate_wpm", "pitch_mean_hz", "pitch_std_hz"):
                    if m in audio_metric_values:
                        vals = audio_metric_values[m]
                        acoustic_dict[m] = round(sum(vals) / len(vals), 1)

                if "pause_duration_ms" in audio_metric_values:
                    p_dur = sum(audio_metric_values["pause_duration_ms"])
                    acoustic_dict["pause_duration_ms"] = int(p_dur) if p_dur.is_integer() else round(p_dur, 1)

                if "pause_count" in audio_metric_values:
                    acoustic_dict["pause_count"] = int(round(sum(audio_metric_values["pause_count"])))

                if "filler_count" in audio_metric_values:
                    acoustic_dict["filler_count"] = int(round(sum(audio_metric_values["filler_count"])))

            # Aggregate visual metrics
            visual_dict: dict[str, Any] = {}
            if w_visual:
                visual_metric_values: dict[str, list[float]] = {}
                for v in w_visual:
                    visual_metric_values.setdefault(v.metric, []).append(float(v.value))

                for m in (
                    "gaze_direction",
                    "head_pitch_degrees",
                    "head_yaw_degrees",
                    "posture_openness",
                    "shoulder_symmetry_ratio",
                    "upper_body_movement_px",
                ):
                    if m in visual_metric_values:
                        vals = visual_metric_values[m]
                        visual_dict[m] = round(sum(vals) / len(vals), 2)

            # Omit empty windows (must have at least one acoustic or visual measurement)
            if not acoustic_dict and not visual_dict:
                continue

            window_entry: dict[str, Any] = {
                "start_ms": w_start,
                "end_ms": w_end,
            }
            if acoustic_dict:
                window_entry["acoustic"] = acoustic_dict
            if visual_dict:
                window_entry["visual"] = visual_dict

            windows.append(window_entry)

        # Do not include speakers that have zero speaking turns and zero observations
        if spk_interval_res.speaking_time_ms == 0 and not windows:
            continue

        by_speaker[spk] = {
            "speaking_time_ms": spk_interval_res.speaking_time_ms,
            "intervals": spk_interval_res.intervals,
            "windows": windows,
        }

    return by_speaker


# Canonical alias for pipeline convenience
aggregate_by_speaker = aggregate_speaker_observations


def _resolve_speaker_name(speaker_label: str, speaker_mappings: dict[str, Any] | None) -> str:
    """Resolve display name for a speaker label using mapping dictionaries."""
    if not speaker_mappings or speaker_label not in speaker_mappings:
        return speaker_label
    val = speaker_mappings[speaker_label]
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("display_name") or val.get("name") or speaker_label
    if hasattr(val, "display_name"):
        return getattr(val, "display_name")
    return speaker_label


def format_speaker_summary_for_prompt(
    by_speaker: dict[str, Any],
    speaker_mappings: dict[str, Any] | None = None,
) -> str:
    """Format compact, bounded text summary of each presenter's delivery metrics for AI-07 prompts."""
    if not by_speaker:
        return "No presenter delivery metrics available."

    lines: list[str] = ["### Presenter Delivery Profiles:"]

    for spk, profile in by_speaker.items():
        name = _resolve_speaker_name(spk, speaker_mappings)
        header = f"#### Presenter: {name} ({spk})" if name != spk else f"#### Presenter: {spk}"
        lines.append(header)

        # Presenting turns
        intervals = profile.get("intervals", [])
        turn_strs = [
            item["formatted"] if isinstance(item, dict) else str(item)
            for item in intervals
        ]
        turns_display = ", ".join(turn_strs) if turn_strs else "None recorded"
        total_time_str = format_duration_ms(profile.get("speaking_time_ms", 0))
        lines.append(f"- Active Speaking Turns: {turns_display} (Total: {total_time_str})")

        windows = profile.get("windows", [])
        if not windows:
            lines.append("- Delivery Metrics: No active windowed observations recorded.\n")
            continue

        # Compute overall delivery highlights across active windows
        wpm_vals = [w["acoustic"]["speaking_rate_wpm"] for w in windows if "acoustic" in w and "speaking_rate_wpm" in w["acoustic"]]
        pitch_vals = [w["acoustic"]["pitch_mean_hz"] for w in windows if "acoustic" in w and "pitch_mean_hz" in w["acoustic"]]
        pause_durs = [w["acoustic"]["pause_duration_ms"] for w in windows if "acoustic" in w and "pause_duration_ms" in w["acoustic"]]
        fillers = [w["acoustic"]["filler_count"] for w in windows if "acoustic" in w and "filler_count" in w["acoustic"]]
        gaze_vals = [w["visual"]["gaze_direction"] for w in windows if "visual" in w and "gaze_direction" in w["visual"]]
        posture_vals = [w["visual"]["posture_openness"] for w in windows if "visual" in w and "posture_openness" in w["visual"]]

        overview_parts: list[str] = []
        if wpm_vals:
            overview_parts.append(f"Avg Pace: {sum(wpm_vals)/len(wpm_vals):.1f} WPM")
        if pitch_vals:
            overview_parts.append(f"Avg Pitch: {sum(pitch_vals)/len(pitch_vals):.1f} Hz")
        if pause_durs:
            overview_parts.append(f"Total Pauses: {sum(pause_durs)} ms")
        if fillers:
            overview_parts.append(f"Total Fillers: {sum(fillers)}")
        if gaze_vals:
            overview_parts.append(f"Avg Gaze Index: {sum(gaze_vals)/len(gaze_vals):.2f}")
        if posture_vals:
            overview_parts.append(f"Avg Posture Openness: {sum(posture_vals)/len(posture_vals):.2f}")

        if overview_parts:
            lines.append(f"- Delivery Highlights: {', '.join(overview_parts)}")

        # 10s Window timeline breakdown
        lines.append("- 10s Window Progression:")
        for w in windows:
            t_span = f"[{format_timestamp_ms(w['start_ms'])} - {format_timestamp_ms(w['end_ms'])}]"
            parts: list[str] = []
            if "acoustic" in w:
                ac = w["acoustic"]
                if "speaking_rate_wpm" in ac:
                    parts.append(f"WPM: {ac['speaking_rate_wpm']}")
                if "pitch_mean_hz" in ac:
                    parts.append(f"Pitch: {ac['pitch_mean_hz']}Hz")
                if "pause_duration_ms" in ac:
                    cnt = ac.get("pause_count", 0)
                    parts.append(f"Pauses: {ac['pause_duration_ms']}ms ({cnt})")
                if "filler_count" in ac:
                    parts.append(f"Fillers: {ac['filler_count']}")
            if "visual" in w:
                vs = w["visual"]
                if "gaze_direction" in vs:
                    parts.append(f"Gaze: {vs['gaze_direction']}")
                if "posture_openness" in vs:
                    parts.append(f"Posture: {vs['posture_openness']}")
                if "shoulder_symmetry_ratio" in vs:
                    parts.append(f"Symmetry: {vs['shoulder_symmetry_ratio']}")

            lines.append(f"  - {t_span}: {' | '.join(parts)}")

        lines.append("")

    return "\n".join(lines).strip()


__all__ = [
    "ACOUSTIC_MEAN_METRICS",
    "ACOUSTIC_SUM_METRICS",
    "SpeakerIntervalResult",
    "VISUAL_MEAN_METRICS",
    "aggregate_by_speaker",
    "aggregate_speaker_observations",
    "clean_diarization_turns",
    "extract_speaker_intervals",
    "filter_active_speaker_observations",
    "format_duration_ms",
    "format_interval",
    "format_speaker_summary_for_prompt",
    "format_timestamp_ms",
]
