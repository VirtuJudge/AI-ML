"""Librosa acoustic and prosody analysis provider.

Extracts timed acoustic measurements (speaking rate, pitch variation, and pauses)
from audio files without making psychological, emotional, or personality claims.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from app.providers.types import AudioObservation

logger = logging.getLogger(__name__)

ALGORITHM_VERSION = f"librosa/{librosa.__version__}+disjoint_v1"


def _build_analysis_windows(
    total_duration_sec: float,
    window_sec: float,
    hop_sec: float,
) -> list[tuple[float, float]]:
    """Build sliding analysis windows covering [0, total_duration_sec]."""
    if total_duration_sec <= window_sec:
        return [(0.0, total_duration_sec)]

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < total_duration_sec:
        w_end = min(total_duration_sec, t + window_sec)
        if w_end - t >= 0.5:  # Require at least 500ms for analysis
            windows.append((t, w_end))
        t += hop_sec
    return windows


def _build_disjoint_regions(
    windows: list[tuple[float, float]],
    total_duration_ms: int,
) -> list[tuple[int, int]]:
    """Partition covered windows into non-overlapping [start_ms, end_ms] counting regions."""
    regions: list[tuple[int, int]] = []
    for idx, (w_start, w_end) in enumerate(windows):
        r_end = min(w_end, windows[idx + 1][0]) if idx + 1 < len(windows) else w_end
        r_start_ms = max(0, round(w_start * 1000))
        r_end_ms = min(total_duration_ms, round(r_end * 1000))
        if regions and r_start_ms < regions[-1][1]:
            r_start_ms = regions[-1][1]
        if r_end_ms > r_start_ms:
            regions.append((r_start_ms, r_end_ms))
    return regions


def _assign_event_to_region(
    ev_start_ms: int,
    ev_end_ms: int,
    regions: list[tuple[int, int]],
) -> int | None:
    """Return index of the disjoint region with largest overlap (tie-breaking earlier)."""
    best_idx: int | None = None
    best_overlap = 0
    for idx, (r_start, r_end) in enumerate(regions):
        overlap = min(ev_end_ms, r_end) - max(ev_start_ms, r_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


class LibrosaAudioProvider:
    """Extracts timed acoustic measurements from audio files using librosa."""

    model_name = "librosa_prosody"

    @property
    def adapter_version(self) -> str:
        """Return adapter/algorithm version dynamically for checkpoint fingerprinting."""
        return ALGORITHM_VERSION

    def checkpoint_config(self) -> dict[str, Any]:
        """Return safe sampling and feature configuration for checkpointing."""
        return {
            "sample_rate": self.sample_rate,
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "silence_threshold_db": self.silence_threshold_db,
            "algorithm_version": ALGORITHM_VERSION,
        }

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        window_sec: float = 5.0,
        hop_sec: float = 2.5,
        hop_length: int = 1024,
        fmin: float = 65.0,
        fmax: float = 400.0,
        silence_threshold_db: float = -35.0,
    ) -> None:
        """Initialize LibrosaAudioProvider.

        Args:
            sample_rate: Target sample rate to load audio (default: 16000 Hz).
            window_sec: Analysis window duration in seconds (default: 5.0s).
            hop_sec: Step size between successive analysis windows (default: 2.5s).
            hop_length: FFT frame hop length for STFT and feature analysis (default: 1024).
            fmin: Minimum fundamental frequency for pitch detection (default: 65.0 Hz).
            fmax: Maximum fundamental frequency for pitch detection (default: 400.0 Hz).
            silence_threshold_db: Threshold in dB below max RMS to classify silence
                (default: -35dB).
        """
        self.sample_rate = max(8000, int(sample_rate))
        self.window_sec = max(0.5, float(window_sec))
        self.hop_sec = max(0.1, float(hop_sec))
        self.hop_length = max(128, int(hop_length))
        self.fmin = max(20.0, float(fmin))
        self.fmax = min(2000.0, float(fmax))
        self.silence_threshold_db = float(silence_threshold_db)

    def _process_audio_sync(
        self, audio_path: Path, source_artifact_id: str = ""
    ) -> list[AudioObservation]:
        """Synchronously load and analyze audio intervals."""
        if not audio_path.is_file():
            raise RuntimeError(f"Failed to load audio file {audio_path}: File does not exist")

        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {exc}") from exc

        if len(y) == 0:
            logger.warning("Loaded empty audio array from %s", audio_path)
            return []

        artifact_id = source_artifact_id or audio_path.name
        total_duration_sec = len(y) / sr
        total_duration_ms = round(total_duration_sec * 1000.0)
        if total_duration_ms <= 0:
            return []

        frame_ms = (self.hop_length / sr) * 1000.0
        min_pause_frames = max(1, round(250.0 / frame_ms))  # 250ms threshold for pause event

        # Build sliding analysis windows and disjoint counting regions so extensive
        # event counts/durations are never double-counted across overlapping hops.
        windows = _build_analysis_windows(total_duration_sec, self.window_sec, self.hop_sec)
        if not windows:
            return []
        regions = _build_disjoint_regions(windows, total_duration_ms)

        # Global pitch & filler detection across the full timeline to avoid
        # splitting or double-counting fillers at window boundaries.
        f0_global: np.ndarray | None = None
        voiced_global: np.ndarray | None = None
        filler_events: list[tuple[int, int]] = []
        try:
            f0_global, voiced_global, _ = librosa.pyin(
                y,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
            )
            min_filler_frames = max(2, round(200.0 / frame_ms))
            max_filler_frames = max(min_filler_frames, round(1000.0 / frame_ms))
            consec_voiced = 0
            run_start_idx = 0
            run_pitches: list[float] = []

            for i, is_v in enumerate(voiced_global):
                if is_v and not np.isnan(f0_global[i]):
                    if consec_voiced == 0:
                        run_start_idx = i
                    consec_voiced += 1
                    run_pitches.append(float(f0_global[i]))
                else:
                    if (
                        min_filler_frames <= consec_voiced <= max_filler_frames
                        and run_pitches
                        and float(np.std(run_pitches)) < 35.0
                    ):
                        ev_s = max(0, round(run_start_idx * frame_ms))
                        ev_e = min(total_duration_ms, round(i * frame_ms))
                        if ev_e > ev_s:
                            filler_events.append((ev_s, ev_e))
                    consec_voiced = 0
                    run_pitches = []

            if (
                min_filler_frames <= consec_voiced <= max_filler_frames
                and run_pitches
                and float(np.std(run_pitches)) < 35.0
            ):
                ev_s = max(0, round(run_start_idx * frame_ms))
                ev_e = min(total_duration_ms, round(len(voiced_global) * frame_ms))
                if ev_e > ev_s:
                    filler_events.append((ev_s, ev_e))
        except Exception:
            logger.debug("Global pitch and filler estimation failed for %s", audio_path)
            f0_global = None
            voiced_global = None

        # Global silence / pause detection via RMS energy thresholding
        pause_events: list[tuple[int, int]] = []
        try:
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
            if max_rms > 1e-5:
                rms_db = librosa.amplitude_to_db(rms, ref=max_rms)
                is_silent = rms_db < self.silence_threshold_db
            else:
                is_silent = np.ones_like(rms, dtype=bool)

            consecutive = 0
            run_start_idx = 0
            for i, silent in enumerate(is_silent):
                if silent:
                    if consecutive == 0:
                        run_start_idx = i
                    consecutive += 1
                else:
                    if consecutive >= min_pause_frames:
                        ev_s = max(0, round(run_start_idx * frame_ms))
                        ev_e = min(total_duration_ms, round(i * frame_ms))
                        if ev_e > ev_s:
                            pause_events.append((ev_s, ev_e))
                    consecutive = 0

            if consecutive >= min_pause_frames:
                ev_s = max(0, round(run_start_idx * frame_ms))
                ev_e = min(total_duration_ms, round(len(is_silent) * frame_ms))
                if ev_e > ev_s:
                    pause_events.append((ev_s, ev_e))
        except Exception:
            logger.debug("Global pause analysis failed for %s", audio_path)

        observations: list[AudioObservation] = []

        # Emit intensive window metrics (speaking rate, pitch mean/std) per sliding window
        for w_start, w_end in windows:
            start_ms = max(0, round(w_start * 1000))
            end_ms = min(total_duration_ms, round(w_end * 1000))
            dur_sec = w_end - w_start
            if end_ms <= start_ms:
                continue

            start_idx = int(w_start * sr)
            end_idx = min(len(y), int(w_end * sr))
            y_win = y[start_idx:end_idx]

            if len(y_win) < self.hop_length:
                continue

            # 1. Speaking rate via speech onsets (approx. syllable nuclei)
            try:
                onsets = librosa.onset.onset_detect(y=y_win, sr=sr, hop_length=self.hop_length)
                # Standard conversion: ~1.5 syllables per word
                wpm = (len(onsets) / 1.5) * (60.0 / dur_sec) if dur_sec > 0 else 0.0
                wpm_val = round(min(300.0, max(0.0, float(wpm))), 1)
            except Exception:
                logger.debug("Onset detection failed for window [%.2f, %.2f]", w_start, w_end)
                wpm_val = 0.0

            # 2. Fundamental frequency (pitch) mean and variation from global pYIN
            pitch_mean: float | None = None
            pitch_std: float | None = None
            if f0_global is not None and voiced_global is not None and len(f0_global) > 0:
                s_frame = max(0, round((w_start * sr) / self.hop_length))
                e_frame = min(
                    len(f0_global), max(s_frame + 1, round((w_end * sr) / self.hop_length))
                )
                f0_win = f0_global[s_frame:e_frame]
                voiced_win = voiced_global[s_frame:e_frame]
                valid_voiced = f0_win[voiced_win & ~np.isnan(f0_win)]
                if len(valid_voiced) > 0:
                    pitch_mean = round(float(np.mean(valid_voiced)), 2)
                    pitch_std = round(float(np.std(valid_voiced)), 2)

            observations.append(
                AudioObservation(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    metric="speaking_rate_wpm",
                    value=wpm_val,
                    unit="words_per_minute",
                    confidence=0.85,
                    source_artifact_id=artifact_id,
                    algorithm_version=ALGORITHM_VERSION,
                )
            )

            # Only emit pitch observations when voiced speech was actually detected
            if pitch_mean is not None and pitch_std is not None:
                observations.extend(
                    [
                        AudioObservation(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            metric="pitch_mean_hz",
                            value=pitch_mean,
                            unit="hertz",
                            confidence=0.85,
                            source_artifact_id=artifact_id,
                            algorithm_version=ALGORITHM_VERSION,
                        ),
                        AudioObservation(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            metric="pitch_std_hz",
                            value=pitch_std,
                            unit="hertz",
                            confidence=0.85,
                            source_artifact_id=artifact_id,
                            algorithm_version=ALGORITHM_VERSION,
                        ),
                    ]
                )

        # Attribute pause and filler events onto disjoint counting regions once each
        region_pause_ms = [0.0] * len(regions)
        region_pause_counts = [0] * len(regions)
        region_filler_counts = [0] * len(regions)

        for p_start, p_end in pause_events:
            owner_idx = _assign_event_to_region(p_start, p_end, regions)
            if owner_idx is not None:
                region_pause_counts[owner_idx] += 1
            for r_idx, (r_start, r_end) in enumerate(regions):
                overlap = min(p_end, r_end) - max(p_start, r_start)
                if overlap > 0:
                    region_pause_ms[r_idx] += float(overlap)

        for f_start, f_end in filler_events:
            owner_idx = _assign_event_to_region(f_start, f_end, regions)
            if owner_idx is not None:
                region_filler_counts[owner_idx] += 1

        for r_idx, (r_start_ms, r_end_ms) in enumerate(regions):
            bounded_pause_ms = round(
                min(float(r_end_ms - r_start_ms), region_pause_ms[r_idx]),
                1,
            )
            observations.extend(
                [
                    AudioObservation(
                        start_ms=r_start_ms,
                        end_ms=r_end_ms,
                        metric="pause_duration_ms",
                        value=bounded_pause_ms,
                        unit="milliseconds",
                        confidence=0.85,
                        source_artifact_id=artifact_id,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=r_start_ms,
                        end_ms=r_end_ms,
                        metric="pause_count",
                        value=float(region_pause_counts[r_idx]),
                        unit="count",
                        confidence=0.85,
                        source_artifact_id=artifact_id,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=r_start_ms,
                        end_ms=r_end_ms,
                        metric="filler_count",
                        value=float(region_filler_counts[r_idx]),
                        unit="count",
                        confidence=0.80,
                        source_artifact_id=artifact_id,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                ]
            )

        return observations

    async def extract_metrics(
        self, audio_path: Path, *, source_artifact_id: str = ""
    ) -> list[AudioObservation]:
        """Asynchronously analyze audio file and extract acoustic observations."""
        return await asyncio.to_thread(
            self._process_audio_sync, Path(audio_path), source_artifact_id
        )


__all__ = [
    "ALGORITHM_VERSION",
    "LibrosaAudioProvider",
]
