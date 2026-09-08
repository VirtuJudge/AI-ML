"""Librosa acoustic and prosody analysis provider.

Extracts timed acoustic measurements (speaking rate, pitch variation, and pauses)
from audio files without making psychological, emotional, or personality claims.
"""

import asyncio
import logging
from pathlib import Path

import librosa
import numpy as np

from app.providers.types import AudioObservation

logger = logging.getLogger(__name__)

ALGORITHM_VERSION = f"librosa/{librosa.__version__}"


class LibrosaAudioProvider:
    """Extracts timed acoustic measurements from audio files using librosa."""

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

    def _process_audio_sync(self, audio_path: Path) -> list[AudioObservation]:
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

        total_duration_sec = len(y) / sr
        frame_ms = (self.hop_length / sr) * 1000.0
        min_pause_frames = max(1, round(250.0 / frame_ms))  # 250ms threshold for pause event

        # Build analysis windows
        windows: list[tuple[float, float]] = []
        if total_duration_sec <= self.window_sec:
            windows.append((0.0, total_duration_sec))
        else:
            t = 0.0
            while t < total_duration_sec:
                w_end = min(total_duration_sec, t + self.window_sec)
                if w_end - t >= 0.5:  # Require at least 500ms for analysis
                    windows.append((t, w_end))
                t += self.hop_sec

        observations: list[AudioObservation] = []

        for w_start, w_end in windows:
            start_ms = round(w_start * 1000)
            end_ms = round(w_end * 1000)
            dur_sec = w_end - w_start

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

            # 2. Fundamental frequency (pitch) mean and variation via pYIN
            pitch_mean = 0.0
            pitch_std = 0.0
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    y_win,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=sr,
                    hop_length=self.hop_length,
                )
                valid_voiced = f0[voiced_flag & ~np.isnan(f0)]
                if len(valid_voiced) > 0:
                    pitch_mean = round(float(np.mean(valid_voiced)), 2)
                    pitch_std = round(float(np.std(valid_voiced)), 2)
            except Exception:
                logger.debug("Pitch estimation failed for window [%.2f, %.2f]", w_start, w_end)

            # 3. Silence / pause detection via RMS energy thresholding
            pause_duration_ms = 0.0
            pause_count = 0
            try:
                rms = librosa.feature.rms(y=y_win, hop_length=self.hop_length)[0]
                max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
                if max_rms > 1e-5:
                    rms_db = librosa.amplitude_to_db(rms, ref=max_rms)
                    is_silent = rms_db < self.silence_threshold_db
                else:
                    is_silent = np.ones_like(rms, dtype=bool)

                silent_frames = int(np.sum(is_silent))
                pause_duration_ms = round(float(silent_frames * frame_ms), 1)

                # Count contiguous pause runs lasting >= 250ms
                consecutive = 0
                for silent in is_silent:
                    if silent:
                        consecutive += 1
                    else:
                        if consecutive >= min_pause_frames:
                            pause_count += 1
                        consecutive = 0
                if consecutive >= min_pause_frames:
                    pause_count += 1
            except Exception:
                logger.debug("Pause analysis failed for window [%.2f, %.2f]", w_start, w_end)

            # Add contract-compliant acoustic observations
            observations.extend(
                [
                    AudioObservation(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        metric="speaking_rate_wpm",
                        value=wpm_val,
                        unit="words_per_minute",
                        confidence=0.85,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        metric="pitch_mean_hz",
                        value=pitch_mean,
                        unit="hertz",
                        confidence=0.85,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        metric="pitch_std_hz",
                        value=pitch_std,
                        unit="hertz",
                        confidence=0.85,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        metric="pause_duration_ms",
                        value=pause_duration_ms,
                        unit="milliseconds",
                        confidence=0.85,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                    AudioObservation(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        metric="pause_count",
                        value=float(pause_count),
                        unit="count",
                        confidence=0.85,
                        algorithm_version=ALGORITHM_VERSION,
                    ),
                ]
            )

        return observations

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]:
        """Asynchronously analyze audio file and extract acoustic observations."""
        return await asyncio.to_thread(self._process_audio_sync, Path(audio_path))


__all__ = [
    "ALGORITHM_VERSION",
    "LibrosaAudioProvider",
]
