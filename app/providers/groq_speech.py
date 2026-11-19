"""GroqCloud Whisper API speech transcription adapter.

Implements SpeechProvider using GroqCloud's OpenAI-compatible speech-to-text API
for fast dev-time and production transcription with word-level timestamps.

Note:
    GroqCloud Whisper API enforces a 25MB maximum file size limit for audio uploads.
    Files exceeding 25MB should be chunked, compressed, or normalized before transcription.
"""

import math
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from app.providers.groq_pool import (
    GroqKeyPool,
    GroqPoolError,
)
from app.providers.types import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


class ProviderError(Exception):
    """Base exception for external AI provider failures."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
        is_timeout: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.is_timeout = is_timeout


class GroqSpeechProviderError(ProviderError):
    """Exception raised when GroqCloud Whisper API request fails."""


class GroqSpeechProvider:
    """Speech transcription adapter powered by GroqCloud's Whisper API.

    Satisfies the SpeechProvider protocol by sending audio to Groq's
    OpenAI-compatible speech-to-text endpoint via httpx. Supports
    multi-key failover and rotation via GroqKeyPool.

    Note:
        GroqCloud enforces a 25MB maximum file size limit for audio uploads.
    """

    DEFAULT_MODEL = "whisper-large-v3-turbo"
    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB strict request limit
    adapter_version = "groq_speech/1.0.0"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        key_pool: GroqKeyPool | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = API_URL,
        timeout: float = 60.0,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize GroqSpeechProvider.

        Args:
            api_key: GroqCloud API key. If omitted and key_pool is None, read from GROQ_API_KEY.
            key_pool: Optional GroqKeyPool instance for multi-key rotation and failover.
            model: Model identifier (default: "whisper-large-v3-turbo").
            api_url: Groq API endpoint URL.
            timeout: Request timeout in seconds (default: 60.0).
            max_file_size_bytes: Maximum audio file size in bytes (default: 25MB).
            http_client: Optional existing httpx.AsyncClient instance.

        Raises:
            ValueError: If no API key is provided and GROQ_API_KEY is unset or empty.
        """
        if key_pool is not None:
            self.key_pool = key_pool
            self.api_key = key_pool.api_keys[0] if key_pool.api_keys else ""
        elif api_key is not None:
            resolved_key = api_key.strip()
            if not resolved_key:
                raise ValueError(
                    "Groq API key is required. Provide `api_key` to GroqSpeechProvider "
                    "or set the GROQ_API_KEY environment variable."
                )
            self.api_key = resolved_key
            self.key_pool = GroqKeyPool(
                api_keys=[resolved_key],
                timeout=timeout,
                http_client=http_client,
            )
        else:
            resolved_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not resolved_key:
                raise ValueError(
                    "Groq API key is required. Provide `api_key` to GroqSpeechProvider "
                    "or set the GROQ_API_KEY environment variable."
                )
            self.api_key = resolved_key
            self.key_pool = GroqKeyPool(
                timeout=timeout,
                http_client=http_client,
            )

        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.max_file_size_bytes = max_file_size_bytes
        self._client = http_client

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    def checkpoint_config(self) -> dict[str, Any]:
        """Return safe configuration metadata for stage checkpoint fingerprinting."""
        return {
            "max_file_size_bytes": self.max_file_size_bytes,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }

    @staticmethod
    def _parse_confidence(raw: Any) -> tuple[float | None, bool]:
        """Return (confidence, malformed): None when absent or unusable, never a guessed 1.0."""
        if raw is None:
            return None, False
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, True
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return None, True
        return round(value, 4), False

    @classmethod
    def _parse_words(
        cls, raw_words: list[dict[str, Any]], stats: dict[str, int] | None = None
    ) -> list[WordTimestamp]:
        """Parse raw word dictionaries into WordTimestamp models."""
        words: list[WordTimestamp] = []
        for w in raw_words:
            w_text = str(w.get("word", "")).strip()
            if not w_text:
                continue
            w_start = round(float(w.get("start", 0.0)) * 1000)
            w_end = round(float(w.get("end", 0.0)) * 1000)
            w_start = max(0, w_start)
            w_end = max(w_start, w_end)
            raw_conf = w.get("confidence", w.get("probability"))
            w_conf, malformed = cls._parse_confidence(raw_conf)
            if stats is not None:
                if malformed:
                    stats["malformed"] += 1
                elif w_conf is None:
                    stats["missing_words"] += 1
            words.append(
                WordTimestamp(
                    word=w_text,
                    start_ms=w_start,
                    end_ms=w_end,
                    confidence=w_conf,
                )
            )
        return words

    @staticmethod
    def _mean_known_confidence(words: list[WordTimestamp]) -> float | None:
        known = [w.confidence for w in words if w.confidence is not None]
        return round(sum(known) / len(known), 4) if known else None

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using GroqCloud Whisper API.

        Args:
            audio_path: Path to the audio file to transcribe. Must be an existing
                file and must not exceed Groq's 25MB upload limit.

        Returns:
            TranscriptionResult with segments, word timestamps, and provider metadata.

        Raises:
            FileNotFoundError: If audio_path does not exist or is not a file.
            GroqSpeechProviderError: If the file exceeds 25MB, upload fails, or API returns error.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = audio_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise GroqSpeechProviderError(
                f"Audio file '{audio_path.name}' ({file_size / (1024 * 1024):.2f}MB) "
                f"exceeds Groq's 25MB upload limit ({self.max_file_size_bytes} bytes).",
                retryable=False,
            )

        # Audio uploads are capped at 25 MB. Read the bounded local file directly
        # so this adapter does not depend on asyncio's process-wide default
        # executor, which can remain shut down across the local test loop lifecycle.
        audio_bytes = audio_path.read_bytes()
        filename = audio_path.name
        mime_type = mimetypes.guess_type(filename)[0] or "audio/wav"

        data: dict[str, Any] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        files = {
            "file": (filename, audio_bytes, mime_type),
        }

        try:
            resp_data = await self.key_pool.post_multipart(
                "/audio/transcriptions",
                data=data,
                files=files,
                timeout=self.timeout,
            )
        except GroqPoolError as err:
            raise GroqSpeechProviderError(
                str(err),
                status_code=err.status_code,
                retryable=err.retryable,
                is_timeout=err.is_timeout,
            ) from err

        if not isinstance(resp_data, dict):
            raise GroqSpeechProviderError(
                "Groq Whisper API returned a response that could not be parsed as JSON.",
                retryable=False,
            )

        full_text = str(resp_data.get("text", "")).strip()
        detected_language = str(resp_data.get("language", "en"))
        duration_val = resp_data.get("duration")
        duration_s: float | None = None
        if duration_val is not None:
            try:
                duration_s = float(duration_val)
            except (ValueError, TypeError):
                duration_s = None

        # Parse segments
        raw_segments = resp_data.get("segments") or []
        segments: list[TranscriptionSegment] = []
        stats = {"malformed": 0, "missing_words": 0}

        for seg in raw_segments:
            s_start = round(float(seg.get("start", 0.0)) * 1000)
            s_end = round(float(seg.get("end", 0.0)) * 1000)
            s_start = max(0, s_start)
            s_end = max(s_start, s_end)
            s_text = str(seg.get("text", "")).strip()

            seg_words = self._parse_words(seg.get("words") or [], stats)

            # Segment confidence: provider avg_logprob, then an explicit confidence, then the
            # mean of known word confidences. Unknown stays None instead of becoming 1.0.
            seg_confidence: float | None = None
            seg_malformed = False
            if seg.get("avg_logprob") is not None:
                try:
                    seg_confidence, seg_malformed = self._parse_confidence(
                        math.exp(float(seg["avg_logprob"]))
                    )
                except (OverflowError, TypeError, ValueError):
                    seg_confidence, seg_malformed = None, True
            elif seg.get("confidence") is not None:
                seg_confidence, seg_malformed = self._parse_confidence(seg["confidence"])
            if seg_confidence is None:
                seg_confidence = self._mean_known_confidence(seg_words)
            if seg_malformed:
                stats["malformed"] += 1

            segments.append(
                TranscriptionSegment(
                    start_ms=s_start,
                    end_ms=s_end,
                    text=s_text,
                    confidence=seg_confidence,
                    words=seg_words,
                )
            )

        # If segments did not contain nested words but root words exist, assign words to segments
        raw_words = resp_data.get("words") or []
        if raw_words and segments and not any(s.words for s in segments):
            top_words = self._parse_words(raw_words, stats)
            for word in top_words:
                best_idx = 0
                best_overlap = -1
                for idx, seg in enumerate(segments):
                    overlap = max(
                        0,
                        min(seg.end_ms, word.end_ms) - max(seg.start_ms, word.start_ms),
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = idx
                if best_overlap == 0:
                    word_mid = (word.start_ms + word.end_ms) / 2
                    best_idx = min(
                        range(len(segments)),
                        key=lambda i: abs(
                            ((segments[i].start_ms + segments[i].end_ms) / 2) - word_mid
                        ),
                    )
                segments[best_idx].words.append(word)
            for segment in segments:
                if segment.confidence is None:
                    segment.confidence = self._mean_known_confidence(segment.words)

        # If no segments exist but text or words exist, synthesize a single segment
        if not segments and (full_text or raw_words):
            top_words = self._parse_words(raw_words, stats)
            start_ms = top_words[0].start_ms if top_words else 0
            end_ms = (
                top_words[-1].end_ms
                if top_words
                else (round(duration_s * 1000) if duration_s else 0)
            )
            segments.append(
                TranscriptionSegment(
                    start_ms=start_ms,
                    end_ms=max(start_ms, end_ms),
                    text=full_text,
                    confidence=self._mean_known_confidence(top_words),
                    words=top_words,
                )
            )

        metadata: dict[str, Any] = {
            "provider": "groq",
            "model": self.model,
            "language": detected_language,
            "duration": duration_s,
            "confidence_missing_segments": sum(1 for seg in segments if seg.confidence is None),
            "confidence_missing_words": stats["missing_words"],
            "confidence_malformed_values": stats["malformed"],
        }

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=detected_language,
            metadata=metadata,
        )


__all__ = [
    "GroqSpeechProvider",
    "GroqSpeechProviderError",
    "ProviderError",
]
