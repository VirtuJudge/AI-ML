"""GroqCloud Whisper API speech transcription adapter.

Implements SpeechProvider using GroqCloud's OpenAI-compatible speech-to-text API
for fast dev-time and production transcription with word-level timestamps.

Note:
    GroqCloud Whisper API enforces a 25MB maximum file size limit for audio uploads.
    Files exceeding 25MB should be chunked, compressed, or normalized before transcription.
"""

import asyncio
import math
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from app.providers.types import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


class ProviderError(Exception):
    """Base exception for external AI provider failures."""


class GroqSpeechProviderError(ProviderError):
    """Exception raised when GroqCloud Whisper API request fails."""


class GroqSpeechProvider:
    """Speech transcription adapter powered by GroqCloud's Whisper API.

    Satisfies the SpeechProvider protocol by sending audio to Groq's
    OpenAI-compatible speech-to-text endpoint via httpx.

    Note:
        GroqCloud enforces a 25MB maximum file size limit for audio uploads.
    """

    DEFAULT_MODEL = "whisper-large-v3-turbo"
    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        api_url: str = API_URL,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize GroqSpeechProvider.

        Args:
            api_key: GroqCloud API key. If omitted, read from GROQ_API_KEY env var.
            model: Model identifier (default: "whisper-large-v3-turbo").
            api_url: Groq API endpoint URL.
            timeout: Request timeout in seconds (default: 60.0).
            http_client: Optional existing httpx.AsyncClient instance.

        Raises:
            ValueError: If no API key is provided and GROQ_API_KEY is unset or empty.
        """
        resolved_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "Groq API key is required. Provide `api_key` to GroqSpeechProvider "
                "or set the GROQ_API_KEY environment variable."
            )

        self.api_key = resolved_key.strip()
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self._client = http_client

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using GroqCloud Whisper API.

        Args:
            audio_path: Path to the audio file to transcribe. Must be an existing
                file and must not exceed Groq's 25MB upload limit.

        Returns:
            TranscriptionResult with segments, word timestamps, and provider metadata.

        Raises:
            FileNotFoundError: If audio_path does not exist or is not a file.
            GroqSpeechProviderError: If the API returns a 4xx/5xx error or connection fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_bytes = await asyncio.to_thread(audio_path.read_bytes)
        filename = audio_path.name
        mime_type = mimetypes.guess_type(filename)[0] or "audio/wav"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        files = {
            "file": (filename, audio_bytes, mime_type),
        }

        try:
            if self._client is not None:
                response = await self._client.post(
                    self.api_url,
                    headers=headers,
                    data=data,
                    files=files,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        data=data,
                        files=files,
                    )
        except httpx.RequestError as err:
            raise GroqSpeechProviderError(
                f"Groq Whisper API request failed due to network error: {type(err).__name__}"
            ) from None

        if response.is_error or response.status_code >= 400:
            raise GroqSpeechProviderError(
                f"Groq Whisper API request failed with status {response.status_code}: "
                f"{response.reason_phrase}"
            )

        try:
            resp_data = response.json()
        except Exception:
            raise GroqSpeechProviderError(
                "Groq Whisper API returned a response that could not be parsed as JSON."
            ) from None

        full_text = str(resp_data.get("text", "")).strip()
        detected_language = str(resp_data.get("language", "en"))
        duration_val = resp_data.get("duration")
        duration_s: float | None = None
        if duration_val is not None:
            try:
                duration_s = float(duration_val)
            except (ValueError, TypeError):
                duration_s = None

        # Parse root-level words if present
        raw_words = resp_data.get("words") or []
        top_words: list[WordTimestamp] = []
        for w in raw_words:
            w_text = str(w.get("word", "")).strip()
            if not w_text:
                continue
            w_start = round(float(w.get("start", 0.0)) * 1000)
            w_end = round(float(w.get("end", 0.0)) * 1000)
            w_start = max(0, w_start)
            w_end = max(w_start, w_end)
            w_conf = float(w.get("confidence", w.get("probability", 1.0)))
            w_conf = max(0.0, min(1.0, w_conf))
            top_words.append(
                WordTimestamp(
                    word=w_text,
                    start_ms=w_start,
                    end_ms=w_end,
                    confidence=round(w_conf, 4),
                )
            )

        # Parse segments
        raw_segments = resp_data.get("segments") or []
        segments: list[TranscriptionSegment] = []

        for seg in raw_segments:
            s_start = round(float(seg.get("start", 0.0)) * 1000)
            s_end = round(float(seg.get("end", 0.0)) * 1000)
            s_start = max(0, s_start)
            s_end = max(s_start, s_end)
            s_text = str(seg.get("text", "")).strip()

            seg_words: list[WordTimestamp] = []
            if seg.get("words"):
                for w in seg["words"]:
                    w_text = str(w.get("word", "")).strip()
                    if not w_text:
                        continue
                    w_start = round(float(w.get("start", 0.0)) * 1000)
                    w_end = round(float(w.get("end", 0.0)) * 1000)
                    w_start = max(0, w_start)
                    w_end = max(w_start, w_end)
                    w_conf = float(w.get("confidence", w.get("probability", 1.0)))
                    w_conf = max(0.0, min(1.0, w_conf))
                    seg_words.append(
                        WordTimestamp(
                            word=w_text,
                            start_ms=w_start,
                            end_ms=w_end,
                            confidence=round(w_conf, 4),
                        )
                    )

            # Calculate segment confidence
            seg_confidence = 1.0
            if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                try:
                    seg_confidence = max(0.0, min(1.0, math.exp(float(seg["avg_logprob"]))))
                except (OverflowError, ValueError):
                    seg_confidence = 1.0
            elif "confidence" in seg and seg["confidence"] is not None:
                try:
                    seg_confidence = max(0.0, min(1.0, float(seg["confidence"])))
                except (ValueError, TypeError):
                    seg_confidence = 1.0
            elif seg_words:
                seg_confidence = sum(w.confidence for w in seg_words) / len(seg_words)

            segments.append(
                TranscriptionSegment(
                    start_ms=s_start,
                    end_ms=s_end,
                    text=s_text,
                    confidence=round(seg_confidence, 4),
                    words=seg_words,
                )
            )

        # If segments did not contain nested words but root words exist, assign words to segments
        if top_words and segments and not any(s.words for s in segments):
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

        # If no segments exist but text or words exist, synthesize a single segment
        if not segments and (full_text or top_words):
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
                    confidence=1.0,
                    words=top_words,
                )
            )

        metadata: dict[str, Any] = {
            "provider": "groq",
            "model": self.model,
            "language": detected_language,
            "duration": duration_s,
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
