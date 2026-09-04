"""Fake speech and diarization provider adapters for testing and local development."""

from pathlib import Path

from app.providers.types import (
    DiarizationResult,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
)


class FakeSpeechProvider:
    """Deterministic speech transcription provider."""

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=4500,
                    text=(
                        "Welcome to our pitch for VirtuJudge, the automated AI "
                        "evaluation platform."
                    ),
                    confidence=0.98,
                ),
                TranscriptionSegment(
                    start_ms=4600,
                    end_ms=9200,
                    text=(
                        "We solve founder preparation bottlenecks by providing "
                        "instant, rubric-grounded feedback."
                    ),
                    confidence=0.95,
                ),
                TranscriptionSegment(
                    start_ms=9300,
                    end_ms=14000,
                    text=(
                        "Our unit economics and defensibility are proven by "
                        "early customer trials."
                    ),
                    confidence=0.97,
                ),
            ],
            full_text=(
                "Welcome to our pitch for VirtuJudge, the automated AI evaluation platform. "
                "We solve founder preparation bottlenecks by providing "
                "instant, rubric-grounded feedback. "
                "Our unit economics and defensibility are proven by early customer trials."
            ),
            language="en",
        )


class FakeDiarizationProvider:
    """Deterministic speaker diarization provider."""

    async def diarize(self, audio_path: Path) -> DiarizationResult:
        return DiarizationResult(
            segments=[
                SpeakerSegment(start_ms=0, end_ms=9200, speaker_label="SPEAKER_00"),
                SpeakerSegment(start_ms=9300, end_ms=14000, speaker_label="SPEAKER_01"),
            ],
            speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        )


__all__ = ["FakeDiarizationProvider", "FakeSpeechProvider"]
