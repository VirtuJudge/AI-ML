"""Base provider protocols for VirtuJudge AI-ML.

Defines structural interfaces (typing.Protocol) for swappable AI/ML adapters.
"""

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.contracts import FollowUpQuestion, PrimaryQuestion
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    TranscriptionResult,
    VisualObservation,
)


class AnswerAssessment(BaseModel):
    """Structured assessment of a candidate answer against a question and rubric."""

    assessment_text: str
    evidence_ids: list[str] = Field(default_factory=list)
    follow_up: FollowUpQuestion | None = None


class SpeechProvider(Protocol):
    """Protocol for speech transcription providers."""

    async def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


class DiarizationProvider(Protocol):
    """Protocol for speaker diarization providers."""

    async def diarize(self, audio_path: Path) -> DiarizationResult: ...


class VisionProvider(Protocol):
    """Protocol for computer vision and body language providers."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]: ...


class AudioMetricsProvider(Protocol):
    """Protocol for acoustic analysis and prosody providers."""

    async def extract_metrics(self, audio_path: Path) -> list[AudioObservation]: ...


class DocumentProvider(Protocol):
    """Protocol for document ingestion, parsing, and vectorization."""

    async def extract_and_embed(
        self, doc_path: Path, *, asset_version_id: str = ""
    ) -> list[DocumentChunk]: ...


class JudgeModelProvider(Protocol):
    """Protocol for LLM reasoning and question generation providers."""

    async def generate_questions(
        self,
        transcript: str,
        rubric_id: str,
        document_chunks: list[DocumentChunk] | None = None,
        evidence_bundle: Any = None,
    ) -> list[PrimaryQuestion]: ...

    async def assess_answer(
        self,
        answer_transcript: str,
        question_text: str,
        rubric_dimension: str,
        *,
        remaining_follow_ups: int = 0,
    ) -> AnswerAssessment: ...


class EmbeddingProvider(Protocol):
    """Protocol for text embedding generation providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


__all__ = [
    "AnswerAssessment",
    "AudioMetricsProvider",
    "DiarizationProvider",
    "DocumentProvider",
    "EmbeddingProvider",
    "JudgeModelProvider",
    "SpeechProvider",
    "VisionProvider",
]
