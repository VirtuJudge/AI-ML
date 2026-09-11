"""Internal provider domain types for VirtuJudge AI-ML.

These types represent raw outputs from AI/ML models (speech recognition,
diarization, computer vision, acoustic feature extraction, and embeddings).
They are strictly internal to the AI-ML worker and must NEVER leak directly
into boundary contract payloads returned to the backend orchestrator.
"""

from typing import Any

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """A single transcribed word with millisecond timing bounds."""

    word: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio with timing bounds."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete transcription output from a speech provider."""

    segments: list[TranscriptionSegment] = Field(default_factory=list)
    full_text: str
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeakerSegment(BaseModel):
    """A diarized speaker turn with millisecond timestamps."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_label: str


class DiarizationResult(BaseModel):
    """Diarization analysis isolating distinct speaker labels."""

    segments: list[SpeakerSegment] = Field(default_factory=list)
    speaker_labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualObservation(BaseModel):
    """A timed visual measurement from a video interval."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    metric: str
    value: float
    unit: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_artifact_id: str = ""
    speaker_label: str | None = None
    algorithm_version: str = ""


class AudioObservation(BaseModel):
    """A timed acoustic measurement from an audio interval."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    metric: str
    value: float
    unit: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_artifact_id: str = ""
    speaker_label: str | None = None
    algorithm_version: str = ""


class DocumentChunk(BaseModel):
    """A parsed and vectorized document chunk."""

    chunk_id: str
    asset_version_id: str = ""
    page_or_slide: int = Field(ge=1)
    text: str
    start_offset: int | None = None
    end_offset: int | None = None
    extraction_method: str = ""
    chunking_version: str = ""
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    embedding: list[float] | None = None


__all__ = [
    "AudioObservation",
    "DiarizationResult",
    "DocumentChunk",
    "SpeakerSegment",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VisualObservation",
    "WordTimestamp",
]
