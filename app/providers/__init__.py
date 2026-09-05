"""Providers package for VirtuJudge AI-ML."""

from app.providers.base import (
    AudioMetricsProvider,
    DiarizationProvider,
    DocumentProvider,
    EmbeddingProvider,
    JudgeModelProvider,
    SpeechProvider,
    VisionProvider,
)
from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.fake_documents import FakeDocumentProvider, FakeEmbeddingProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.fake_vision import FakeVisionProvider
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    VisualObservation,
)

__all__ = [
    "AudioMetricsProvider",
    "AudioObservation",
    "DiarizationProvider",
    "DiarizationResult",
    "DocumentChunk",
    "DocumentProvider",
    "EmbeddingProvider",
    "FakeAudioMetricsProvider",
    "FakeDiarizationProvider",
    "FakeDocumentProvider",
    "FakeEmbeddingProvider",
    "FakeJudgeModelProvider",
    "FakeSpeechProvider",
    "FakeVisionProvider",
    "JudgeModelProvider",
    "SpeakerSegment",
    "SpeechProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VisionProvider",
    "VisualObservation",
]
