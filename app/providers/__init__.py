"""Providers package for VirtuJudge AI-ML."""

from typing import TYPE_CHECKING, Any

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
from app.providers.groq_speech import (
    GroqSpeechProvider,
    GroqSpeechProviderError,
    ProviderError,
)
from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    DocumentChunk,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    VisualObservation,
    WordTimestamp,
)

if TYPE_CHECKING:
    from app.providers.pyannote_diarization import PyannoteDiarizationProvider

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
    "GroqSpeechProvider",
    "GroqSpeechProviderError",
    "JudgeModelProvider",
    "ProviderError",
    "SpeakerSegment",
    "SpeechProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VisionProvider",
    "VisualObservation",
    "WordTimestamp",
]

# Conditionally export real providers when their underlying ML packages are available
try:
    import pyannote.audio  # type: ignore[import-not-found]  # noqa: F401
    import pyannote.audio  # noqa: F401

    from app.providers.pyannote_diarization import PyannoteDiarizationProvider  # noqa: F401

    __all__.append("PyannoteDiarizationProvider")
except ImportError:
    pass


def __getattr__(name: str) -> Any:
    """Allow lazy resolution of real providers when imported directly."""
    if name == "PyannoteDiarizationProvider":
        from app.providers.pyannote_diarization import PyannoteDiarizationProvider

        return PyannoteDiarizationProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
