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
    from app.providers.http_embedding import (
        HttpEmbeddingProvider,
        HttpEmbeddingProviderError,
    )
    from app.providers.librosa_audio import LibrosaAudioProvider
    from app.providers.mediapipe_vision import MediaPipeVisionProvider
    from app.providers.pyannote_diarization import PyannoteDiarizationProvider
    from app.providers.pymupdf_documents import (
        DocumentExtractionError,
        PyMuPDFDocumentProvider,
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
    import librosa  # noqa: F401

    from app.providers.librosa_audio import LibrosaAudioProvider  # noqa: F401

    __all__.append("LibrosaAudioProvider")
except ImportError:
    pass

try:
    import mediapipe  # noqa: F401

    from app.providers.mediapipe_vision import MediaPipeVisionProvider  # noqa: F401

    __all__.append("MediaPipeVisionProvider")
except ImportError:
    pass

try:
    import pyannote.audio  # noqa: F401

    from app.providers.pyannote_diarization import PyannoteDiarizationProvider  # noqa: F401

    __all__.append("PyannoteDiarizationProvider")
except ImportError:
    pass

try:
    import pymupdf  # noqa: F401

    from app.providers.pymupdf_documents import (  # noqa: F401
        DocumentExtractionError,
        PyMuPDFDocumentProvider,
    )

    __all__.extend(["DocumentExtractionError", "PyMuPDFDocumentProvider"])
except ImportError:
    pass

try:
    from app.providers.http_embedding import (  # noqa: F401
        HttpEmbeddingProvider,
        HttpEmbeddingProviderError,
    )

    __all__.extend(["HttpEmbeddingProvider", "HttpEmbeddingProviderError"])
except ImportError:
    pass


def __getattr__(name: str) -> Any:
    """Allow lazy resolution of real providers when imported directly."""
    if name == "LibrosaAudioProvider":
        from app.providers.librosa_audio import LibrosaAudioProvider

        return LibrosaAudioProvider
    if name == "MediaPipeVisionProvider":
        from app.providers.mediapipe_vision import MediaPipeVisionProvider

        return MediaPipeVisionProvider
    if name == "PyannoteDiarizationProvider":
        from app.providers.pyannote_diarization import PyannoteDiarizationProvider

        return PyannoteDiarizationProvider
    if name == "PyMuPDFDocumentProvider":
        from app.providers.pymupdf_documents import PyMuPDFDocumentProvider

        return PyMuPDFDocumentProvider
    if name == "DocumentExtractionError":
        from app.providers.pymupdf_documents import DocumentExtractionError

        return DocumentExtractionError
    if name == "HttpEmbeddingProvider":
        from app.providers.http_embedding import HttpEmbeddingProvider

        return HttpEmbeddingProvider
    if name == "HttpEmbeddingProviderError":
        from app.providers.http_embedding import HttpEmbeddingProviderError

        return HttpEmbeddingProviderError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
