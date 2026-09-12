"""CLI entrypoint to run the VirtuJudge AI-ML worker daemon."""

import asyncio
import logging
import os
from pathlib import Path

from app.backend_client import BackendClient
from app.document_store import FakeDocumentStore
from app.pipeline import FakePipeline
from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.fake_documents import FakeDocumentProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.fake_vision import FakeVisionProvider
from app.providers.groq_judge import GroqJudgeModelProvider
from app.providers.groq_speech import GroqSpeechProvider
from app.queue_consumer import RedisQueueConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.worker")


def _load_env() -> None:
    """Load variables from .env if present."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def build_pipeline() -> FakePipeline:
    """Compose PitchAnalysisPipeline using configured environment providers."""
    provider_mode = os.getenv("AI_PROVIDER_MODE", "fake").lower()

    if provider_mode in ("groq", "live"):
        logger.info("Initializing pipeline with LIVE Groq and ML providers...")
        speech_provider = GroqSpeechProvider()
        try:
            from app.providers.pyannote_diarization import PyannoteDiarizationProvider

            diarization_provider = PyannoteDiarizationProvider()
        except Exception:
            diarization_provider = FakeDiarizationProvider()

        try:
            from app.providers.mediapipe_vision import MediaPipeVisionProvider

            vision_provider = MediaPipeVisionProvider()
        except Exception:
            vision_provider = FakeVisionProvider()

        try:
            from app.providers.librosa_audio import LibrosaAudioProvider

            audio_provider = LibrosaAudioProvider()
        except Exception:
            audio_provider = FakeAudioMetricsProvider()

        try:
            from app.providers.pymupdf_documents import PyMuPDFDocumentProvider

            doc_provider = PyMuPDFDocumentProvider()
        except Exception:
            doc_provider = FakeDocumentProvider()

        judge_provider = GroqJudgeModelProvider()
        doc_store = FakeDocumentStore()
    else:
        logger.info(
            "Initializing pipeline with stage providers (AI_PROVIDER_MODE=%s)...",
            provider_mode,
        )
        speech_provider = FakeSpeechProvider()
        diarization_provider = FakeDiarizationProvider()
        vision_provider = FakeVisionProvider()
        audio_provider = FakeAudioMetricsProvider()
        doc_provider = FakeDocumentProvider()
        judge_provider = (
            GroqJudgeModelProvider()
            if os.getenv("GROQ_API_KEY")
            else FakeJudgeModelProvider()
        )
        doc_store = FakeDocumentStore()

    return FakePipeline(
        speech_provider=speech_provider,
        diarization_provider=diarization_provider,
        vision_provider=vision_provider,
        audio_provider=audio_provider,
        document_provider=doc_provider,
        judge_provider=judge_provider,
        document_store=doc_store,
    )


async def main() -> None:
    """Initialize pipeline, backend client, and start queue consumer."""
    _load_env()
    pipeline = build_pipeline()
    backend_client = BackendClient()
    consumer = RedisQueueConsumer(
        pipeline=pipeline,
        backend_client=backend_client,
    )
    consumer.attach_signal_handlers()

    logger.info("Starting Redis Queue Consumer on queues %s...", consumer.queue_names)
    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())
