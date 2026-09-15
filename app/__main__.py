"""CLI entrypoint to run the VirtuJudge AI-ML worker daemon."""

import asyncio
import logging
import os
from pathlib import Path

from app.backend_client import BackendClient
from app.document_store import FakeDocumentStore, create_document_store
from app.pipeline import FakePipeline
from app.storage import create_object_storage
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


async def build_pipeline() -> FakePipeline:
    """Compose PitchAnalysisPipeline using configured environment providers."""
    provider_mode = os.getenv("AI_PROVIDER_MODE", "fake").lower()

    if provider_mode in ("groq", "live"):
        logger.info("Initializing pipeline with LIVE Groq and ML providers...")
        speech_provider = GroqSpeechProvider()
        try:
            from app.providers.pyannote_diarization import PyannoteDiarizationProvider

            diarization_provider = PyannoteDiarizationProvider()
        except Exception as exc:
            logger.warning("PyAnnote diarization failed in %s mode (%s). Falling back to FakeDiarizationProvider.", provider_mode, exc)
            diarization_provider = FakeDiarizationProvider()

        try:
            from app.providers.mediapipe_vision import MediaPipeVisionProvider

            vision_provider = MediaPipeVisionProvider()
        except Exception as exc:
            if provider_mode == "live":
                logger.error("FATAL: Failed to initialize MediaPipe vision in live mode: %s", exc)
                raise RuntimeError(f"MediaPipe vision failed in live mode: {exc}") from exc
            logger.warning("MediaPipe vision fallback to fake: %s", exc)
            vision_provider = FakeVisionProvider()

        try:
            from app.providers.librosa_audio import LibrosaAudioProvider

            audio_provider = LibrosaAudioProvider()
        except Exception as exc:
            if provider_mode == "live":
                logger.error("FATAL: Failed to initialize Librosa audio in live mode: %s", exc)
                raise RuntimeError(f"Librosa audio failed in live mode: {exc}") from exc
            logger.warning("Librosa audio fallback to fake: %s", exc)
            audio_provider = FakeAudioMetricsProvider()

        try:
            from app.providers.pymupdf_documents import PyMuPDFDocumentProvider

            doc_provider = PyMuPDFDocumentProvider()
        except Exception as exc:
            if provider_mode == "live":
                logger.error("FATAL: Failed to initialize PyMuPDF in live mode: %s", exc)
                raise RuntimeError(f"PyMuPDF document provider failed in live mode: {exc}") from exc
            logger.warning("PyMuPDF fallback to fake: %s", exc)
            doc_provider = FakeDocumentProvider()

        judge_provider = GroqJudgeModelProvider()
        try:
            doc_store = await create_document_store()
        except Exception as exc:
            logger.warning("Failed to initialize live pgvector document store (%s). Falling back to FakeDocumentStore.", exc)
            doc_store = FakeDocumentStore()

        object_storage = create_object_storage()
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
        object_storage = create_object_storage()

    return FakePipeline(
        speech_provider=speech_provider,
        diarization_provider=diarization_provider,
        vision_provider=vision_provider,
        audio_provider=audio_provider,
        document_provider=doc_provider,
        judge_provider=judge_provider,
        document_store=doc_store,
        object_storage=object_storage,
    )


def main() -> None:
    """Initialize environment and start Celery worker on queue ai_jobs."""
    _load_env()
    logger.info("Starting VirtuJudge AI Celery Worker on queue 'ai_jobs'...")
    from app.celery_app import celery_app

    worker = celery_app.Worker(
        queues=["ai_jobs"],
        concurrency=int(os.getenv("CELERY_CONCURRENCY", "1")),
        pool=os.getenv("CELERY_POOL", "solo"),
        loglevel=os.getenv("CELERY_LOGLEVEL", "INFO"),
    )
    worker.start()


if __name__ == "__main__":
    main()
