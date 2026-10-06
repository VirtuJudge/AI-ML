"""CLI entrypoint to run the VirtuJudge AI-ML worker daemon."""

import logging
import os
from pathlib import Path

from app.document_store import create_document_store
from app.pipeline import FakePipeline
from app.providers.fake_audio import FakeAudioMetricsProvider
from app.providers.fake_documents import FakeDocumentProvider
from app.providers.fake_judge import FakeJudgeModelProvider
from app.providers.fake_speech import FakeDiarizationProvider, FakeSpeechProvider
from app.providers.fake_vision import FakeVisionProvider
from app.providers.groq_judge import GroqJudgeModelProvider
from app.providers.groq_speech import GroqSpeechProvider
from app.storage import create_object_storage

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
    provider_mode = os.getenv("AI_PROVIDER_MODE", "fake").strip().lower()
    live_modes = {"live", "groq"}
    supported_modes = live_modes | {"fake", "mixed"}
    if provider_mode not in supported_modes:
        raise ValueError("AI_PROVIDER_MODE must be one of: fake, mixed, live, or groq.")

    if provider_mode in live_modes:
        logger.info("Initializing pipeline with LIVE Groq and ML providers...")
        speech_provider = GroqSpeechProvider()
        try:
            from app.providers.pyannote_diarization import PyannoteDiarizationProvider

            diarization_provider = PyannoteDiarizationProvider()
        except Exception as exc:
            logger.error(
                "PyAnnote diarization initialization failed in %s mode (%s).",
                provider_mode,
                type(exc).__name__,
            )
            raise RuntimeError("PyAnnote diarization failed in live mode.") from None

        try:
            from app.providers.mediapipe_vision import MediaPipeVisionProvider

            vision_provider = MediaPipeVisionProvider()
        except Exception as exc:
            logger.error(
                "MediaPipe vision initialization failed in %s mode (%s).",
                provider_mode,
                type(exc).__name__,
            )
            raise RuntimeError("MediaPipe vision failed in live mode.") from None

        try:
            from app.providers.librosa_audio import LibrosaAudioProvider

            audio_provider = LibrosaAudioProvider()
        except Exception as exc:
            logger.error(
                "Librosa audio initialization failed in %s mode (%s).",
                provider_mode,
                type(exc).__name__,
            )
            raise RuntimeError("Librosa audio failed in live mode.") from None

        try:
            from app.providers.pymupdf_documents import PyMuPDFDocumentProvider

            doc_provider = PyMuPDFDocumentProvider()
        except Exception as exc:
            logger.error(
                "PyMuPDF document provider initialization failed in %s mode (%s).",
                provider_mode,
                type(exc).__name__,
            )
            raise RuntimeError("PyMuPDF document provider failed in live mode.") from None

        judge_provider = GroqJudgeModelProvider()
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("DATABASE_URL is required for live document persistence.")
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
            GroqJudgeModelProvider() if provider_mode == "mixed" else FakeJudgeModelProvider()
        )
        object_storage = create_object_storage()

    # Store selection is independent of model mode. A configured persistent
    # store must never silently become an empty fake during erasure.
    try:
        doc_store = await create_document_store()
    except Exception as exc:
        logger.error("Document store initialization failed (%s).", type(exc).__name__)
        raise RuntimeError("Document store initialization failed.") from None
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
    from app.worker import _run_async, set_default_pipeline

    # Build providers before the worker begins consuming jobs so broken live
    # dependencies, credentials, or stores fail at startup rather than mid-job.
    set_default_pipeline(_run_async(build_pipeline()))
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
