"""init_models.py: Pre-downloads and verifies models and dependencies on startup."""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import app.compat  # noqa: F401
from app.__main__ import _load_env

logger = logging.getLogger("virtujudge.init")


def check_ffmpeg() -> bool:
    """Verify ffmpeg binary is available on PATH or Conda Library."""
    # Check Conda Library\bin on Windows
    _conda_bin = Path(sys.prefix) / "Library" / "bin"
    if _conda_bin.is_dir() and str(_conda_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_conda_bin) + os.pathsep + os.environ.get("PATH", "")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info("ffmpeg verified: %s", ffmpeg_path)
        return True
    logger.warning("ffmpeg NOT found on PATH!")
    return False


def ensure_mediapipe_models(model_dir: Path | None = None) -> list[Path]:
    """Pre-download MediaPipe Face and Pose task models into persistent/cache dir."""
    try:
        from app.providers.mediapipe_vision import (
            FACE_MODEL_URL,
            POSE_MODEL_URL,
            _ensure_model,
        )

        target_dir = model_dir or Path(
            os.getenv(
                "MEDIAPIPE_MODEL_DIR",
                str(Path.home() / ".cache" / "virtujudge" / "models"),
            )
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pre-downloading MediaPipe models to %s...", target_dir)
        face_path = _ensure_model(target_dir / "face_landmarker.task", FACE_MODEL_URL)
        pose_path = _ensure_model(target_dir / "pose_landmarker_lite.task", POSE_MODEL_URL)

        logger.info("MediaPipe models ready: %s, %s", face_path.name, pose_path.name)
        return [face_path, pose_path]
    except Exception as exc:
        logger.warning("MediaPipe pre-download warning: %s", exc)
        return []


def preload_pyannote_model() -> tuple[bool, str | None]:
    """Preload PyAnnote Diarization pipeline into Hugging Face cache if HF_TOKEN is present."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found; skipping PyAnnote pre-download.")
        return False, "HF_TOKEN not found in environment"

    try:
        from pyannote.audio import Pipeline

        raw_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
        model_name = (
            "pyannote/speaker-diarization-3.1"
            if raw_model in ("pyannote-community-1", "community-1", "pyannote/speaker-diarization-community-1")
            else raw_model
        )
        logger.info("Preloading PyAnnote diarization model '%s'...", model_name)
        try:
            _ = Pipeline.from_pretrained(model_name, token=hf_token)
        except TypeError:
            _ = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        logger.info("PyAnnote diarization model cached successfully.")
        return True, None
    except Exception as exc:
        logger.warning("Failed to preload PyAnnote model: %s", exc)
        return False, f"{type(exc).__name__}: {exc}"


def warm_up_all() -> dict[str, Any]:
    """Run all pre-downloads and checks on boot."""
    _load_env()
    logger.info("Starting VirtuJudge AI model initialization and warmup...")
    pyannote_ok, pyannote_err = preload_pyannote_model()
    results: dict[str, Any] = {
        "ffmpeg": check_ffmpeg(),
        "mediapipe": len(ensure_mediapipe_models()) == 2,
        "pyannote": pyannote_ok,
    }
    if pyannote_err:
        results["pyannote_error"] = pyannote_err
    logger.info("Warmup status: %s", results)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    warm_up_all()
