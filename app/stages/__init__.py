"""Stages package for VirtuJudge AI-ML pipeline."""

from app.stages.audio import (
    AudioStageResult,
    run_audio_stage,
    validate_audio_observations,
)
from app.stages.media import (
    SUPPORTED_MEDIA_EXTENSIONS,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    MediaNormalizationError,
    MediaNormalizationResult,
    MediaSplitResult,
    create_normalization_limitation,
    get_wav_metadata,
    normalize_media,
    split_media,
)
from app.stages.speech import (
    SpeechStageResult,
    run_speech_stage,
    validate_word_timestamps,
)
from app.stages.vision import (
    VisionStageResult,
    check_coverage_limitations,
    run_vision_stage,
    validate_visual_observations,
)

__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "AudioStageResult",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "SpeechStageResult",
    "VisionStageResult",
    "check_coverage_limitations",
    "create_normalization_limitation",
    "get_wav_metadata",
    "normalize_media",
    "run_audio_stage",
    "run_speech_stage",
    "run_vision_stage",
    "split_media",
    "validate_audio_observations",
    "validate_visual_observations",
    "validate_word_timestamps",
]
