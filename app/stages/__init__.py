"""Stages package for VirtuJudge AI-ML pipeline."""

from app.stages.answer_assessment import (
    AnswerAssessmentResult,
    run_answer_assessment_stage,
)
from app.stages.answers import (
    AnswerSpeechResult,
    run_answer_speech_stage,
)
from app.stages.audio import (
    AudioStageResult,
    run_audio_stage,
    validate_audio_observations,
)
from app.stages.common import validate_observation_timestamps
from app.stages.documents import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DocumentStageResult,
    run_document_stage,
)
from app.stages.evidence import (
    EvidenceBundle,
    EvidenceItem,
    build_evidence_bundle,
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
from app.stages.questions import (
    QuestionStageResult,
    run_question_stage,
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
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "SUPPORTED_MEDIA_EXTENSIONS",
    "AnswerAssessmentResult",
    "AnswerSpeechResult",
    "AudioStageResult",
    "DocumentStageResult",
    "EvidenceBundle",
    "EvidenceItem",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "QuestionStageResult",
    "SpeechStageResult",
    "VisionStageResult",
    "build_evidence_bundle",
    "check_coverage_limitations",
    "create_normalization_limitation",
    "get_wav_metadata",
    "normalize_media",
    "run_answer_assessment_stage",
    "run_answer_speech_stage",
    "run_audio_stage",
    "run_document_stage",
    "run_question_stage",
    "run_speech_stage",
    "run_vision_stage",
    "split_media",
    "validate_audio_observations",
    "validate_observation_timestamps",
    "validate_visual_observations",
    "validate_word_timestamps",
]
