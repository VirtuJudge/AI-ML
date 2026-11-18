"""Stages package for VirtuJudge AI-ML pipeline."""

from app.stages.aggregation import (
    SpeakerIntervalResult,
    aggregate_by_speaker,
    aggregate_speaker_observations,
    clean_diarization_turns,
    extract_speaker_intervals,
    filter_active_speaker_observations,
    format_duration_ms,
    format_interval,
    format_speaker_summary_for_prompt,
    format_timestamp_ms,
)
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
from app.stages.report_loader import (
    ReportEvidenceBundle,
    build_synthetic_analysis_data,
    build_synthetic_qa_data,
    load_report_evidence,
)
from app.stages.report_markdown import (
    format_speaker_metrics_highlight,
    generate_markdown_report,
)
from app.stages.reporting import (
    ReportStageResult,
    run_report_stage,
)
from app.stages.scoring import (
    DEFAULT_LABEL_BANDS,
    FIXED_WEIGHT_DIMENSIONS,
    RUBRIC_CONFIGS,
    RUBRIC_DEFINITIONS,
    STARTUP_PITCH_RUBRIC_DEFINITION,
    STARTUP_PITCH_RUBRIC_V1,
    WEIGHTING_POLICY_VERSION,
    LabelBands,
    QAScoreResult,
    RubricDefinition,
    calculate_individual_delivery_scores,
    calculate_overall_score,
    calculate_qa_score,
    evaluate_qa_scores,
    evaluate_rubric,
    get_rubric_definition,
    normalize_effective_weights,
    score_to_display,
    score_to_label,
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
    "DEFAULT_LABEL_BANDS",
    "FIXED_WEIGHT_DIMENSIONS",
    "RUBRIC_CONFIGS",
    "RUBRIC_DEFINITIONS",
    "STARTUP_PITCH_RUBRIC_DEFINITION",
    "STARTUP_PITCH_RUBRIC_V1",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "SUPPORTED_MEDIA_EXTENSIONS",
    "WEIGHTING_POLICY_VERSION",
    "AnswerAssessmentResult",
    "AnswerSpeechResult",
    "AudioStageResult",
    "DocumentStageResult",
    "EvidenceBundle",
    "EvidenceItem",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "LabelBands",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "QAScoreResult",
    "QuestionStageResult",
    "ReportEvidenceBundle",
    "ReportStageResult",
    "RubricDefinition",
    "SpeakerIntervalResult",
    "SpeechStageResult",
    "VisionStageResult",
    "aggregate_by_speaker",
    "aggregate_speaker_observations",
    "build_evidence_bundle",
    "build_synthetic_analysis_data",
    "build_synthetic_qa_data",
    "calculate_individual_delivery_scores",
    "calculate_overall_score",
    "calculate_qa_score",
    "check_coverage_limitations",
    "clean_diarization_turns",
    "create_normalization_limitation",
    "evaluate_qa_scores",
    "evaluate_rubric",
    "extract_speaker_intervals",
    "filter_active_speaker_observations",
    "format_duration_ms",
    "format_interval",
    "format_speaker_metrics_highlight",
    "format_speaker_summary_for_prompt",
    "format_timestamp_ms",
    "generate_markdown_report",
    "get_rubric_definition",
    "get_wav_metadata",
    "load_report_evidence",
    "normalize_effective_weights",
    "normalize_media",
    "run_answer_assessment_stage",
    "run_answer_speech_stage",
    "run_audio_stage",
    "run_document_stage",
    "run_question_stage",
    "run_report_stage",
    "run_speech_stage",
    "run_vision_stage",
    "score_to_display",
    "score_to_label",
    "split_media",
    "validate_audio_observations",
    "validate_observation_timestamps",
    "validate_visual_observations",
    "validate_word_timestamps",
]
