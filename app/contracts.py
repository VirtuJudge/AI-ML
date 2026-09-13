"""Contract models and data structures for VirtuJudge AI-ML worker."""

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SHA256_HEX_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class JobType(StrEnum):
    """Supported job types for the AI-ML worker."""

    ANALYZE_SESSION = "analyze_session"
    ANALYZE_ANSWER = "analyze_answer"
    GENERATE_REPORT = "generate_report"
    ERASE_AI_DATA = "erase_ai_data"


class UpdateStatus(StrEnum):
    """Status values for worker progress and terminal updates."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(StrEnum):
    """Standard safe error codes for pipeline failures."""

    INVALID_JOB_TYPE = "invalid_job_type"
    MALFORMED_PAYLOAD = "malformed_payload"
    INVALID_CHECKSUM = "invalid_checksum"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    INTERNAL_ERROR = "internal_error"


class ArtifactRef(BaseModel):
    """Reference to an artifact stored in object storage."""

    artifact_id: str
    object_key: str
    checksum: str
    schema_version: int | None = None

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        if not SHA256_HEX_RE.match(v):
            raise ValueError("checksum must match 'sha256:' followed by 64 hex characters")
        return v


class AssetInput(BaseModel):
    """Input asset reference provided for analysis jobs."""

    artifact_id: str
    object_key: str
    checksum: str
    media_type: str
    duration_ms: int | None = None

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        if not v.startswith("sha256:"):
            raise ValueError("checksum must start with 'sha256:'")
        return v


class AudioAssetInput(AssetInput):
    """Input audio asset reference requiring explicit duration in milliseconds."""

    duration_ms: int


class RubricRef(BaseModel):
    """Reference to the evaluation rubric configuration."""

    rubric_id: str
    version: int


class SpeakerMapping(BaseModel):
    """Mapping between diarized speaker label and user identity."""

    speaker_label: str
    user_id: str
    display_name: str


class PrimaryQuestion(BaseModel):
    """Primary grounded question generated from pitch session analysis."""

    candidate_id: str
    text: str
    reason: str
    rubric_dimension: str
    evidence_ids: list[str] = Field(default_factory=list)


class FollowUpQuestion(BaseModel):
    """Follow-up question generated during QA rounds."""

    text: str
    reason: str
    rubric_dimension: str
    evidence_ids: list[str] = Field(default_factory=list)


class Limitation(BaseModel):
    """System or analysis limitation record."""

    code: str
    scope: str
    message: str
    affected_dimensions: list[str] = Field(default_factory=list)


class AnalyzeSessionPayload(BaseModel):
    """Payload for analyze_session jobs."""

    presentation: AssetInput
    supporting_documents: list[AssetInput] = Field(default_factory=list)
    rubric: RubricRef
    requested_capabilities: list[str] = Field(default_factory=list)
    practice_session_id: str | None = None


class AnalyzeAnswerPayload(BaseModel):
    """Payload for analyze_answer jobs."""

    qa_round_id: str
    question_id: str
    answer_id: str
    answered_by: str
    audio: AudioAssetInput
    remaining_follow_ups: int

    @field_validator("audio", mode="before")
    @classmethod
    def coerce_audio_asset(cls, v: Any) -> Any:
        if isinstance(v, AssetInput) and not isinstance(v, AudioAssetInput):
            if v.duration_ms is None:
                raise ValueError("audio must have duration_ms specified")
            return AudioAssetInput(
                artifact_id=v.artifact_id,
                object_key=v.object_key,
                checksum=v.checksum,
                media_type=v.media_type,
                duration_ms=v.duration_ms,
            )
        return v


class GenerateReportPayload(BaseModel):
    """Payload for generate_report jobs."""

    report_id: str
    analysis_artifact: ArtifactRef
    qa_artifact: ArtifactRef
    speaker_mappings: list[SpeakerMapping] = Field(default_factory=list)


class EraseAIDataPayload(BaseModel):
    """Payload for erase_ai_data jobs."""

    erasure_request_id: str
    scope: Literal["asset", "practice_session", "project", "team"]
    scope_id: str


class SessionAnalysisCompleted(BaseModel):
    """Completed payload for session analysis."""

    analysis_artifact: ArtifactRef
    primary_questions: list[PrimaryQuestion]
    speaker_labels: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)

    @field_validator("primary_questions")
    @classmethod
    def validate_three_primary_questions(
        cls, v: list[PrimaryQuestion]
    ) -> list[PrimaryQuestion]:
        if len(v) != 3:
            raise ValueError(f"primary_questions must contain exactly 3 items, got {len(v)}")
        return v


class AnswerAnalysisCompleted(BaseModel):
    """Completed payload for answer analysis."""

    answer_id: str
    transcript_artifact_id: str
    assessment_artifact_id: str
    follow_up: FollowUpQuestion | None = None


class ReportCompleted(BaseModel):
    """Completed payload for report generation."""

    evaluation_artifact: ArtifactRef
    report_artifact: ArtifactRef
    member_feedback_user_ids: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)


class ErasureCompleted(BaseModel):
    """Completed payload for AI data erasure."""

    erasure_request_id: str
    deleted_records: int = Field(ge=0)
    deleted_objects: int = Field(ge=0)


class StartedPayload(BaseModel):
    """Payload emitted when a worker starts processing a job."""

    pipeline_version: str


class ProgressPayload(BaseModel):
    """Payload emitted for intermediate job progress."""

    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str


class SafeFailure(BaseModel):
    """Safe failure record matching data contracts."""

    stage: str
    code: ErrorCode | str
    retryable: bool
    message: str


class FailedPayload(BaseModel):
    """Payload emitted when job execution fails."""

    stage: str
    code: ErrorCode | str
    retryable: bool
    attempts: int = Field(ge=0)
    message: str


class QueueMessage(BaseModel):
    """Incoming queue message envelope."""

    schema_version: int = 1
    job_id: str
    job_type: JobType
    practice_session_id: str
    analysis_attempt: int = 1
    created_at: datetime
    trace_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerUpdate(BaseModel):
    """Worker update envelope sent back to the orchestrator."""

    schema_version: int = 1
    sequence: int
    status: UpdateStatus
    occurred_at: datetime
    trace_id: str
    payload: Any = Field(default_factory=dict)


__all__ = [
    "AnalyzeAnswerPayload",
    "AnalyzeSessionPayload",
    "AnswerAnalysisCompleted",
    "ArtifactRef",
    "AssetInput",
    "AudioAssetInput",
    "EraseAIDataPayload",
    "ErasureCompleted",
    "ErrorCode",
    "FailedPayload",
    "FollowUpQuestion",
    "GenerateReportPayload",
    "JobType",
    "Limitation",
    "PrimaryQuestion",
    "ProgressPayload",
    "QueueMessage",
    "ReportCompleted",
    "RubricRef",
    "SafeFailure",
    "SessionAnalysisCompleted",
    "SpeakerMapping",
    "StartedPayload",
    "UpdateStatus",
    "WorkerUpdate",
]
