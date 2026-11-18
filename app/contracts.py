"""Contract models and data structures for VirtuJudge AI-ML worker."""

import re
from datetime import datetime

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from typing import Annotated, Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_HEX_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def is_valid_job_id(value: Any) -> TypeGuard[str]:
    """Return True if value matches a Backend UUID or compatibility ULID."""
    return isinstance(value, str) and bool(
        UUID_PATTERN.fullmatch(value) or ULID_PATTERN.fullmatch(value)
    )


def is_safe_identifier(value: Any) -> TypeGuard[str]:
    """Return True if value is a bounded, log- and path-safe identifier string."""
    return isinstance(value, str) and bool(SAFE_IDENTIFIER_PATTERN.fullmatch(value))


MAX_QUESTION_TEXT_LENGTH = 1000


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
    CHECKSUM_MISMATCH = "checksum_mismatch"
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
    schema_version: int = 1

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
    display_name: str | None = None


class PrimaryQuestion(BaseModel):
    """Primary grounded question generated from pitch session analysis."""

    candidate_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=MAX_QUESTION_TEXT_LENGTH)
    reason: str = Field(min_length=1)
    rubric_dimension: str = Field(min_length=1)
    evidence_ids: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


class FollowUpQuestion(BaseModel):
    """Follow-up question generated during QA rounds."""

    text: str = Field(min_length=1, max_length=MAX_QUESTION_TEXT_LENGTH)
    reason: str = Field(min_length=1)
    rubric_dimension: str = Field(min_length=1)
    evidence_ids: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


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
    audio: AudioAssetInput | None = None
    question_text: str = ""
    rubric_dimension: str = ""
    question_evidence_ids: list[str] = Field(default_factory=list)
    remaining_follow_ups: int
    practice_session_id: str | None = None

    @field_validator("audio", mode="before")
    @classmethod
    def coerce_audio_asset(cls, v: Any) -> Any:
        if v is None:
            return None
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
    rubric: RubricRef = Field(
        default_factory=lambda: RubricRef(rubric_id="startup_pitch", version=1)
    )
    speaker_mappings: list[SpeakerMapping] = Field(default_factory=list)


class EraseAIDataPayload(BaseModel):
    """Payload for erase_ai_data jobs."""

    erasure_request_id: str
    scope: Literal["asset", "practice_session", "project", "team"]
    scope_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    practice_session_ids: list[str] = Field(default_factory=list)
    answer_ids: list[str] = Field(default_factory=list)
    asset_version_ids: list[str] = Field(default_factory=list)
    retention_only: bool = False

    @field_validator("practice_session_ids", "answer_ids", "asset_version_ids")
    @classmethod
    def validate_scoped_ids(cls, values: list[str]) -> list[str]:
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", value) is None for value in values):
            raise ValueError("Invalid erasure resource identifier")
        return values


class SessionAnalysisCompleted(BaseModel):
    """Completed payload for session analysis."""

    analysis_artifact: ArtifactRef
    primary_questions: list[PrimaryQuestion]
    speaker_labels: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)

    @field_validator("primary_questions")
    @classmethod
    def validate_three_primary_questions(cls, v: list[PrimaryQuestion]) -> list[PrimaryQuestion]:
        if len(v) != 3:
            raise ValueError(f"primary_questions must contain exactly 3 items, got {len(v)}")
        for idx, q in enumerate(v):
            if not q.evidence_ids:
                raise ValueError(f"primary_questions[{idx}].evidence_ids should be non-empty")
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


ScoreStatus = Literal["scored", "not_evaluated"]
ScoreLabel = Literal["needs_work", "developing", "good", "strong"]
FindingKind = Literal[
    "strength",
    "improvement",
    "alignment",
    "contradiction",
    "omission",
    "observation",
]


class ScoreComponent(BaseModel):
    """A scored or un-evaluated dimension component in the rubric."""

    dimension: str
    status: ScoreStatus
    normalized_score: float | None = None
    display_score: int | None = None
    label: ScoreLabel | None = None
    configured_weight: float
    effective_weight: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    limitation_code: str | None = None


class Finding(BaseModel):
    """Concrete feedback observation or finding tied to evidence."""

    id: str
    kind: FindingKind
    title: str
    detail: str
    recommendation: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rubric_dimension: str | None = None
    speaker_labels: list[str] = Field(default_factory=list)


class SpeakingInterval(BaseModel):
    """Exact presentation turn interval with millisecond timing and formatted timestamp."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    formatted: str


class FeedbackSection(BaseModel):
    """Feedback container for team or member assessments."""

    summary: str
    strengths: list[Finding] = Field(default_factory=list)
    improvements: list[Finding] = Field(default_factory=list)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)


class MemberFeedback(BaseModel):
    """Individual presenter feedback with exact presentation turn timestamps."""

    user_id: str
    display_name: str
    speaker_labels: list[str] = Field(default_factory=list)
    speaking_intervals: list[SpeakingInterval] = Field(default_factory=list)
    speaking_time_ms: int = 0
    summary: str
    strengths: list[Finding] = Field(default_factory=list)
    improvements: list[Finding] = Field(default_factory=list)
    delivery_components: list[ScoreComponent] = Field(default_factory=list)


class Evaluation(BaseModel):
    """Structured evaluation artifact combining session analysis and Q&A rubric scoring."""

    id: str
    schema_version: int = 1
    analysis_attempt_id: str
    qa_round_id: str
    rubric: RubricRef
    overall_score: float | None = None
    components: list[ScoreComponent] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    team_feedback: FeedbackSection
    member_feedback: list[MemberFeedback] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)


class QAArtifactQuestion(BaseModel):
    """Question entry inside a serialized qa.json artifact."""

    model_config = ConfigDict(extra="allow")

    id: str
    kind: Literal["primary", "follow_up"] | str = "primary"
    position: int = 1
    text: str
    reason: str = ""
    rubric_dimension: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    answer_id: str | None = None
    parent_answer_id: str | None = None
    state: str = "answered"


class QAArtifactAnswer(BaseModel):
    """Answer entry inside a serialized qa.json artifact."""

    model_config = ConfigDict(extra="allow")

    id: str
    question_id: str
    answered_by: str = ""
    status: str = "submitted"
    audio_asset_version_id: str | None = None
    transcript_artifact_id: str | None = None
    assessment_artifact_id: str | None = None
    duration_ms: int | None = None
    submitted_at: str | None = None
    transcript: str | None = None


class QAArtifactAssessment(BaseModel):
    """Assessment entry inside a serialized qa.json artifact.

    Backend currently serializes question_id, assessment_artifact_id,
    assessment_text, and evidence_ids without score or status; the report
    stage recovers score/status from the verified assessment.json artifact
    when absent.
    """

    model_config = ConfigDict(extra="allow")

    question_id: str
    assessment_artifact_id: str | None = None
    assessment_text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ScoreStatus | None = None
    answer_id: str | None = None


class QAArtifact(BaseModel):
    """Serialized qa.json round artifact consumed by generate_report."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    artifact_id: str
    qa_round_id: str
    practice_session_id: str
    analysis_attempt: int | None = None
    questions: list[QAArtifactQuestion] = Field(default_factory=list)
    answers: list[QAArtifactAnswer] = Field(default_factory=list)
    assessments: list[QAArtifactAssessment] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerAssessmentDetail(BaseModel):
    """Inner assessment payload persisted in assessment.json."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ScoreStatus | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class AnswerAssessmentArtifact(BaseModel):
    """Derived answer_assessment artifact persisted at assessment.json."""

    model_config = ConfigDict(extra="allow")

    artifact_id: str
    answer_id: str
    practice_session_id: str | None = None
    question_id: str
    kind: Literal["answer_assessment"] = "answer_assessment"
    schema_version: int = 1
    producer_version: str = ""
    source_artifact_ids: list[str] = Field(default_factory=list)
    created_at: str = ""
    assessment: AnswerAssessmentDetail
    follow_up: FollowUpQuestion | None = None
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


class CancelledPayload(BaseModel):
    """Payload emitted when a job is cancelled."""

    stage: str | None = None
    message: str = "Analysis cancelled by user request."


class JobCancelledError(Exception):
    """Raised when a job is cancelled by user request during pipeline execution."""

    def __init__(
        self,
        job_id: str,
        stage: str,
        message: str = "Analysis cancelled by user request.",
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.stage = stage
        self.message = message


class ChecksumMismatchError(Exception):
    """Raised when resolved input bytes do not match the declared SHA-256 checksum."""

    def __init__(
        self,
        *,
        stage: str = "ingestion",
        artifact_id: str | None = None,
        message: str = "Input artifact failed checksum integrity verification.",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.artifact_id = artifact_id
        self.message = message


class QueueMessage(BaseModel):
    """Incoming queue message envelope."""

    schema_version: int = 1
    job_id: str
    job_type: JobType
    practice_session_id: str | None = None
    analysis_attempt: int = 1
    created_at: datetime
    trace_id: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ancestry(self) -> "QueueMessage":
        if self.job_type != JobType.ERASE_AI_DATA and not self.practice_session_id:
            raise ValueError("Analysis and report jobs require practice_session_id")
        return self


class WorkerUpdate(BaseModel):
    """Worker update envelope sent back to the orchestrator."""

    schema_version: int = 1
    sequence: int
    status: UpdateStatus
    occurred_at: datetime
    trace_id: str
    payload: Any = Field(default_factory=dict)


__all__ = [
    "MAX_QUESTION_TEXT_LENGTH",
    "SAFE_IDENTIFIER_PATTERN",
    "ULID_PATTERN",
    "UUID_PATTERN",
    "AnalyzeAnswerPayload",
    "AnalyzeSessionPayload",
    "AnswerAnalysisCompleted",
    "AnswerAssessmentArtifact",
    "AnswerAssessmentDetail",
    "ArtifactRef",
    "AssetInput",
    "AudioAssetInput",
    "CancelledPayload",
    "ChecksumMismatchError",
    "EraseAIDataPayload",
    "ErasureCompleted",
    "ErrorCode",
    "Evaluation",
    "FailedPayload",
    "FeedbackSection",
    "Finding",
    "FindingKind",
    "FollowUpQuestion",
    "GenerateReportPayload",
    "JobCancelledError",
    "JobType",
    "Limitation",
    "MemberFeedback",
    "PrimaryQuestion",
    "ProgressPayload",
    "QAArtifact",
    "QAArtifactAnswer",
    "QAArtifactAssessment",
    "QAArtifactQuestion",
    "QueueMessage",
    "ReportCompleted",
    "RubricRef",
    "SafeFailure",
    "ScoreComponent",
    "ScoreLabel",
    "ScoreStatus",
    "SessionAnalysisCompleted",
    "SpeakerMapping",
    "SpeakingInterval",
    "StartedPayload",
    "UpdateStatus",
    "WorkerUpdate",
    "is_safe_identifier",
    "is_valid_job_id",
]
