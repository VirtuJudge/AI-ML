"""FFmpeg-based audio normalization stage for VirtuJudge AI-ML.

Extracts and normalizes audio from arbitrary video/audio containers (.mp4, .webm,
.mov, .mkv, .avi, etc.) with arbitrary codecs and sample rates (e.g., 44.1kHz or 48kHz stereo)
into a canonical 16kHz, mono, 16-bit PCM WAV:
    ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav

Both Whisper (or Groq Whisper) and pyannote expect 16kHz mono audio. Standardizing
once upfront saves downstream stages from repeated audio conversions.
"""

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import re
import shutil
import tempfile
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field

from app.contracts import AssetInput, ErrorCode, Limitation
from app.stages.common import StageValidationError

logger = logging.getLogger(__name__)

MEDIA_NORMALIZATION_VERSION = "ffmpeg_pcm_s16le_16k_mono_v1"
DURATION_MISMATCH_TOLERANCE_MS = 250
JOB_TEMP_ROOT_ENV = "AI_MEDIA_TEMP_DIR"
DEFAULT_JOB_TEMP_DIRNAME = "virtujudge_media_jobs"
MEDIA_TIMEOUT_ENV = "AI_MEDIA_TIMEOUT_SECONDS"
DEFAULT_MEDIA_TIMEOUT_SECONDS = 300.0
MEDIA_TERMINATE_GRACE_ENV = "AI_MEDIA_TERMINATE_GRACE_SECONDS"
DEFAULT_MEDIA_TERMINATE_GRACE_SECONDS = 5.0

SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
        ".m4a",
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".aac",
    }
)


class MediaNormalizationResult(BaseModel):
    """Result of media audio extraction and normalization."""

    output_path: Path
    duration_ms: int = Field(ge=0)
    sample_rate: int = Field(default=16000, ge=1)
    channels: int = Field(default=1, ge=1)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0


class MediaSplitResult(BaseModel):
    """Result of splitting a presentation video into audio and vision streams."""

    audio_path: Path
    video_path: Path | None = None
    duration_ms: int = Field(ge=0)
    sample_rate: int = Field(default=16000, ge=1)
    channels: int = Field(default=1, ge=1)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0


class MediaNormalizationError(Exception):
    """Base exception for media normalization failures."""


class FFmpegNotFoundError(MediaNormalizationError):
    """Raised when ffmpeg executable is not available on PATH."""


class FFmpegExecutionError(MediaNormalizationError):
    """Raised when ffmpeg exits with a non-zero status code."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class FFmpegTimeoutError(MediaNormalizationError, TimeoutError):
    """Raised when ffmpeg exceeds the configured media processing deadline."""


class WavMetadata(NamedTuple):
    """Explicit metadata extracted from a WAV file header."""

    duration_ms: int
    sample_rate: int
    channels: int


def get_wav_metadata(wav_path: Path) -> WavMetadata:
    """Read duration_ms, sample_rate, and channels from a WAV file header."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_ms = round((n_frames / float(sample_rate)) * 1000) if sample_rate > 0 else 0
            return WavMetadata(
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=channels,
            )
    except Exception as err:
        raise MediaNormalizationError("Failed to read normalized WAV file header.") from err


def get_media_temp_root(*, base_dir: Path | str | None = None) -> Path:
    """Return the root directory for worker-owned per-job temporary media."""
    if base_dir is not None:
        return Path(base_dir)
    configured = os.getenv(JOB_TEMP_ROOT_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / DEFAULT_JOB_TEMP_DIRNAME


def _scope_tag(prefix: str, value: str | None) -> str | None:
    if not value:
        return None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _remove_empty_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _unlink_owned_output(path: Path, source_path: Path, *, should_unlink: bool) -> None:
    if not should_unlink:
        return
    if path.resolve() == source_path.resolve():
        return
    path.unlink(missing_ok=True)


@contextlib.contextmanager
def owned_job_temp_dir(
    *,
    job_kind: str = "job",
    job_id: str | None = None,
    session_id: str | None = None,
    asset_version_id: str | None = None,
    answer_id: str | None = None,
    base_dir: Path | str | None = None,
) -> Iterator[Path]:
    """Create an owned per-job temporary directory and remove it on exit.

    Cleans up the directory after normal completion, failures, and cancellation.
    """
    root = get_media_temp_root(base_dir=base_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_kind = re.sub(r"[^A-Za-z0-9_-]", "", job_kind) or "job"
    tags = [
        tag
        for tag in (
            _scope_tag("s", session_id),
            _scope_tag("v", asset_version_id),
            _scope_tag("a", answer_id),
            _scope_tag("j", job_id),
        )
        if tag is not None
    ]
    prefix = "_".join(["vj", safe_kind, *tags]) + "_"
    job_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
    try:
        yield job_dir
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        _remove_empty_dir(root)


def cleanup_job_temp_dirs_for_scope(
    *,
    session_ids: list[str] | None = None,
    asset_version_ids: list[str] | None = None,
    answer_ids: list[str] | None = None,
    base_dir: Path | str | None = None,
) -> int:
    """Remove any orphaned per-job temporary directories matching erased resource IDs."""
    root = get_media_temp_root(base_dir=base_dir)
    if not root.is_dir():
        return 0

    target_tags: set[str] = set()
    for sid in session_ids or ():
        tag = _scope_tag("s", sid)
        if tag:
            target_tags.add(tag)
    for vid in asset_version_ids or ():
        tag = _scope_tag("v", vid)
        if tag:
            target_tags.add(tag)
    for aid in answer_ids or ():
        tag = _scope_tag("a", aid)
        if tag:
            target_tags.add(tag)

    if not target_tags:
        return 0

    removed = 0
    for entry in list(root.iterdir()):
        if not entry.is_dir():
            continue
        if any(f"_{tag}_" in entry.name for tag in target_tags):
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1

    _remove_empty_dir(root)
    return removed


def cleanup_stale_worker_temp_dirs(
    *,
    base_dir: Path | str | None = None,
    scratch_dirs: tuple[Path, ...] | None = None,
) -> int:
    """Remove leftover owned per-job and worker scratch directories on worker startup."""
    removed = 0
    root = get_media_temp_root(base_dir=base_dir)
    if root.is_dir():
        # Only remove directories this worker creates (owned_job_temp_dir uses the "vj_"
        # prefix), so a shared AI_MEDIA_TEMP_DIR never loses files owned by anything else.
        for entry in list(root.iterdir()):
            if entry.is_dir() and entry.name.startswith("vj_"):
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        _remove_empty_dir(root)

    target_scratch = (
        scratch_dirs
        if scratch_dirs is not None
        else (Path(".storage/temp_media"), Path(".storage/temp_reports"))
    )
    for scratch_root in target_scratch:
        if scratch_root.is_dir():
            for entry in list(scratch_root.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
    return removed


def create_normalization_limitation(
    message: str = "Audio normalization unavailable or failed; speech analysis may be impacted.",
) -> Limitation:
    """Create a log-safe limitation for media normalization failures."""
    return Limitation(
        code="media_normalization_unavailable",
        scope="media",
        message=message,
        affected_dimensions=["delivery_pace", "acoustic_features"],
    )


def create_missing_duration_limitation(
    message: str = (
        "Media duration could not be determined from the media file or job metadata; "
        "duration-bounded validation and pacing metrics were omitted."
    ),
) -> Limitation:
    """Create a log-safe limitation when media duration is unavailable."""
    return Limitation(
        code="media_duration_unavailable",
        scope="media",
        message=message,
        affected_dimensions=["delivery_pace", "timing_and_speech_mechanics"],
    )


def create_duration_mismatch_limitation(
    message: str = (
        "Declared media duration did not match the probed media duration; "
        "analysis used the probed duration."
    ),
) -> Limitation:
    """Create a log-safe limitation when declared duration disagrees with probed duration."""
    return Limitation(
        code="media_duration_mismatch",
        scope="media",
        message=message,
        affected_dimensions=[],
    )


def resolve_media_duration(
    *,
    probed_duration_ms: int | None,
    declared_duration_ms: int | None,
    media_role: str = "Media",
    stage: str = "ingestion",
    mismatch_tolerance_ms: int = DURATION_MISMATCH_TOLERANCE_MS,
) -> tuple[int | None, list[Limitation]]:
    """Resolve authoritative media duration from probed media and declared job metadata.

    Policy:
        - When probed media duration is available:
          * Reject non-positive (<= 0) probed durations with StageValidationError;
            never fall back to declared duration or an invented default.
          * Use the positive probed duration as authoritative.
          * If declared duration is also supplied and disagrees beyond tolerance
            (or is non-positive), emit a safe ``media_duration_mismatch`` limitation.
        - When probed duration is unavailable (None):
          * If declared duration is supplied and non-positive (<= 0), fail validation
            with StageValidationError.
          * If declared duration is positive, use it.
          * If neither probed nor declared duration is available, return None and emit
            an explicit ``media_duration_unavailable`` limitation instead of defaulting
            to an arbitrary length.
    """
    if probed_duration_ms is not None:
        if probed_duration_ms <= 0:
            raise StageValidationError(
                f"{media_role} media probed duration must be positive "
                f"(got {probed_duration_ms}ms).",
                stage=stage,
                code=ErrorCode.VALIDATION_ERROR,
                safe_message=f"{media_role} media has zero or invalid duration.",
            )

        limitations: list[Limitation] = []
        if declared_duration_ms is not None and (
            declared_duration_ms <= 0
            or abs(probed_duration_ms - declared_duration_ms) > mismatch_tolerance_ms
        ):
            logger.warning(
                "%s declared duration (%dms) mismatched probed duration (%dms); "
                "using probed duration.",
                media_role,
                declared_duration_ms,
                probed_duration_ms,
            )
            limitations.append(create_duration_mismatch_limitation())
        return probed_duration_ms, limitations

    if declared_duration_ms is not None:
        if declared_duration_ms <= 0:
            raise StageValidationError(
                f"{media_role} media declared duration must be positive "
                f"(got {declared_duration_ms}ms).",
                stage=stage,
                code=ErrorCode.VALIDATION_ERROR,
                safe_message=f"{media_role} media has zero or invalid duration.",
            )
        return declared_duration_ms, []

    return None, [create_missing_duration_limitation()]


class _PreparedOutput(NamedTuple):
    """Resolved output path paired with ownership metadata for failure cleanup."""

    path: Path
    created_temp: bool
    existed_before: bool

    @property
    def is_owned(self) -> bool:
        return self.created_temp or not self.existed_before


def _resolve_positive_seconds(
    explicit: float | None,
    *,
    env_var: str,
    default: float,
    label: str,
) -> float:
    """Resolve a positive finite duration in seconds from argument, env var, or default."""
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.getenv(env_var, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError as err:
            raise ValueError(f"{env_var} must be a positive finite number.") from err
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number.")
    return value


def get_media_timeout_seconds(timeout_seconds: float | None = None) -> float:
    """Return the effective deadline in seconds for FFmpeg media processing."""
    return _resolve_positive_seconds(
        timeout_seconds,
        env_var=MEDIA_TIMEOUT_ENV,
        default=DEFAULT_MEDIA_TIMEOUT_SECONDS,
        label="timeout_seconds",
    )


def get_media_terminate_grace_seconds(terminate_grace_seconds: float | None = None) -> float:
    """Return the grace period in seconds before escalating FFmpeg termination to SIGKILL."""
    return _resolve_positive_seconds(
        terminate_grace_seconds,
        env_var=MEDIA_TERMINATE_GRACE_ENV,
        default=DEFAULT_MEDIA_TERMINATE_GRACE_SECONDS,
        label="terminate_grace_seconds",
    )


def _resolve_input_file(input_path: Path | str | AssetInput) -> Path:
    input_file = (
        Path(input_path.object_key) if isinstance(input_path, AssetInput) else Path(input_path)
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return input_file


def _prepare_output_path(
    output_path: Path | str | None,
    *,
    temp_root: Path,
    suffix: str,
) -> _PreparedOutput:
    if output_path is None:
        temp_root.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(suffix=suffix, dir=str(temp_root))
        os.close(fd)
        return _PreparedOutput(path=Path(temp_path_str), created_temp=True, existed_before=False)

    target = Path(output_path)
    existed_before = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    return _PreparedOutput(path=target, created_temp=False, existed_before=existed_before)


async def _reap_uninterruptibly(proc: asyncio.subprocess.Process) -> None:
    """Await proc.wait() until the child is reaped, re-raising any later cancellation."""
    pending_cancel: asyncio.CancelledError | None = None
    wait_task = asyncio.ensure_future(proc.wait())
    while not wait_task.done():
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError as err:
            pending_cancel = err
    wait_task.result()
    if pending_cancel is not None:
        raise pending_cancel


async def _terminate_and_reap(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    """Terminate a running child process, escalate to kill if needed, and reap it."""
    # Checking isinstance(..., int) handles real Processes (None while running, int on exit)
    # and prevents unconfigured mock objects from bypassing termination and reaping.
    if isinstance(proc.returncode, int) and not isinstance(proc.returncode, bool):
        return

    with contextlib.suppress(ProcessLookupError):
        proc.terminate()

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await _reap_uninterruptibly(proc)
        raise

    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    await _reap_uninterruptibly(proc)


async def _run_ffmpeg_command(
    cmd: list[str],
    *,
    ffmpeg_bin: str,
    timeout_seconds: float,
    terminate_grace_seconds: float,
) -> tuple[int, str]:
    """Execute ffmpeg with an explicit deadline, terminating and reaping on cancel or timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffmpeg binary '{ffmpeg_bin}' was not found in PATH.") from err

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError as err:
        await _terminate_and_reap(proc, grace_seconds=terminate_grace_seconds)
        raise FFmpegTimeoutError(
            f"ffmpeg process exceeded {timeout_seconds:g}s stage deadline."
        ) from err
    except BaseException:
        await _terminate_and_reap(proc, grace_seconds=terminate_grace_seconds)
        raise

    returncode = proc.returncode if isinstance(proc.returncode, int) else 0
    stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
    return returncode, stderr_text


async def normalize_media(
    input_path: Path | str | AssetInput,
    output_path: Path | str | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    timeout_seconds: float | None = None,
    terminate_grace_seconds: float | None = None,
) -> MediaNormalizationResult:
    """Normalize input media file to canonical 16kHz mono 16-bit PCM WAV via ffmpeg.

    Accepts arbitrary video and audio container formats (.mp4, .webm, .mov, .mkv, .avi, etc.)
    with arbitrary codecs and sample rates (e.g. 44.1kHz or 48kHz stereo) and normalizes
    them into a uniform 16kHz mono 16-bit PCM WAV:
        ffmpeg -y -i <input> -vn -acodec pcm_s16le -ar 16000 -ac 1 <output.wav>

    Args:
        input_path: Path to the input video or audio file, or AssetInput object.
        output_path: Destination path for normalized WAV. If None, a temp file is created.
        ffmpeg_bin: Path or name of the ffmpeg binary.
        target_sample_rate: Target sample rate in Hz (default 16000).
        target_channels: Target channel count (default 1 for mono).
        timeout_seconds: Explicit stage deadline in seconds (defaults to AI_MEDIA_TIMEOUT_SECONDS
            or 300s).
        terminate_grace_seconds: Grace period in seconds after SIGTERM before escalating to
            SIGKILL (defaults to AI_MEDIA_TERMINATE_GRACE_SECONDS or 5s).

    Returns:
        MediaNormalizationResult containing output path, duration_ms, sample_rate, and channels.

    Raises:
        FileNotFoundError: If the input file does not exist.
        FFmpegNotFoundError: If ffmpeg is not installed or not in PATH.
        FFmpegTimeoutError: If ffmpeg exceeds the stage deadline.
        FFmpegExecutionError: If ffmpeg fails with non-zero exit code.
        MediaNormalizationError: If output WAV metadata cannot be read.
    """
    resolved_timeout = get_media_timeout_seconds(timeout_seconds)
    resolved_grace = get_media_terminate_grace_seconds(terminate_grace_seconds)
    input_file = _resolve_input_file(input_path)

    temp_root = get_media_temp_root()
    target = _prepare_output_path(output_path, temp_root=temp_root, suffix="_normalized.wav")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sample_rate),
        "-ac",
        str(target_channels),
        str(target.path),
    ]

    try:
        returncode, stderr_text = await _run_ffmpeg_command(
            cmd,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=resolved_timeout,
            terminate_grace_seconds=resolved_grace,
        )
        if returncode != 0:
            raise FFmpegExecutionError(
                f"ffmpeg process exited with code {returncode}",
                returncode=returncode,
                stderr=stderr_text,
            )

        meta = get_wav_metadata(target.path)
    except BaseException:
        _unlink_owned_output(target.path, input_file, should_unlink=target.is_owned)
        if target.created_temp:
            _remove_empty_dir(temp_root)
        raise

    return MediaNormalizationResult(
        output_path=target.path,
        duration_ms=meta.duration_ms,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
    )


async def split_media(
    input_path: Path | str | AssetInput,
    *,
    output_audio_path: Path | str | None = None,
    output_video_path: Path | str | None = None,
    ffmpeg_bin: str = "ffmpeg",
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    timeout_seconds: float | None = None,
    terminate_grace_seconds: float | None = None,
) -> MediaSplitResult:
    """Split presentation video into 16kHz mono WAV audio and audio-free video for vision.

    In a single ffmpeg invocation, demuxes the input presentation:
      - Audio stream: decoded & resampled to 16kHz mono 16-bit PCM WAV (for Whisper & Diarization)
      - Video stream: extracted with audio stripped (-an -c:v copy) for computer vision models

    If the input media is audio-only (no video stream), gracefully normalizes audio and sets
    video_path to None.

    Args:
        input_path: Path to the input video/audio file, or AssetInput object.
        output_audio_path: Destination path for normalized WAV. If None, a temp file is created.
        output_video_path: Destination path for vision video stream.
            If None, a temp file is created.
        ffmpeg_bin: Path or name of the ffmpeg binary.
        target_sample_rate: Target audio sample rate in Hz (default 16000).
        target_channels: Target audio channel count (default 1 for mono).
        timeout_seconds: Explicit stage deadline in seconds (defaults to AI_MEDIA_TIMEOUT_SECONDS
            or 300s).
        terminate_grace_seconds: Grace period in seconds after SIGTERM before escalating to
            SIGKILL (defaults to AI_MEDIA_TERMINATE_GRACE_SECONDS or 5s).

    Returns:
        MediaSplitResult containing audio_path, video_path, duration_ms, sample_rate, and channels.

    Raises:
        FileNotFoundError: If the input file does not exist.
        FFmpegNotFoundError: If ffmpeg is not installed or not in PATH.
        FFmpegTimeoutError: If ffmpeg exceeds the stage deadline.
        FFmpegExecutionError: If ffmpeg fails with non-zero exit code.
        MediaNormalizationError: If output WAV metadata cannot be read.
    """
    resolved_timeout = get_media_timeout_seconds(timeout_seconds)
    resolved_grace = get_media_terminate_grace_seconds(terminate_grace_seconds)
    input_file = _resolve_input_file(input_path)

    temp_root = get_media_temp_root()
    audio_out = _prepare_output_path(
        output_audio_path, temp_root=temp_root, suffix="_normalized.wav"
    )
    try:
        video_suffix = input_file.suffix if input_file.suffix else ".mp4"
        video_out = _prepare_output_path(
            output_video_path, temp_root=temp_root, suffix=f"_vision{video_suffix}"
        )
    except BaseException:
        _unlink_owned_output(audio_out.path, input_file, should_unlink=audio_out.is_owned)
        if audio_out.created_temp:
            _remove_empty_dir(temp_root)
        raise

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sample_rate),
        "-ac",
        str(target_channels),
        str(audio_out.path),
        "-an",
        "-c:v",
        "copy",
        str(video_out.path),
    ]

    started_at = time.monotonic()
    try:
        returncode, stderr_text = await _run_ffmpeg_command(
            cmd,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=resolved_timeout,
            terminate_grace_seconds=resolved_grace,
        )
        has_video = True

        if returncode != 0:
            if (
                "Output file does not contain any stream" in stderr_text
                or "matches no streams" in stderr_text
            ):
                _unlink_owned_output(video_out.path, input_file, should_unlink=video_out.is_owned)
                remaining_timeout = resolved_timeout - (time.monotonic() - started_at)
                if remaining_timeout <= 0:
                    raise FFmpegTimeoutError(
                        f"ffmpeg process exceeded {resolved_timeout:g}s stage deadline."
                    )
                norm_res = await normalize_media(
                    input_file,
                    output_path=audio_out.path,
                    ffmpeg_bin=ffmpeg_bin,
                    target_sample_rate=target_sample_rate,
                    target_channels=target_channels,
                    timeout_seconds=remaining_timeout,
                    terminate_grace_seconds=resolved_grace,
                )
                return MediaSplitResult(
                    audio_path=norm_res.output_path,
                    video_path=None,
                    duration_ms=norm_res.duration_ms,
                    sample_rate=norm_res.sample_rate,
                    channels=norm_res.channels,
                )
            raise FFmpegExecutionError(
                f"ffmpeg process exited with code {returncode}",
                returncode=returncode,
                stderr=stderr_text,
            )

        meta = get_wav_metadata(audio_out.path)
    except BaseException:
        _unlink_owned_output(audio_out.path, input_file, should_unlink=audio_out.is_owned)
        _unlink_owned_output(video_out.path, input_file, should_unlink=video_out.is_owned)
        if audio_out.created_temp or video_out.created_temp:
            _remove_empty_dir(temp_root)
        raise

    return MediaSplitResult(
        audio_path=audio_out.path,
        video_path=video_out.path if has_video else None,
        duration_ms=meta.duration_ms,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
    )


__all__ = [
    "DEFAULT_JOB_TEMP_DIRNAME",
    "DEFAULT_MEDIA_TERMINATE_GRACE_SECONDS",
    "DEFAULT_MEDIA_TIMEOUT_SECONDS",
    "DURATION_MISMATCH_TOLERANCE_MS",
    "JOB_TEMP_ROOT_ENV",
    "MEDIA_NORMALIZATION_VERSION",
    "MEDIA_TERMINATE_GRACE_ENV",
    "MEDIA_TIMEOUT_ENV",
    "SUPPORTED_MEDIA_EXTENSIONS",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "FFmpegTimeoutError",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "WavMetadata",
    "cleanup_job_temp_dirs_for_scope",
    "cleanup_stale_worker_temp_dirs",
    "create_duration_mismatch_limitation",
    "create_missing_duration_limitation",
    "create_normalization_limitation",
    "get_media_temp_root",
    "get_media_terminate_grace_seconds",
    "get_media_timeout_seconds",
    "get_wav_metadata",
    "normalize_media",
    "owned_job_temp_dir",
    "resolve_media_duration",
    "split_media",
]
