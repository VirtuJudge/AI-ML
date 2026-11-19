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
import os
import re
import shutil
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field

from app.contracts import AssetInput, Limitation

MEDIA_NORMALIZATION_VERSION = "ffmpeg_pcm_s16le_16k_mono_v1"
JOB_TEMP_ROOT_ENV = "AI_MEDIA_TEMP_DIR"
DEFAULT_JOB_TEMP_DIRNAME = "virtujudge_media_jobs"

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


async def normalize_media(
    input_path: Path | str | AssetInput,
    output_path: Path | str | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    target_sample_rate: int = 16000,
    target_channels: int = 1,
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

    Returns:
        MediaNormalizationResult containing output path, duration_ms, sample_rate, and channels.

    Raises:
        FileNotFoundError: If the input file does not exist.
        FFmpegNotFoundError: If ffmpeg is not installed or not in PATH.
        FFmpegExecutionError: If ffmpeg fails with non-zero exit code.
        MediaNormalizationError: If output WAV metadata cannot be read.
    """
    if isinstance(input_path, AssetInput):
        input_file = Path(input_path.object_key)
    else:
        input_file = Path(input_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    temp_root = get_media_temp_root()
    created_temp = False
    existed_before = False
    if output_path is None:
        temp_root.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(suffix="_normalized.wav", dir=str(temp_root))
        os.close(fd)
        target_output = Path(temp_path_str)
        created_temp = True
    else:
        target_output = Path(output_path)
        existed_before = target_output.exists()
        target_output.parent.mkdir(parents=True, exist_ok=True)

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
        str(target_output),
    ]

    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as err:
            raise FFmpegNotFoundError(
                f"ffmpeg binary '{ffmpeg_bin}' was not found in PATH."
            ) from err

        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            raise FFmpegExecutionError(
                f"ffmpeg process exited with code {proc.returncode}",
                returncode=proc.returncode,
                stderr=stderr_text,
            )

        meta = get_wav_metadata(target_output)
    except BaseException:
        _unlink_owned_output(
            target_output,
            input_file,
            should_unlink=created_temp or not existed_before,
        )
        if created_temp:
            _remove_empty_dir(temp_root)
        raise

    return MediaNormalizationResult(
        output_path=target_output,
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

    Returns:
        MediaSplitResult containing audio_path, video_path, duration_ms, sample_rate, and channels.
    """
    if isinstance(input_path, AssetInput):
        input_file = Path(input_path.object_key)
    else:
        input_file = Path(input_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    temp_root = get_media_temp_root()
    created_audio_temp = False
    created_video_temp = False
    audio_existed_before = False
    video_existed_before = False

    if output_audio_path is None:
        temp_root.mkdir(parents=True, exist_ok=True)
        fd_a, temp_audio_str = tempfile.mkstemp(suffix="_normalized.wav", dir=str(temp_root))
        os.close(fd_a)
        target_audio = Path(temp_audio_str)
        created_audio_temp = True
    else:
        target_audio = Path(output_audio_path)
        audio_existed_before = target_audio.exists()
        target_audio.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_video_path is None:
            temp_root.mkdir(parents=True, exist_ok=True)
            suffix = input_file.suffix if input_file.suffix else ".mp4"
            fd_v, temp_video_str = tempfile.mkstemp(suffix=f"_vision{suffix}", dir=str(temp_root))
            os.close(fd_v)
            target_video = Path(temp_video_str)
            created_video_temp = True
        else:
            target_video = Path(output_video_path)
            video_existed_before = target_video.exists()
            target_video.parent.mkdir(parents=True, exist_ok=True)
    except BaseException:
        _unlink_owned_output(
            target_audio,
            input_file,
            should_unlink=created_audio_temp or not audio_existed_before,
        )
        if created_audio_temp:
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
        str(target_audio),
        "-an",
        "-c:v",
        "copy",
        str(target_video),
    ]

    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as err:
            raise FFmpegNotFoundError(
                f"ffmpeg binary '{ffmpeg_bin}' was not found in PATH."
            ) from err

        _, stderr = await proc.communicate()
        has_video = True

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if (
                "Output file does not contain any stream" in stderr_text
                or "matches no streams" in stderr_text
            ):
                _unlink_owned_output(
                    target_video,
                    input_file,
                    should_unlink=created_video_temp or not video_existed_before,
                )
                norm_res = await normalize_media(
                    input_file,
                    output_path=target_audio,
                    ffmpeg_bin=ffmpeg_bin,
                    target_sample_rate=target_sample_rate,
                    target_channels=target_channels,
                )
                return MediaSplitResult(
                    audio_path=norm_res.output_path,
                    video_path=None,
                    duration_ms=norm_res.duration_ms,
                    sample_rate=norm_res.sample_rate,
                    channels=norm_res.channels,
                )
            raise FFmpegExecutionError(
                f"ffmpeg process exited with code {proc.returncode}",
                returncode=proc.returncode,
                stderr=stderr_text,
            )

        meta = get_wav_metadata(target_audio)
    except BaseException:
        _unlink_owned_output(
            target_audio,
            input_file,
            should_unlink=created_audio_temp or not audio_existed_before,
        )
        _unlink_owned_output(
            target_video,
            input_file,
            should_unlink=created_video_temp or not video_existed_before,
        )
        if created_audio_temp or created_video_temp:
            _remove_empty_dir(temp_root)
        raise

    return MediaSplitResult(
        audio_path=target_audio,
        video_path=target_video if has_video else None,
        duration_ms=meta.duration_ms,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
    )


__all__ = [
    "DEFAULT_JOB_TEMP_DIRNAME",
    "JOB_TEMP_ROOT_ENV",
    "MEDIA_NORMALIZATION_VERSION",
    "SUPPORTED_MEDIA_EXTENSIONS",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "WavMetadata",
    "cleanup_job_temp_dirs_for_scope",
    "cleanup_stale_worker_temp_dirs",
    "create_normalization_limitation",
    "get_media_temp_root",
    "get_wav_metadata",
    "normalize_media",
    "owned_job_temp_dir",
    "split_media",
]
