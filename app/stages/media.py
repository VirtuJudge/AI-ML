"""FFmpeg-based audio normalization stage for VirtuJudge AI-ML.

Extracts and normalizes audio from arbitrary video/audio containers (.mp4, .webm,
.mov, .mkv, .avi, etc.) with arbitrary codecs and sample rates (e.g., 44.1kHz or 48kHz stereo)
into a canonical 16kHz, mono, 16-bit PCM WAV:
    ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav

Both Whisper (or Groq Whisper) and pyannote expect 16kHz mono audio. Standardizing
once upfront saves downstream stages from repeated audio conversions.
"""

import asyncio
import os
import tempfile
import wave
from pathlib import Path

from pydantic import BaseModel, Field

from app.contracts import AssetInput, Limitation

SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset({
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
})


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


def get_wav_metadata(wav_path: Path) -> tuple[int, int, int]:
    """Read duration_ms, sample_rate, and channels from a WAV file header."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_ms = round((n_frames / float(sample_rate)) * 1000) if sample_rate > 0 else 0
            return duration_ms, sample_rate, channels
    except Exception as err:
        raise MediaNormalizationError("Failed to read normalized WAV file header.") from err


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

    if output_path is None:
        fd, temp_path_str = tempfile.mkstemp(suffix="_normalized.wav")
        os.close(fd)
        target_output = Path(temp_path_str)
    else:
        target_output = Path(output_path)
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffmpeg binary '{ffmpeg_bin}' was not found in PATH.") from err

    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")
        raise FFmpegExecutionError(
            f"ffmpeg process exited with code {proc.returncode}",
            returncode=proc.returncode,
            stderr=stderr_text,
        )

    duration_ms, sample_rate, channels = get_wav_metadata(target_output)

    return MediaNormalizationResult(
        output_path=target_output,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
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

    if output_audio_path is None:
        fd_a, temp_audio_str = tempfile.mkstemp(suffix="_normalized.wav")
        os.close(fd_a)
        target_audio = Path(temp_audio_str)
    else:
        target_audio = Path(output_audio_path)
        target_audio.parent.mkdir(parents=True, exist_ok=True)

    if output_video_path is None:
        suffix = input_file.suffix if input_file.suffix else ".mp4"
        fd_v, temp_video_str = tempfile.mkstemp(suffix=f"_vision{suffix}")
        os.close(fd_v)
        target_video = Path(temp_video_str)
    else:
        target_video = Path(output_video_path)
        target_video.parent.mkdir(parents=True, exist_ok=True)

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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffmpeg binary '{ffmpeg_bin}' was not found in PATH.") from err

    _, stderr = await proc.communicate()
    has_video = True

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")
        if (
            "Output file does not contain any stream" in stderr_text
            or "matches no streams" in stderr_text
        ):
            norm_res = await normalize_media(
                input_file,
                output_path=target_audio,
                ffmpeg_bin=ffmpeg_bin,
                target_sample_rate=target_sample_rate,
                target_channels=target_channels,
            )
            if target_video.exists():
                target_video.unlink(missing_ok=True)
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

    duration_ms, sample_rate, channels = get_wav_metadata(target_audio)

    return MediaSplitResult(
        audio_path=target_audio,
        video_path=target_video if has_video else None,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
    )


__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "MediaNormalizationError",
    "MediaNormalizationResult",
    "MediaSplitResult",
    "create_normalization_limitation",
    "get_wav_metadata",
    "normalize_media",
    "split_media",
]
