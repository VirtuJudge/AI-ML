"""Unit tests for app.stages.media (FFmpeg-based audio normalization)."""

import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.contracts import AssetInput, Limitation
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


def _create_minimal_wav(
    path: Path, duration_ms: int = 2000, sample_rate: int = 16000, channels: int = 1
) -> Path:
    """Helper to create a small valid PCM WAV file for tests."""
    n_frames = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames * channels)
    return path


def test_get_wav_metadata(tmp_path: Path) -> None:
    """Verify get_wav_metadata reads correct duration, sample_rate, and channels."""
    wav_path = tmp_path / "sample.wav"
    _create_minimal_wav(wav_path, duration_ms=2500, sample_rate=16000, channels=1)

    duration_ms, sample_rate, channels = get_wav_metadata(wav_path)
    assert duration_ms == 2500
    assert sample_rate == 16000
    assert channels == 1


def test_get_wav_metadata_invalid_file(tmp_path: Path) -> None:
    """Verify get_wav_metadata raises MediaNormalizationError on corrupted/non-WAV file."""
    bad_file = tmp_path / "bad.wav"
    bad_file.write_text("not a real wav file content")

    with pytest.raises(MediaNormalizationError, match="Failed to read normalized WAV file header"):
        get_wav_metadata(bad_file)


@pytest.mark.asyncio
async def test_normalize_media_success(tmp_path: Path) -> None:
    """Verify normalize_media executes ffmpeg and returns MediaNormalizationResult."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"dummy video content")
    output_file = tmp_path / "output.wav"

    async def fake_communicate():
        # Simulate ffmpeg creating the output wav
        _create_minimal_wav(output_file, duration_ms=3000, sample_rate=16000, channels=1)
        return (b"", b"")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(side_effect=fake_communicate)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await normalize_media(input_file, output_path=output_file)

        assert isinstance(result, MediaNormalizationResult)
        assert result.output_path == output_file
        assert result.duration_ms == 3000
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.duration_s == 3.0

        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0]
        assert "ffmpeg" in cmd[0]
        assert "-vn" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd


@pytest.mark.asyncio
async def test_normalize_media_missing_input(tmp_path: Path) -> None:
    """Verify normalize_media raises FileNotFoundError when input file does not exist."""
    missing = tmp_path / "non_existent.mp4"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        await normalize_media(missing)


@pytest.mark.asyncio
async def test_normalize_media_ffmpeg_not_found(tmp_path: Path) -> None:
    """Verify normalize_media raises FFmpegNotFoundError when ffmpeg is not installed."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"dummy")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No such binary")),
        pytest.raises(FFmpegNotFoundError, match=r"ffmpeg binary.*was not found in PATH"),
    ):
        await normalize_media(input_file)


@pytest.mark.asyncio
async def test_normalize_media_execution_error(tmp_path: Path) -> None:
    """Verify normalize_media raises FFmpegExecutionError when ffmpeg fails with non-zero exit."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"dummy")

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"Invalid data found when processing input")
    )

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        pytest.raises(FFmpegExecutionError) as exc_info,
    ):
        await normalize_media(input_file)

    assert exc_info.value.returncode == 1
    assert "Invalid data found" in exc_info.value.stderr


def test_create_normalization_limitation() -> None:
    """Verify create_normalization_limitation returns a valid log-safe Limitation."""
    limitation = create_normalization_limitation()
    assert isinstance(limitation, Limitation)
    assert limitation.code == "media_normalization_unavailable"
    assert limitation.scope == "media"
    assert "unavailable" in limitation.message or "failed" in limitation.message


def test_supported_media_extensions() -> None:
    """Verify common video containers are present in SUPPORTED_MEDIA_EXTENSIONS."""
    for ext in [".mp4", ".webm", ".mov", ".mkv", ".wav"]:
        assert ext in SUPPORTED_MEDIA_EXTENSIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", [".mp4", ".webm", ".mov", ".mkv"])
async def test_normalize_media_various_containers(tmp_path: Path, extension: str) -> None:
    """Verify normalize_media accepts various container formats (.mp4, .webm, .mov, .mkv)."""
    input_file = tmp_path / f"presentation{extension}"
    input_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.wav"

    async def fake_communicate():
        _create_minimal_wav(output_file, duration_ms=2000, sample_rate=16000, channels=1)
        return (b"", b"")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(side_effect=fake_communicate)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await normalize_media(input_file, output_path=output_file)

        assert isinstance(result, MediaNormalizationResult)
        assert result.output_path == output_file
        assert result.sample_rate == 16000
        assert result.channels == 1

        cmd = mock_exec.call_args[0]
        assert str(input_file) in cmd
        assert "-vn" in cmd
        assert "pcm_s16le" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd


@pytest.mark.asyncio
async def test_normalize_media_accepts_asset_input(tmp_path: Path) -> None:
    """Verify normalize_media accepts an AssetInput object directly."""
    input_file = tmp_path / "presentation.webm"
    input_file.write_bytes(b"dummy webm data")
    output_file = tmp_path / "output.wav"

    asset = AssetInput(
        artifact_id="01JTEST0000000000000000001",
        object_key=str(input_file),
        checksum="sha256:" + "a" * 64,
        media_type="video/webm",
        duration_ms=4000,
    )

    async def fake_communicate():
        _create_minimal_wav(output_file, duration_ms=4000, sample_rate=16000, channels=1)
        return (b"", b"")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(side_effect=fake_communicate)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await normalize_media(asset, output_path=output_file)

        assert isinstance(result, MediaNormalizationResult)
        assert result.duration_ms == 4000
        cmd = mock_exec.call_args[0]
        assert str(input_file) in cmd


@pytest.mark.asyncio
async def test_split_media_success(tmp_path: Path) -> None:
    """Verify split_media separates presentation video into audio and vision video."""
    input_file = tmp_path / "presentation.mp4"
    input_file.write_bytes(b"dummy video data")
    output_audio = tmp_path / "custom_audio.wav"
    output_video = tmp_path / "custom_video.mp4"

    async def fake_communicate():
        _create_minimal_wav(output_audio, duration_ms=5000, sample_rate=16000, channels=1)
        output_video.write_bytes(b"dummy vision video")
        return (b"", b"")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(side_effect=fake_communicate)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await split_media(
            input_file,
            output_audio_path=output_audio,
            output_video_path=output_video,
        )

        assert isinstance(result, MediaSplitResult)
        assert result.audio_path == output_audio
        assert result.video_path == output_video
        assert result.duration_ms == 5000
        assert result.sample_rate == 16000
        assert result.channels == 1

        cmd = mock_exec.call_args[0]
        assert "-vn" in cmd
        assert "pcm_s16le" in cmd
        assert "-an" in cmd
        assert "-c:v" in cmd
        assert "copy" in cmd


@pytest.mark.asyncio
async def test_split_media_audio_only_fallback(tmp_path: Path) -> None:
    """Verify split_media handles audio-only inputs by setting video_path to None."""
    input_file = tmp_path / "podcast.wav"
    input_file.write_bytes(b"dummy audio data")
    output_audio = tmp_path / "normalized.wav"

    mock_fail_proc = AsyncMock()
    mock_fail_proc.returncode = 1
    mock_fail_proc.communicate = AsyncMock(
        return_value=(b"", b"Output file does not contain any stream")
    )

    fake_norm = MediaNormalizationResult(
        output_path=output_audio,
        duration_ms=3000,
        sample_rate=16000,
        channels=1,
    )

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_fail_proc),
        patch("app.stages.media.normalize_media", new_callable=AsyncMock) as mock_norm,
    ):
        mock_norm.return_value = fake_norm
        result = await split_media(input_file, output_audio_path=output_audio)

        assert isinstance(result, MediaSplitResult)
        assert result.audio_path == output_audio
        assert result.video_path is None
        assert result.duration_ms == 3000


