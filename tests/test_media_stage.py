"""Unit tests for app.stages.media (FFmpeg-based audio normalization)."""

import asyncio
import sys
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.contracts import AssetInput, Limitation
from app.stages import media as media_stage
from app.stages.media import (
    JOB_TEMP_ROOT_ENV,
    MEDIA_TERMINATE_GRACE_ENV,
    MEDIA_TIMEOUT_ENV,
    SUPPORTED_MEDIA_EXTENSIONS,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    MediaNormalizationError,
    MediaNormalizationResult,
    MediaSplitResult,
    WavMetadata,
    create_normalization_limitation,
    get_wav_metadata,
    normalize_media,
    split_media,
)


def _create_minimal_wav(
    path: Path, duration_ms: int = 2000, sample_rate: int = 16000, channels: int = 1
) -> Path:
    """Helper to create a minimal valid PCM WAV file on disk for testing."""
    n_frames = int((duration_ms / 1000.0) * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames * channels)
    return path


def test_get_wav_metadata(tmp_path: Path) -> None:
    """Verify get_wav_metadata returns WavMetadata with explicit fields."""
    wav_path = tmp_path / "sample.wav"
    _create_minimal_wav(wav_path, duration_ms=2500, sample_rate=16000, channels=1)

    meta = get_wav_metadata(wav_path)
    assert isinstance(meta, WavMetadata)
    assert meta.duration_ms == 2500
    assert meta.sample_rate == 16000
    assert meta.channels == 1

    # Also verify backward-compatible tuple unpacking by position
    duration_ms, sample_rate, channels = meta
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


class _ControlledSubprocess:
    """Controlled fake asyncio.subprocess.Process recording lifecycle events."""

    def __init__(
        self,
        *,
        graceful_exit: bool = True,
        on_start: Any = None,
    ) -> None:
        self.returncode: int | None = None
        self.graceful_exit = graceful_exit
        self.on_start = on_start
        self.events: list[str] = []
        self.communicate_started = asyncio.Event()
        self._stopped = asyncio.Event()
        self.stopped = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.events.append("communicate_start")
        if self.on_start is not None:
            self.on_start()
        self.communicate_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.events.append("communicate_cancelled")
            raise
        return (b"", b"")

    def terminate(self) -> None:
        self.events.append("terminate")
        if self.graceful_exit:
            self.returncode = -15
            self.stopped = True
            self._stopped.set()

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9
        self.stopped = True
        self._stopped.set()

    async def wait(self) -> int:
        self.events.append("wait_enter")
        await self._stopped.wait()
        self.events.append("wait_exit")
        return self.returncode if self.returncode is not None else -9


def _install_unlink_order_spy(
    monkeypatch: pytest.MonkeyPatch,
    proc: _ControlledSubprocess,
) -> list[tuple[Path, bool]]:
    """Spy on _unlink_owned_output to prove child stopped before partial files are removed."""
    original = media_stage._unlink_owned_output
    observed: list[tuple[Path, bool]] = []

    def spy(path: Path, source_path: Path, *, should_unlink: bool) -> None:
        proc.events.append(f"unlink:{path.name}:{should_unlink}")
        observed.append((path, proc.stopped))
        assert proc.stopped, f"Attempted to clean {path} before child process stopped!"
        original(path, source_path, should_unlink=should_unlink)

    monkeypatch.setattr(media_stage, "_unlink_owned_output", spy)
    return observed


@pytest.mark.asyncio
async def test_normalize_media_normal_completion_does_not_terminate_or_kill(
    tmp_path: Path,
) -> None:
    """Normal completion returns metadata and never calls terminate, kill, or wait."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"video")
    output_file = tmp_path / "out.wav"

    proc = _ControlledSubprocess()

    async def fast_communicate() -> tuple[bytes, bytes]:
        proc.events.append("communicate_start")
        _create_minimal_wav(output_file, duration_ms=1500, sample_rate=16000, channels=1)
        proc.returncode = 0
        proc.stopped = True
        return (b"", b"")

    proc.communicate = fast_communicate  # type: ignore[method-assign]

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await normalize_media(input_file, output_path=output_file, timeout_seconds=5.0)

    assert result.output_path == output_file
    assert result.duration_ms == 1500
    assert proc.events == ["communicate_start"]


@pytest.mark.asyncio
async def test_split_media_normal_completion_does_not_terminate_or_kill(
    tmp_path: Path,
) -> None:
    """Normal split_media completion returns metadata and never calls terminate, kill, or wait."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"video")
    out_audio = tmp_path / "audio.wav"
    out_video = tmp_path / "vision.mp4"

    proc = _ControlledSubprocess()

    async def fast_communicate() -> tuple[bytes, bytes]:
        proc.events.append("communicate_start")
        _create_minimal_wav(out_audio, duration_ms=2500, sample_rate=16000, channels=1)
        out_video.write_bytes(b"vision")
        proc.returncode = 0
        proc.stopped = True
        return (b"", b"")

    proc.communicate = fast_communicate  # type: ignore[method-assign]

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await split_media(
            input_file,
            output_audio_path=out_audio,
            output_video_path=out_video,
            timeout_seconds=5.0,
        )

    assert result.audio_path == out_audio
    assert result.video_path == out_video
    assert result.duration_ms == 2500
    assert proc.events == ["communicate_start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("graceful_exit", [True, False], ids=["graceful_exit", "forced_kill"])
async def test_normalize_media_cancellation_reaps_child_and_cleans_only_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graceful_exit: bool,
) -> None:
    """Cancelling normalize_media terminates/kills/reaps FFmpeg before cleaning owned output."""
    temp_root = tmp_path / "temp_root"
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))

    input_file = tmp_path / "private_input.mp4"
    input_file.write_bytes(b"private input")
    owned_output = tmp_path / "owned_partial.wav"

    proc = _ControlledSubprocess(
        graceful_exit=graceful_exit,
        on_start=lambda: owned_output.write_bytes(b"partial wav bytes"),
    )
    observed = _install_unlink_order_spy(monkeypatch, proc)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(
            normalize_media(
                input_file,
                output_path=owned_output,
                timeout_seconds=5.0,
                terminate_grace_seconds=0.05,
            )
        )
        await proc.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert proc.stopped is True
    assert "terminate" in proc.events
    assert "wait_exit" in proc.events
    if graceful_exit:
        assert "kill" not in proc.events
    else:
        assert "kill" in proc.events
        assert proc.events.index("terminate") < proc.events.index("kill")
        assert proc.events.index("kill") < proc.events.index("wait_exit")

    assert observed == [(owned_output, True)]
    assert not owned_output.exists()
    assert input_file.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("graceful_exit", [True, False], ids=["graceful_exit", "forced_kill"])
async def test_split_media_cancellation_reaps_child_and_cleans_only_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graceful_exit: bool,
) -> None:
    """Cancelling split_media terminates/kills/reaps FFmpeg before cleaning owned outputs."""
    input_file = tmp_path / "private_input.mp4"
    input_file.write_bytes(b"private input")
    owned_audio = tmp_path / "owned_audio.wav"
    existing_video = tmp_path / "caller_existing_video.mp4"
    existing_video.write_bytes(b"caller video")

    def write_partials() -> None:
        owned_audio.write_bytes(b"partial audio")

    proc = _ControlledSubprocess(graceful_exit=graceful_exit, on_start=write_partials)
    observed = _install_unlink_order_spy(monkeypatch, proc)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(
            split_media(
                input_file,
                output_audio_path=owned_audio,
                output_video_path=existing_video,
                timeout_seconds=5.0,
                terminate_grace_seconds=0.05,
            )
        )
        await proc.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert proc.stopped is True
    assert "terminate" in proc.events
    assert "wait_exit" in proc.events
    if graceful_exit:
        assert "kill" not in proc.events
    else:
        assert "kill" in proc.events
        assert proc.events.index("terminate") < proc.events.index("kill")
        assert proc.events.index("kill") < proc.events.index("wait_exit")

    assert observed == [(owned_audio, True), (existing_video, True)]
    assert not owned_audio.exists()
    assert existing_video.read_bytes() == b"caller video"
    assert input_file.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("graceful_exit", [True, False], ids=["graceful_exit", "forced_kill"])
async def test_normalize_media_timeout_reaps_child_and_cleans_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graceful_exit: bool,
) -> None:
    """Timeout in normalize_media terminates/kills/reaps FFmpeg and cleans owned temp file."""
    temp_root = tmp_path / "temp_root"
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    monkeypatch.setenv(MEDIA_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(MEDIA_TERMINATE_GRACE_ENV, "0.05")

    input_file = tmp_path / "stalled.mp4"
    input_file.write_bytes(b"stalled input")

    proc = _ControlledSubprocess(graceful_exit=graceful_exit)
    observed = _install_unlink_order_spy(monkeypatch, proc)

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        pytest.raises(FFmpegTimeoutError, match="stage deadline") as exc_info,
    ):
        await normalize_media(input_file)

    assert isinstance(exc_info.value, MediaNormalizationError)
    assert isinstance(exc_info.value, TimeoutError)
    assert proc.stopped is True
    assert "terminate" in proc.events
    assert "wait_exit" in proc.events
    assert ("kill" in proc.events) is (not graceful_exit)
    assert len(observed) == 1
    assert observed[0][1] is True
    assert not temp_root.exists()
    assert input_file.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("graceful_exit", [True, False], ids=["graceful_exit", "forced_kill"])
async def test_split_media_timeout_reaps_child_and_cleans_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graceful_exit: bool,
) -> None:
    """Timeout in split_media terminates/kills/reaps FFmpeg and cleans owned temp files."""
    temp_root = tmp_path / "temp_root"
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))

    input_file = tmp_path / "stalled.mp4"
    input_file.write_bytes(b"stalled input")

    proc = _ControlledSubprocess(graceful_exit=graceful_exit)
    observed = _install_unlink_order_spy(monkeypatch, proc)

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        pytest.raises(FFmpegTimeoutError, match="stage deadline"),
    ):
        await split_media(
            input_file,
            timeout_seconds=0.05,
            terminate_grace_seconds=0.05,
        )

    assert proc.stopped is True
    assert "terminate" in proc.events
    assert "wait_exit" in proc.events
    assert ("kill" in proc.events) is (not graceful_exit)
    assert len(observed) == 2
    assert all(stopped for _, stopped in observed)
    assert not temp_root.exists()
    assert input_file.is_file()


@pytest.mark.asyncio
async def test_normalize_media_preserves_preexisting_output_on_cancel_and_timeout(
    tmp_path: Path,
) -> None:
    """Pre-existing caller output_path is preserved on both cancellation and timeout."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"input")
    existing_out = tmp_path / "existing.wav"
    existing_out.write_bytes(b"keep me")

    # 1. Cancellation
    proc_cancel = _ControlledSubprocess(graceful_exit=True)
    with patch("asyncio.create_subprocess_exec", return_value=proc_cancel):
        task = asyncio.create_task(
            normalize_media(input_file, output_path=existing_out, timeout_seconds=5.0)
        )
        await proc_cancel.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert existing_out.read_bytes() == b"keep me"

    # 2. Timeout
    proc_timeout = _ControlledSubprocess(graceful_exit=True)
    with (
        patch("asyncio.create_subprocess_exec", return_value=proc_timeout),
        pytest.raises(FFmpegTimeoutError),
    ):
        await normalize_media(
            input_file,
            output_path=existing_out,
            timeout_seconds=0.05,
            terminate_grace_seconds=0.05,
        )
    assert existing_out.read_bytes() == b"keep me"


@pytest.mark.asyncio
async def test_split_media_audio_only_fallback_cancellation_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio-only fallback in split_media enforces deadline and cleans up on cancel/timeout."""
    temp_root = tmp_path / "temp_root"
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))

    input_file = tmp_path / "audio_only.wav"
    input_file.write_bytes(b"audio only")

    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(
        return_value=(b"", b"Output file does not contain any stream")
    )

    # 1. Timeout during fallback normalize_media
    fallback_timeout_proc = _ControlledSubprocess(graceful_exit=False)
    procs = iter([fail_proc, fallback_timeout_proc])
    with (
        patch("asyncio.create_subprocess_exec", side_effect=lambda *_a, **_kw: next(procs)),
        pytest.raises(FFmpegTimeoutError),
    ):
        await split_media(
            input_file,
            timeout_seconds=0.08,
            terminate_grace_seconds=0.04,
        )
    assert fallback_timeout_proc.stopped is True
    assert "kill" in fallback_timeout_proc.events
    assert not temp_root.exists()

    # 2. Cancellation during fallback normalize_media
    fallback_cancel_proc = _ControlledSubprocess(graceful_exit=True)
    procs2 = iter([fail_proc, fallback_cancel_proc])
    with patch("asyncio.create_subprocess_exec", side_effect=lambda *_a, **_kw: next(procs2)):
        task = asyncio.create_task(
            split_media(input_file, timeout_seconds=5.0, terminate_grace_seconds=0.05)
        )
        await fallback_cancel_proc.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert fallback_cancel_proc.stopped is True
    assert "terminate" in fallback_cancel_proc.events
    assert "wait_exit" in fallback_cancel_proc.events
    assert not temp_root.exists()


@pytest.mark.asyncio
async def test_second_cancellation_during_graceful_wait_escalates_and_reaps(
    tmp_path: Path,
) -> None:
    """A second cancellation during graceful wait still kills and reaps the child."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"input")
    output_file = tmp_path / "partial.wav"

    proc = _ControlledSubprocess(
        graceful_exit=False,
        on_start=lambda: output_file.write_bytes(b"partial"),
    )
    wait_entered = asyncio.Event()
    orig_wait = proc.wait

    async def tracked_wait() -> int:
        wait_entered.set()
        return await orig_wait()

    proc.wait = tracked_wait  # type: ignore[method-assign]

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(
            normalize_media(
                input_file,
                output_path=output_file,
                timeout_seconds=5.0,
                terminate_grace_seconds=5.0,
            )
        )
        await proc.communicate_started.wait()
        task.cancel()
        await wait_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert proc.stopped is True
    assert "terminate" in proc.events
    assert "kill" in proc.events
    assert "wait_exit" in proc.events
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_media_invalid_deadline_or_grace_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-positive or unparseable deadlines fail fast before creating temp files."""
    temp_root = tmp_path / "temp_root"
    monkeypatch.setenv(JOB_TEMP_ROOT_ENV, str(temp_root))
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"input")

    with pytest.raises(ValueError, match="timeout_seconds"):
        await normalize_media(input_file, timeout_seconds=0)
    with pytest.raises(ValueError, match="terminate_grace_seconds"):
        await split_media(input_file, terminate_grace_seconds=-1.0)
    monkeypatch.setenv(MEDIA_TIMEOUT_ENV, "not-a-number")
    with pytest.raises(ValueError, match=MEDIA_TIMEOUT_ENV):
        await normalize_media(input_file)
    assert not temp_root.exists()


@pytest.mark.asyncio
async def test_real_subprocess_cancellation_terminates_reaps_and_cleans_output(
    tmp_path: Path,
) -> None:
    """Real OS child process is terminated, reaped, and its partial file deleted on cancel."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"input")
    output_file = tmp_path / "real_partial.wav"

    real_create = asyncio.create_subprocess_exec
    spawned: list[asyncio.subprocess.Process] = []
    ready_file = tmp_path / "child_ready.flag"

    child_script = (
        "import pathlib, time, sys; "
        "pathlib.Path(sys.argv[1]).write_bytes(b'partial'); "
        "pathlib.Path(sys.argv[2]).write_text('ready'); "
        "time.sleep(30)"
    )

    async def spawn_real_child(*_cmd: str, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_create(
            sys.executable,
            "-c",
            child_script,
            str(output_file),
            str(ready_file),
            **kwargs,
        )
        spawned.append(proc)
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=spawn_real_child):
        task = asyncio.create_task(
            normalize_media(
                input_file,
                output_path=output_file,
                timeout_seconds=10.0,
                terminate_grace_seconds=1.0,
            )
        )
        while not ready_file.exists():
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(spawned) == 1
    assert spawned[0].returncode is not None
    assert not output_file.exists()
    assert input_file.is_file()
