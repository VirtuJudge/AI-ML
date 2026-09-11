"""Unit tests for the vision measurement stage (app.stages.vision)."""

from pathlib import Path

import pytest

from app.providers.fake_vision import FakeVisionProvider
from app.providers.types import VisualObservation
from app.stages.vision import (
    VisionStageResult,
    check_coverage_limitations,
    run_vision_stage,
    validate_visual_observations,
)


class FailingVisionProvider:
    """Provider that simulates an unexpected failure/crash."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        raise RuntimeError("MediaPipe crash simulation")


class EmptyVisionProvider:
    """Provider that returns an empty observation list."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        return []


class FaceOnlyVisionProvider:
    """Provider returning only face-related observations."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        return [
            VisualObservation(
                start_ms=0,
                end_ms=1000,
                metric="gaze_direction",
                value=1.0,
                unit="categorical_index",
                algorithm_version="test/1.0",
            )
        ]


class BodyOnlyVisionProvider:
    """Provider returning only body-related observations."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        return [
            VisualObservation(
                start_ms=0,
                end_ms=1000,
                metric="shoulder_symmetry_ratio",
                value=0.95,
                unit="ratio",
                algorithm_version="test/1.0",
            )
        ]


class OutOfBoundsVisionProvider:
    """Provider returning observations with timestamps exceeding duration or inverted."""

    async def analyze_video(self, video_path: Path) -> list[VisualObservation]:
        return [
            VisualObservation(
                start_ms=0,
                end_ms=25000,
                metric="gaze_direction",
                value=1.0,
                unit="categorical_index",
                algorithm_version="test/1.0",
            ),
            VisualObservation(
                start_ms=3000,
                end_ms=2000,
                metric="shoulder_symmetry_ratio",
                value=0.92,
                unit="ratio",
                algorithm_version="test/1.0",
            ),
        ]


@pytest.fixture
def dummy_video_file(tmp_path: Path) -> Path:
    """Create a temporary dummy video file."""
    video = tmp_path / "presentation.mp4"
    video.write_bytes(b"\x00\x00\x00\x1cftypisom")
    return video


@pytest.mark.asyncio
async def test_vision_stage_happy_path(dummy_video_file: Path) -> None:
    """Verify happy path execution with FakeVisionProvider."""
    provider = FakeVisionProvider(duration_ms=15000)
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=15000)

    assert isinstance(result, VisionStageResult)
    assert len(result.observations) == 9
    assert len(result.limitations) == 0
    assert result.metadata["observation_count"] == 9
    assert result.metadata["provider"] == "FakeVisionProvider"
    assert result.metadata["media_duration_ms"] == 15000


@pytest.mark.asyncio
async def test_vision_stage_timestamps_within_duration(dummy_video_file: Path) -> None:
    """Verify that all observations respect media duration bounds."""
    duration_ms = 12000
    provider = FakeVisionProvider(duration_ms=duration_ms)
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=duration_ms)

    assert len(result.observations) > 0
    for obs in result.observations:
        assert obs.start_ms >= 0
        assert obs.end_ms <= duration_ms
        assert obs.start_ms <= obs.end_ms


@pytest.mark.asyncio
async def test_vision_stage_observations_have_required_fields(dummy_video_file: Path) -> None:
    """Verify all visual observations satisfy contract fields."""
    provider = FakeVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=15000)

    for obs in result.observations:
        assert isinstance(obs.start_ms, int)
        assert isinstance(obs.end_ms, int)
        assert isinstance(obs.metric, str) and len(obs.metric) > 0
        assert isinstance(obs.value, float)
        assert isinstance(obs.unit, str) and len(obs.unit) > 0
        assert 0.0 <= obs.confidence <= 1.0
        assert isinstance(obs.algorithm_version, str) and len(obs.algorithm_version) > 0


@pytest.mark.asyncio
async def test_vision_stage_no_video_stream() -> None:
    """Verify graceful handling when video_path is None (audio-only input)."""
    provider = FakeVisionProvider()
    result = await run_vision_stage(None, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "no_video_stream"
    assert lim.scope == "vision"
    assert "visual_delivery" in lim.affected_dimensions


@pytest.mark.asyncio
async def test_vision_stage_missing_video_file(tmp_path: Path) -> None:
    """Verify graceful handling when video file path does not exist on disk."""
    non_existent = tmp_path / "missing_video.mp4"
    provider = FakeVisionProvider()
    result = await run_vision_stage(non_existent, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "video_file_missing"
    assert lim.scope == "vision"


@pytest.mark.asyncio
async def test_vision_stage_provider_failure(dummy_video_file: Path) -> None:
    """Verify provider exception is caught and converted to a safe limitation."""
    provider = FailingVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "vision_analysis_failed"
    assert lim.scope == "vision"


@pytest.mark.asyncio
async def test_vision_stage_empty_observations(dummy_video_file: Path) -> None:
    """Verify empty observations list produces no_visual_observations limitation."""
    provider = EmptyVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=10000)

    assert len(result.observations) == 0
    assert len(result.limitations) == 1
    lim = result.limitations[0]
    assert lim.code == "no_visual_observations"
    assert lim.scope == "vision"


@pytest.mark.asyncio
async def test_vision_stage_no_face_detected(dummy_video_file: Path) -> None:
    """Verify limitation emitted when only body observations are detected."""
    provider = BodyOnlyVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=10000)

    lim_codes = [lim.code for lim in result.limitations]
    assert "no_face_detected" in lim_codes
    assert "no_body_detected" not in lim_codes
    face_lim = next(lim for lim in result.limitations if lim.code == "no_face_detected")
    assert "eye_contact" in face_lim.affected_dimensions


@pytest.mark.asyncio
async def test_vision_stage_no_body_detected(dummy_video_file: Path) -> None:
    """Verify limitation emitted when only face observations are detected."""
    provider = FaceOnlyVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=10000)

    lim_codes = [lim.code for lim in result.limitations]
    assert "no_body_detected" in lim_codes
    assert "no_face_detected" not in lim_codes
    body_lim = next(lim for lim in result.limitations if lim.code == "no_body_detected")
    assert "body_language" in body_lim.affected_dimensions


@pytest.mark.asyncio
async def test_vision_stage_timestamps_clamped(dummy_video_file: Path) -> None:
    """Verify observations exceeding media duration are clamped and report limitation."""
    provider = OutOfBoundsVisionProvider()
    media_duration_ms = 10000
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=media_duration_ms)

    lim_codes = [lim.code for lim in result.limitations]
    assert "vision_observation_timestamp_clamped" in lim_codes

    for obs in result.observations:
        assert obs.start_ms >= 0
        assert obs.end_ms <= media_duration_ms
        assert obs.start_ms <= obs.end_ms


@pytest.mark.asyncio
async def test_vision_stage_observable_language(dummy_video_file: Path) -> None:
    """Verify no emotion/confidence/anxiety/personality claims in observations or limitations."""
    prohibited_terms = {
        "emotion",
        "emotional",
        "happy",
        "sad",
        "angry",
        "nervous",
        "nervousness",
        "anxiety",
        "anxious",
        "honesty",
        "honest",
        "deceptive",
        "personality",
        "mental state",
        "stress",
    }

    provider = FakeVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=15000)

    for obs in result.observations:
        metric_lower = obs.metric.lower()
        for term in prohibited_terms:
            assert term not in metric_lower, (
                f"Prohibited term '{term}' found in metric '{obs.metric}'"
            )

    for lim in result.limitations:
        msg_lower = lim.message.lower()
        for term in prohibited_terms:
            assert term not in msg_lower, (
                f"Prohibited term '{term}' found in limitation '{lim.message}'"
            )


@pytest.mark.asyncio
async def test_vision_stage_no_private_frames_in_metadata(dummy_video_file: Path) -> None:
    """Verify metadata contains only safe operational keys, with no image frames or pixels."""
    provider = FakeVisionProvider()
    result = await run_vision_stage(dummy_video_file, provider, media_duration_ms=15000)

    prohibited_keys = {"frame", "frames", "pixels", "image", "image_data", "raw_bytes"}
    for key in result.metadata:
        assert key.lower() not in prohibited_keys
        # Value must not be raw bytes
        assert not isinstance(result.metadata[key], (bytes, bytearray))


def test_validate_visual_observations_inversion() -> None:
    """Unit test validate_visual_observations handles inverted timestamps and out-of-bounds start."""
    obs = [
        VisualObservation(
            start_ms=5000,
            end_ms=2000,
            metric="gaze_direction",
            value=1.0,
            unit="categorical_index",
        ),
        VisualObservation(
            start_ms=12000,
            end_ms=15000,
            metric="shoulder_symmetry_ratio",
            value=0.95,
            unit="ratio",
        ),
    ]
    validated, limitations = validate_visual_observations(
        obs, media_duration_ms=10000, source_artifact_id="video_art_456"
    )

    # Inverted 5000->2000 is corrected; out-of-bounds 12000ms is discarded
    assert len(validated) == 1
    assert validated[0].start_ms == 2000
    assert validated[0].end_ms == 5000
    assert validated[0].source_artifact_id == "video_art_456"
    assert len(limitations) == 1
    assert limitations[0].code == "vision_observation_timestamp_clamped"


def test_check_coverage_limitations_both_missing() -> None:
    """Unit test check_coverage_limitations when neither face nor body metrics exist."""
    obs = [
        VisualObservation(
            start_ms=0,
            end_ms=1000,
            metric="unknown_custom_metric",
            value=0.5,
            unit="ratio",
        )
    ]
    limitations = check_coverage_limitations(obs, media_duration_ms=5000)
    codes = [lim.code for lim in limitations]
    assert "no_face_detected" in codes
    assert "no_body_detected" in codes
