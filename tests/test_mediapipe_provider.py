"""Unit tests for app.providers.mediapipe_vision."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.providers.base import VisionProvider
from app.providers.mediapipe_vision import (
    ALGORITHM_VERSION,
    MediaPipeVisionProvider,
    _ensure_model,
)
from app.providers.types import VisualObservation


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a 1-second synthetic MP4 video for testing."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for _ in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_mediapipe_provider_satisfies_protocol() -> None:
    """Verify MediaPipeVisionProvider satisfies the VisionProvider Protocol."""
    provider: VisionProvider = MediaPipeVisionProvider(
        face_landmarker=MagicMock(),
        pose_landmarker=MagicMock(),
    )
    assert hasattr(provider, "analyze_video")
    assert callable(provider.analyze_video)


@pytest.mark.asyncio
async def test_mediapipe_provider_invalid_video(tmp_path: Path) -> None:
    """Verify RuntimeError is raised when video file cannot be opened."""
    provider = MediaPipeVisionProvider(
        face_landmarker=MagicMock(),
        pose_landmarker=MagicMock(),
    )
    non_existent = tmp_path / "does_not_exist.mp4"
    with pytest.raises(RuntimeError, match="Failed to open video file"):
        await provider.analyze_video(non_existent)


@pytest.mark.asyncio
async def test_mediapipe_provider_with_mock_detections(synthetic_video: Path) -> None:
    """Verify observations are extracted correctly from mock landmark detections."""
    # Create mock face landmarks
    mock_face_landmark = MagicMock()
    mock_face_landmark.x = 0.5
    mock_face_landmark.y = 0.5
    mock_face_landmark.z = 0.0

    # 478 mock face landmarks
    mock_face_landmarks = [mock_face_landmark] * 478
    mock_face_result = MagicMock()
    mock_face_result.face_landmarks = [mock_face_landmarks]

    # Create mock pose landmarks (33 landmarks)
    mock_pose_landmark = MagicMock()
    mock_pose_landmark.x = 0.5
    mock_pose_landmark.y = 0.5
    mock_pose_landmark.z = 0.0
    mock_pose_landmarks = [mock_pose_landmark] * 33
    mock_pose_result = MagicMock()
    mock_pose_result.pose_landmarks = [mock_pose_landmarks]

    mock_face = MagicMock()
    mock_face.detect.return_value = mock_face_result

    mock_pose = MagicMock()
    mock_pose.detect.return_value = mock_pose_result

    provider = MediaPipeVisionProvider(
        sample_fps=2.0,
        face_landmarker=mock_face,
        pose_landmarker=mock_pose,
    )

    observations = await provider.analyze_video(synthetic_video)

    assert isinstance(observations, list)
    assert len(observations) > 0
    assert all(isinstance(obs, VisualObservation) for obs in observations)

    metrics = {obs.metric for obs in observations}
    assert "gaze_direction" in metrics
    assert "head_pitch_degrees" in metrics
    assert "head_yaw_degrees" in metrics
    assert "mouth_aspect_ratio" in metrics
    assert "shoulder_symmetry_ratio" in metrics
    assert "posture_openness" in metrics
    assert "upper_body_movement_px" in metrics

    for obs in observations:
        assert obs.algorithm_version == ALGORITHM_VERSION
        assert obs.start_ms < obs.end_ms
        assert obs.end_ms <= 1000
        assert 0.0 <= obs.confidence <= 1.0
        assert obs.speaker_label is not None
        assert obs.speaker_label.startswith("PERSON_")


@pytest.mark.asyncio
async def test_mediapipe_provider_real_models_blank_video(synthetic_video: Path) -> None:
    """Verify real MediaPipe models run against a blank video without crashing (0 detections)."""
    provider = MediaPipeVisionProvider(sample_fps=2.0)
    observations = await provider.analyze_video(synthetic_video)
    # On a blank black video, no face or pose will be detected
    assert isinstance(observations, list)
    assert len(observations) == 0


def test_ensure_model_atomic_and_size_validation(tmp_path: Path) -> None:
    """Verify incomplete or interrupted model downloads are rejected and cleaned up."""
    model_path = tmp_path / "model.task"

    def write_partial(url: str, dest: str) -> None:
        Path(dest).write_bytes(b"corrupted_partial_data")

    with patch("urllib.request.urlretrieve", side_effect=write_partial):
        with pytest.raises(RuntimeError, match="incomplete or corrupt"):
            _ensure_model(model_path, "https://example.com/model.task")

    # Target file should not exist after failure
    assert not model_path.exists()
    # Temporary file should also be cleaned up
    assert not model_path.with_suffix(".tmp").exists()


@pytest.mark.asyncio
async def test_mediapipe_provider_streaming_unknown_frame_count(synthetic_video: Path) -> None:
    """Verify provider decodes sequentially when container returns unknown frame count (<= 0)."""
    mock_face = MagicMock()
    mock_face.detect.return_value = MagicMock(face_landmarks=[])
    mock_pose = MagicMock()
    mock_pose.detect.return_value = MagicMock(pose_landmarks=[])

    provider = MediaPipeVisionProvider(
        sample_fps=2.0,
        face_landmarker=mock_face,
        pose_landmarker=mock_pose,
    )

    # Patch cv2.VideoCapture.get to return -1 for CAP_PROP_FRAME_COUNT (e.g. streaming/fragmented WebM/MP4)
    original_get = cv2.VideoCapture.get

    def mock_get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return -1.0
        return original_get(self, propId)

    with patch.object(cv2.VideoCapture, "get", mock_get):
        observations = await provider.analyze_video(synthetic_video)
        assert isinstance(observations, list)
        # Verify face and pose detection were called despite total_frames <= 0
        assert mock_face.detect.call_count > 0


@pytest.mark.asyncio
async def test_mediapipe_provider_strict_tracking_threshold(synthetic_video: Path) -> None:
    """Verify faces separated by > 0.40 distance are allocated separate tracks."""
    mock_landmark_1 = MagicMock(x=0.1, y=0.1, z=0.0)
    mock_face_1 = MagicMock(face_landmarks=[[mock_landmark_1] * 478])

    mock_landmark_2 = MagicMock(x=0.85, y=0.85, z=0.0)
    mock_face_2 = MagicMock(face_landmarks=[[mock_landmark_2] * 478])

    mock_face = MagicMock()
    # Frame 0: face at (0.1, 0.1), Frame 1: face at (0.85, 0.85)
    mock_face.detect.side_effect = [mock_face_1, mock_face_2]

    mock_pose = MagicMock()
    mock_pose.detect.return_value = MagicMock(pose_landmarks=[])

    provider = MediaPipeVisionProvider(
        sample_fps=2.0,
        face_landmarker=mock_face,
        pose_landmarker=mock_pose,
    )

    observations = await provider.analyze_video(synthetic_video)
    speaker_labels = {obs.speaker_label for obs in observations}

    # Should allocate PERSON_00 and PERSON_01, not force PERSON_00 across the screen
    assert "PERSON_00" in speaker_labels
    assert "PERSON_01" in speaker_labels


@pytest.mark.asyncio
async def test_mediapipe_provider_movement_recency_reset(synthetic_video: Path) -> None:
    """Verify displacement resets to 0.0 when pose tracking has a temporal gap."""
    def make_pose(x: float, y: float):
        landmarks = [MagicMock(x=x, y=y, z=0.0) for _ in range(33)]
        # Left/right shoulders (11, 12)
        landmarks[11] = MagicMock(x=x - 0.1, y=y, z=0.0)
        landmarks[12] = MagicMock(x=x + 0.1, y=y, z=0.0)
        # Elbows (13, 14) and wrists (15, 16)
        landmarks[13] = MagicMock(x=x - 0.15, y=y + 0.1, z=0.0)
        landmarks[14] = MagicMock(x=x + 0.15, y=y + 0.1, z=0.0)
        landmarks[15] = MagicMock(x=x - 0.2, y=y + 0.2, z=0.0)
        landmarks[16] = MagicMock(x=x + 0.2, y=y + 0.2, z=0.0)
        return MagicMock(pose_landmarks=[landmarks])

    mock_face = MagicMock()
    mock_face.detect.return_value = MagicMock(face_landmarks=[])

    mock_pose = MagicMock()
    # Frame 0: detected at (0.5, 0.5) (displacement 0.0)
    # Frame 1 & 2: tracking gap
    # Frame 3: detected at (0.9, 0.9) (gap reset -> displacement 0.0)
    # Frame 4: detected at (0.92, 0.9) (continuous -> displacement > 0.0)
    mock_pose.detect.side_effect = [
        make_pose(0.5, 0.5),
        MagicMock(pose_landmarks=[]),
        MagicMock(pose_landmarks=[]),
        make_pose(0.9, 0.9),
        make_pose(0.92, 0.9),
    ]

    provider = MediaPipeVisionProvider(
        sample_fps=30.0,
        face_landmarker=mock_face,
        pose_landmarker=mock_pose,
    )

    observations = await provider.analyze_video(synthetic_video)
    movement_obs = [obs for obs in observations if obs.metric == "upper_body_movement_px"]

    assert len(movement_obs) == 3
    # First detection displacement should be 0.0
    assert movement_obs[0].value == 0.0
    # Following a multi-frame tracking gap, displacement should reset to 0.0
    assert movement_obs[1].value == 0.0
    # Immediately consecutive frame should compute real displacement > 0.0
    assert movement_obs[2].value > 0.0

