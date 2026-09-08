"""Unit tests for app.providers.mediapipe_vision."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.providers.base import VisionProvider
from app.providers.mediapipe_vision import (
    ALGORITHM_VERSION,
    MediaPipeVisionProvider,
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
    assert "upper_body_movement_px" in metrics

    for obs in observations:
        assert obs.algorithm_version == ALGORITHM_VERSION
        assert obs.start_ms <= obs.end_ms
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
