"""MediaPipe computer vision provider adapter.

Extracts timed visual measurements (gaze direction, head pitch/yaw, shoulder symmetry,
and upper-body movement) using MediaPipe Face Landmarker and Pose Landmarker.
Does not make psychological, emotional, or personality claims.
"""

import asyncio
import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.providers.types import VisualObservation

logger = logging.getLogger(__name__)

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

ALGORITHM_VERSION = f"mediapipe/{mp.__version__}"


MIN_MODEL_BYTES = 500_000  # Task models are 5-10MB; anything under 500KB is incomplete


def _ensure_model(model_path: Path, url: str) -> Path:
    """Ensure that the required task model file exists and is valid, downloading atomically if necessary."""
    if model_path.is_file() and model_path.stat().st_size >= MIN_MODEL_BYTES:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    logger.info("Downloading MediaPipe model from %s to %s", url, model_path)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if not tmp_path.is_file() or tmp_path.stat().st_size < MIN_MODEL_BYTES:
            size = tmp_path.stat().st_size if tmp_path.is_file() else 0
            raise RuntimeError(f"Downloaded model is incomplete or corrupt ({size} bytes)")
        os.replace(tmp_path, model_path)
    except Exception as exc:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download MediaPipe model from {url} to {model_path}: {exc}"
        ) from exc
    return model_path


class MediaPipeVisionProvider:
    """Extracts timed visual measurements from video files using MediaPipe."""

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        sample_fps: float = 2.0,
        max_persons: int | None = None,
        face_landmarker: Any = None,
        pose_landmarker: Any = None,
    ) -> None:
        """Initialize MediaPipeVisionProvider.

        Args:
            model_dir: Directory to cache downloaded MediaPipe task models.
            sample_fps: Number of video frames to sample per second.
            max_persons: Maximum number of persons to detect simultaneously (default: 50).
            face_landmarker: Optional pre-configured FaceLandmarker instance (useful for testing).
            pose_landmarker: Optional pre-configured PoseLandmarker instance (useful for testing).
        """
        self.sample_fps = max(0.1, float(sample_fps))
        resolved_max = (
            int(max_persons)
            if max_persons is not None
            else int(os.getenv("MEDIAPIPE_MAX_PERSONS", "50"))
        )
        self.max_persons = max(1, resolved_max)

        default_dir = str(Path.home() / ".cache" / "virtujudge" / "models")
        resolved_dir = (
            Path(model_dir)
            if model_dir is not None
            else Path(os.getenv("MEDIAPIPE_MODEL_DIR", default_dir))
        )
        self.model_dir = resolved_dir

        self._face_landmarker = face_landmarker
        self._pose_landmarker = pose_landmarker

    def _get_face_landmarker(self) -> Any:
        if self._face_landmarker is not None:
            return self._face_landmarker

        model_path = _ensure_model(self.model_dir / "face_landmarker.task", FACE_MODEL_URL)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_persons,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._face_landmarker

    def _get_pose_landmarker(self) -> Any:
        if self._pose_landmarker is not None:
            return self._pose_landmarker

        model_path = _ensure_model(self.model_dir / "pose_landmarker_lite.task", POSE_MODEL_URL)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.max_persons,
        )
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._pose_landmarker

    def _process_video_sync(
        self,
        video_path: Path,
        media_duration_ms: int | None = None,
        source_artifact_id: str = "",
    ) -> list[VisualObservation]:
        """Synchronously sample frames and extract visual observations per tracked person."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frames = max(1, round(fps / self.sample_fps))
        interval_ms = round(1000.0 / self.sample_fps)

        if media_duration_ms is None and total_frames > 0 and fps > 0:
            media_duration_ms = round(total_frames / fps * 1000.0)

        artifact_id = source_artifact_id or video_path.name

        face_landmarker = self._get_face_landmarker()
        pose_landmarker = self._get_pose_landmarker()

        observations: list[VisualObservation] = []

        # Multi-person tracking state across frames
        tracked_persons: dict[str, tuple[float, float]] = {}
        prev_upper_body_midpoints: dict[str, tuple[float, float, int]] = {}

        frame_idx = 0
        try:
            while True:
                if total_frames > 0 and frame_idx >= total_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_h, frame_w = frame.shape[:2]
                frame_ms = round(frame_idx / fps * 1000.0)
                if media_duration_ms is not None and frame_ms >= media_duration_ms:
                    break

                start_ms = frame_ms
                end_ms = frame_ms + interval_ms
                if media_duration_ms is not None and media_duration_ms > start_ms:
                    end_ms = min(end_ms, media_duration_ms)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detected_face_persons: list[tuple[str, Any]] = []

                # 1. Face landmarks & multi-person tracking
                try:
                    face_result = face_landmarker.detect(mp_image)
                    if face_result.face_landmarks:
                        for fl in face_result.face_landmarks:
                            fc_x, fc_y = fl[1].x, fl[1].y

                            # Match to existing tracked person or allocate new ID
                            assigned_pids = {pid for pid, _ in detected_face_persons}
                            best_pid = None
                            best_dist = 0.40
                            for pid, (px, py) in tracked_persons.items():
                                if pid in assigned_pids:
                                    continue
                                dist = math.sqrt((fc_x - px) ** 2 + (fc_y - py) ** 2)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pid = pid

                            if best_pid is None:
                                # No existing unassigned track is close enough (dist < 0.40)
                                if len(tracked_persons) < self.max_persons:
                                    best_pid = f"PERSON_{len(tracked_persons):02d}"
                                    tracked_persons[best_pid] = (fc_x, fc_y)
                                else:
                                    # All slots full: fallback to closest unassigned track if any exist
                                    unassigned = [
                                        p for p in tracked_persons if p not in assigned_pids
                                    ]
                                    if unassigned:
                                        best_pid = min(
                                            unassigned,
                                            key=lambda p: math.sqrt(
                                                (fc_x - tracked_persons[p][0]) ** 2
                                                + (fc_y - tracked_persons[p][1]) ** 2
                                            ),
                                        )
                                    else:
                                        continue

                            # Smoothly update centroid with exponential moving average
                            old_x, old_y = tracked_persons[best_pid]
                            tracked_persons[best_pid] = (
                                0.7 * old_x + 0.3 * fc_x,
                                0.7 * old_y + 0.3 * fc_y,
                            )
                            detected_face_persons.append((best_pid, fl))

                            # Head pitch from forehead (10) to chin (152)
                            dy_face = fl[152].y - fl[10].y
                            dz_face = fl[152].z - fl[10].z
                            pitch_deg = math.degrees(math.atan2(dz_face, dy_face))

                            # Head yaw from left cheek (234) to right cheek (454)
                            dx_face = fl[454].x - fl[234].x
                            dz_yaw = fl[454].z - fl[234].z
                            yaw_deg = math.degrees(math.atan2(dz_yaw, dx_face))

                            # Gaze direction categorical index: 1=camera, 2=left, 3=right, 4=down
                            if pitch_deg > 20.0:
                                gaze_idx = 4.0
                            elif yaw_deg > 15.0:
                                gaze_idx = 3.0
                            elif yaw_deg < -15.0:
                                gaze_idx = 2.0
                            else:
                                gaze_idx = 1.0

                            # Mouth aspect ratio (lip movement metric)
                            # Upper lip (13) vs Lower lip (14)
                            # Left corner (61) vs Right corner (291)
                            dy_lip = math.sqrt(
                                (fl[14].x - fl[13].x) ** 2 + (fl[14].y - fl[13].y) ** 2
                            )
                            dx_lip = math.sqrt(
                                (fl[291].x - fl[61].x) ** 2 + (fl[291].y - fl[61].y) ** 2
                            )
                            mar = dy_lip / dx_lip if dx_lip > 1e-4 else 0.0

                            observations.extend(
                                [
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="gaze_direction",
                                        value=gaze_idx,
                                        unit="categorical_index",
                                        confidence=0.90,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="head_pitch_degrees",
                                        value=round(pitch_deg, 2),
                                        unit="degrees",
                                        confidence=0.90,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="head_yaw_degrees",
                                        value=round(yaw_deg, 2),
                                        unit="degrees",
                                        confidence=0.90,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="mouth_aspect_ratio",
                                        value=round(mar, 4),
                                        unit="ratio",
                                        confidence=0.90,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                ]
                            )
                except Exception:
                    logger.debug(
                        "Face landmark detection failed at frame %d", frame_idx, exc_info=True
                    )

                # 2. Pose landmarks & association with tracked persons
                try:
                    pose_result = pose_landmarker.detect(mp_image)
                    if pose_result.pose_landmarks:
                        assigned_pose_pids: set[str] = set()
                        for pl in pose_result.pose_landmarks:
                            ls, rs = pl[11], pl[12]
                            dy_shoulder = abs(ls.y - rs.y)
                            dx_shoulder = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)

                            sym_ratio = (
                                max(0.0, min(1.0, 1.0 - (dy_shoulder / dx_shoulder)))
                                if dx_shoulder > 1e-4
                                else 1.0
                            )

                            # Posture openness: ratio of arm span (wrists or elbows) to shoulder width
                            # MediaPipe Pose: 13=left elbow, 14=right elbow, 15=left wrist, 16=right wrist
                            le, re = pl[13], pl[14]
                            lw, rw = pl[15], pl[16]
                            wrist_span = math.sqrt((lw.x - rw.x) ** 2 + (lw.y - rw.y) ** 2)
                            elbow_span = math.sqrt((le.x - re.x) ** 2 + (le.y - re.y) ** 2)
                            arm_span = max(wrist_span, elbow_span)
                            posture_openness = (
                                round(arm_span / dx_shoulder, 2)
                                if dx_shoulder > 1e-4
                                else 1.0
                            )

                            shoulder_cx = (ls.x + rs.x) / 2.0
                            mid_x = shoulder_cx * frame_w
                            mid_y = (ls.y + rs.y) / 2.0 * frame_h

                            # Match pose to the nearest detected face person or existing tracked person
                            best_pid = None
                            best_xdiff = 0.40
                            for pid, fl in detected_face_persons:
                                if pid in assigned_pose_pids:
                                    continue
                                xdiff = abs(fl[1].x - shoulder_cx)
                                if xdiff < best_xdiff:
                                    best_xdiff = xdiff
                                    best_pid = pid

                            if best_pid is None:
                                for pid, (px, py) in tracked_persons.items():
                                    if pid in assigned_pose_pids:
                                        continue
                                    xdiff = abs(px - shoulder_cx)
                                    if xdiff < best_xdiff:
                                        best_xdiff = xdiff
                                        best_pid = pid

                            if best_pid is None:
                                if len(tracked_persons) < self.max_persons:
                                    best_pid = f"PERSON_{len(tracked_persons):02d}"
                                    tracked_persons[best_pid] = (
                                        shoulder_cx,
                                        (ls.y + rs.y) / 2.0,
                                    )
                                elif tracked_persons:
                                    unassigned_tracked = [
                                        p for p in tracked_persons if p not in assigned_pose_pids
                                    ]
                                    if unassigned_tracked:
                                        best_pid = min(
                                            unassigned_tracked,
                                            key=lambda p: abs(tracked_persons[p][0] - shoulder_cx),
                                        )
                                    else:
                                        best_pid = "PERSON_00"
                                else:
                                    best_pid = "PERSON_00"
                                    tracked_persons[best_pid] = (
                                        shoulder_cx,
                                        (ls.y + rs.y) / 2.0,
                                    )

                            assigned_pose_pids.add(best_pid)

                            prev_data = prev_upper_body_midpoints.get(best_pid)
                            if prev_data is not None:
                                prev_mid_x, prev_mid_y, prev_frame_idx = prev_data
                                if frame_idx - prev_frame_idx <= step_frames * 1.5:
                                    displacement = math.sqrt(
                                        (mid_x - prev_mid_x) ** 2 + (mid_y - prev_mid_y) ** 2
                                    )
                                else:
                                    displacement = 0.0
                            else:
                                displacement = 0.0
                            prev_upper_body_midpoints[best_pid] = (mid_x, mid_y, frame_idx)

                            observations.extend(
                                [
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="shoulder_symmetry_ratio",
                                        value=round(sym_ratio, 4),
                                        unit="ratio",
                                        confidence=0.88,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="posture_openness",
                                        value=float(posture_openness),
                                        unit="ratio",
                                        confidence=0.88,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                    VisualObservation(
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                        metric="upper_body_movement_px",
                                        value=round(displacement, 2),
                                        unit="pixels",
                                        confidence=0.88,
                                        source_artifact_id=artifact_id,
                                        speaker_label=best_pid,
                                        algorithm_version=ALGORITHM_VERSION,
                                    ),
                                ]
                            )
                except Exception:
                    logger.debug(
                        "Pose landmark detection failed at frame %d", frame_idx, exc_info=True
                    )

                # Explicitly discard frames to safeguard memory and privacy
                del frame
                del rgb_frame
                del mp_image

                # Fast-forward next (step_frames - 1) frames without full decoding
                for _ in range(step_frames - 1):
                    frame_idx += 1
                    if total_frames > 0 and frame_idx >= total_frames:
                        break
                    if not cap.grab():
                        break

                frame_idx += 1
        finally:
            cap.release()

        return observations

    async def analyze_video(
        self,
        video_path: Path,
        *,
        media_duration_ms: int | None = None,
        source_artifact_id: str = "",
    ) -> list[VisualObservation]:
        """Asynchronously analyze video file and extract visual observations."""
        return await asyncio.to_thread(
            self._process_video_sync,
            Path(video_path),
            media_duration_ms,
            source_artifact_id,
        )


__all__ = [
    "ALGORITHM_VERSION",
    "FACE_MODEL_URL",
    "POSE_MODEL_URL",
    "MediaPipeVisionProvider",
]
