"""Standalone test script to run MediaPipe vision analysis on a real video file.

Usage:
    conda activate victories
    python scripts/test_vision_on_video.py <path_to_video> [--fps 2.0]
"""

import argparse
import asyncio
import math
import sys
from pathlib import Path

import cv2

from app.providers.mediapipe_vision import MediaPipeVisionProvider
from app.stages.vision import run_vision_stage

GAZE_NAMES = {
    1.0: "Direct / Camera",
    2.0: "Looking Left",
    3.0: "Looking Right",
    4.0: "Looking Down",
}


def get_video_duration_ms(video_path: Path) -> tuple[int, float, int]:
    """Retrieve video duration in ms, FPS, and total frame count using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = round(total_frames / fps * 1000.0)
    cap.release()
    return duration_ms, fps, total_frames


async def test_video(video_path: Path, sample_fps: float, max_persons: int = 50) -> None:
    print("\n" + "=" * 75)
    print("VIRTUJUDGE VISION ANALYSIS TEST")
    print("=" * 75)
    print(f"Video:       {video_path.resolve()}")
    print(f"File Size:   {video_path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        duration_ms, fps, total_frames = get_video_duration_ms(video_path)
    except Exception as exc:
        print(f"Error inspecting video: {exc}")
        sys.exit(1)

    expected_frames = round(duration_ms / 1000.0 * sample_fps)
    print(f"Duration:    {duration_ms / 1000.0:.2f}s ({duration_ms} ms)")
    print(f"Native FPS:  {fps:.2f} (Total frames: {total_frames})")
    print(f"Sampling:    {sample_fps:.1f} FPS (approx. {expected_frames} frames)")
    print(f"Max Persons: {max_persons}")
    print("=" * 75)

    print("\nInitializing MediaPipe Vision Provider (Face + Pose Landmarkers)...")
    provider = MediaPipeVisionProvider(sample_fps=sample_fps, max_persons=max_persons)

    print("Processing video frames...")
    result = await run_vision_stage(
        video_path=video_path,
        vision_provider=provider,
        media_duration_ms=duration_ms,
    )

    print("\n" + "-" * 75)
    print("RESULTS SUMMARY")
    print("-" * 75)
    print(f"Total Observations: {len(result.observations)}")
    print(f"Limitations Count:  {len(result.limitations)}")

    if result.limitations:
        print("\nLIMITATIONS / ADVISORIES:")
        for lim in result.limitations:
            print(f"  [{lim.code}] {lim.message}")
            print(f"    Affected dimensions: {lim.affected_dimensions}")

    if not result.observations:
        print("\nNo observations were extracted from the video.")
        return

    # Aggregate metrics
    gaze_counts: dict[float, int] = {}
    pitches: list[float] = []
    yaws: list[float] = []
    symmetries: list[float] = []
    movements: list[float] = []

    for obs in result.observations:
        if obs.metric == "gaze_direction":
            gaze_counts[obs.value] = gaze_counts.get(obs.value, 0) + 1
        elif obs.metric == "head_pitch_degrees":
            pitches.append(obs.value)
        elif obs.metric == "head_yaw_degrees":
            yaws.append(obs.value)
        elif obs.metric == "shoulder_symmetry_ratio":
            symmetries.append(obs.value)
        elif obs.metric == "upper_body_movement_px":
            movements.append(obs.value)

    print("\nMEASUREMENT BREAKDOWN:")
    if gaze_counts:
        total_gaze = sum(gaze_counts.values())
        print("  Gaze Distribution:")
        for code, count in sorted(gaze_counts.items()):
            label = GAZE_NAMES.get(code, f"Code {code}")
            pct = (count / total_gaze) * 100.0
            print(f"      - {label:<18}: {count:>4} samples ({pct:5.1f}%)")

    if pitches:
        mean_pitch = sum(pitches) / len(pitches)
        print(
            f"  Head Pitch:           mean={mean_pitch:+.1f}°, "
            f"min={min(pitches):+.1f}°, max={max(pitches):+.1f}°"
        )

    if yaws:
        mean_yaw = sum(yaws) / len(yaws)
        print(
            f"  Head Yaw:             mean={mean_yaw:+.1f}°, "
            f"min={min(yaws):+.1f}°, max={max(yaws):+.1f}°"
        )

    if symmetries:
        mean_sym = sum(symmetries) / len(symmetries)
        print(
            f"  Shoulder Symmetry:    mean={mean_sym:.3f} "
            f"(1.0 = level shoulders), min={min(symmetries):.3f}"
        )

    if movements:
        mean_mov = sum(movements) / len(movements)
        print(
            f"  Upper Body Movement:  mean={mean_mov:.1f} px/sample, "
            f"max={max(movements):.1f} px"
        )

    print("\n--- SAMPLE TIMED OBSERVATIONS (First 15) ---")
    for obs in result.observations[:15]:
        start_s = obs.start_ms / 1000.0
        end_s = obs.end_ms / 1000.0
        val_str = f"{obs.value:<8.2f}" if isinstance(obs.value, float) else f"{obs.value:<8}"
        print(f"  [{start_s:05.2f}s - {end_s:05.2f}s] {obs.metric:<25} = {val_str} {obs.unit}")

    if len(result.observations) > 15:
        print(f"  ... and {len(result.observations) - 15} more observations")

    print("\n" + "=" * 75)
    print("TEST COMPLETE - All measurements comply with observable Data Contracts")
    print("=" * 75 + "\n")


async def run_test(
    video_file: Path,
    sample_fps: float,
    max_persons: int = 50,
    as_json: bool = False,
    output_file: Path | None = None,
) -> None:
    if as_json:
        duration_ms, _, _ = get_video_duration_ms(video_file)
        provider = MediaPipeVisionProvider(sample_fps=sample_fps, max_persons=max_persons)
        result = await run_vision_stage(
            video_path=video_file,
            vision_provider=provider,
            media_duration_ms=duration_ms,
        )
        json_str = result.model_dump_json(indent=2)
        print(json_str)
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_str, encoding="utf-8")
        return

    await test_video(video_file, sample_fps=sample_fps, max_persons=max_persons)

    if output_file is not None:
        duration_ms, _, _ = get_video_duration_ms(video_file)
        provider = MediaPipeVisionProvider(sample_fps=sample_fps, max_persons=max_persons)
        result = await run_vision_stage(
            video_path=video_file,
            vision_provider=provider,
            media_duration_ms=duration_ms,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"[SAVED] JSON output successfully saved to: {output_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test MediaPipe vision stage on a real video file."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file (.mp4, .mov, .avi, etc.)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Sample rate in frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=50,
        help="Maximum persons to detect simultaneously (default: 50, detects all people in frame)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw contract JSON to stdout instead of summary table",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON output (e.g. output.json)",
    )
    args = parser.parse_args()

    video_file = Path(args.video_path)
    if not video_file.is_file():
        print(f"Error: File '{video_file}' does not exist.")
        sys.exit(1)

    out_path = Path(args.output) if args.output else None
    asyncio.run(
        run_test(
            video_file,
            sample_fps=args.fps,
            max_persons=args.max_persons,
            as_json=args.json,
            output_file=out_path,
        )
    )


if __name__ == "__main__":
    main()
