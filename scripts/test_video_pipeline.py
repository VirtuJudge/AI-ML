"""Test end-to-end video pipeline:

1. Split input video into 16kHz mono WAV (for audio) and video stream (for vision).
2. Transcribe audio with Whisper (GroqCloud).
3. Diarize audio with PyAnnote (or graceful fallback).
4. Analyze video with VisionProvider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env variables before importing application modules
_env_path = Path(".env")
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip())

from app.providers.fake_speech import FakeDiarizationProvider  # noqa: E402
from app.providers.fake_vision import FakeVisionProvider  # noqa: E402
from app.providers.groq_speech import GroqSpeechProvider  # noqa: E402
from app.stages.media import split_media  # noqa: E402
from app.stages.speech import validate_word_timestamps  # noqa: E402


async def run_pipeline_on_video(
    video_path: Path,
    num_speakers: int | None = None,
) -> None:
    print("\n" + "=" * 70)
    print(f"🎬 INPUT VIDEO: {video_path.name}")
    print(f"📁 Full Path:   {video_path.resolve()}")
    print(f"📦 File Size:   {video_path.stat().st_size / (1024 * 1024):.2f} MB")
    if num_speakers:
        print(f"🎯 Expected Speakers Hint: {num_speakers}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Split video into audio (WAV) and video for vision
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] ✂️  Splitting video into Audio and Vision streams via FFmpeg...")
    split_result = await split_media(video_path)
    print(f"  ✓ Audio stream (16kHz mono WAV): {split_result.audio_path}")
    print(f"  ✓ Video stream (for Vision):     {split_result.video_path or 'Audio-only input'}")
    print(
        f"  ✓ Media Duration:               "
        f"{split_result.duration_s:.2f}s ({split_result.duration_ms}ms)"
    )

    # -------------------------------------------------------------------------
    # Step 2: Transcribe audio using Whisper (GroqCloud API)
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] 🎙️  Transcribing audio via Whisper Large V3 Turbo (GroqCloud)...")
    speech_provider = GroqSpeechProvider()
    transcription = await speech_provider.transcribe(split_result.audio_path)

    print(f"  ✓ Language Detected: {transcription.language}")
    print(f"  ✓ Segments count:    {len(transcription.segments)}")
    total_words = sum(len(s.words) for s in transcription.segments)
    print(f"  ✓ Total Words:       {total_words}")
    print("\n--- 📜 TRANSCRIPT TEXT ---")
    print(transcription.full_text or "(No spoken speech detected)")

    # Validate word timestamps
    if split_result.duration_ms > 0:
        ts_limitations = validate_word_timestamps(
            transcription, split_result.duration_ms, strict=False
        )
        if ts_limitations:
            for lim in ts_limitations:
                print(f"  ⚠️  Timestamp notice: {lim.message}")

    # -------------------------------------------------------------------------
    # Step 3: Diarize speakers (PyAnnote or Fallback)
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] 👥 Running speaker diarization...")
    try:
        from app.providers.pyannote_diarization import PyannoteDiarizationProvider

        diar_provider = PyannoteDiarizationProvider(
            min_speakers=num_speakers,
            max_speakers=num_speakers,
        )
        print("  ✓ Using PyAnnote Community-1 / 3.1 (local neural diarization)")
    except Exception as exc:
        err_msg = str(exc)
        if "segmentation-3.0" in err_msg or "GatedRepoError" in type(exc).__name__:
            print(
                "  ⚠️  PyAnnote gated model access needed: accept terms at "
                "https://hf.co/pyannote/segmentation-3.0 and https://hf.co/pyannote/speaker-diarization-3.1"
            )
        else:
            print(f"  ⚠️  PyAnnote not initialized ({type(exc).__name__}); using fallback")
        diar_provider = FakeDiarizationProvider()

    diar_result = await diar_provider.diarize(split_result.audio_path)
    print(f"  ✓ Speaker labels detected: {diar_result.speaker_labels}")
    print("--- 🗣️  SPEAKER TURNS ---")
    for turn in diar_result.segments:
        start_s = turn.start_ms / 1000.0
        end_s = turn.end_ms / 1000.0
        print(f"  [{start_s:05.2f}s - {end_s:05.2f}s] {turn.speaker_label}")

    # -------------------------------------------------------------------------
    # Step 4: Vision stream analysis
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] 👁️  Analyzing visual stream for gaze and body language...")
    vision_provider = FakeVisionProvider()
    target_video = split_result.video_path or video_path
    visual_observations = await vision_provider.analyze_video(target_video)
    print(f"  ✓ Observations extracted: {len(visual_observations)}")
    for obs in visual_observations:
        gaze = obs.gaze_direction
        posture = obs.posture
        print(
            f"  [{obs.timestamp_ms / 1000.0:05.2f}s] "
            f"Gaze: {gaze:<8} | Posture: {posture:<8} (Confidence: {obs.confidence:.2f})"
        )

    print("\n" + "=" * 70)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: conda run -n vic-AI python scripts/test_video_pipeline.py "
            "<path_to_video> [expected_num_speakers]"
        )
        sys.exit(1)

    video_input = Path(sys.argv[1])
    if not video_input.is_file():
        print(f"Error: File '{video_input}' does not exist.")
        sys.exit(1)

    expected_speakers = None
    if len(sys.argv) >= 3:
        try:
            expected_speakers = int(sys.argv[2])
        except ValueError:
            expected_speakers = None

    asyncio.run(run_pipeline_on_video(video_input, num_speakers=expected_speakers))
