"""Test end-to-end video pipeline:

1. Split input video into 16kHz mono WAV (for audio) and video stream (for vision).
2. Transcribe audio with Whisper (GroqCloud).
3. Diarize audio with PyAnnote (or graceful fallback).
4. Analyze video with MediaPipe VisionProvider.
5. Analyze acoustic prosody with LibrosaAudioProvider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load .env variables before importing application modules
_env_path = Path(".env")
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8", errors="replace").splitlines():
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
    print(f"[INPUT VIDEO] {video_path.name}")
    print(f"  Path:       {video_path.resolve()}")
    print(f"  Size:       {video_path.stat().st_size / (1024 * 1024):.2f} MB")
    if num_speakers:
        print(f"  Expected Speakers: {num_speakers}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Split video into audio (WAV) and video for vision
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Splitting video into Audio and Vision streams via FFmpeg...")
    split_result = await split_media(video_path)
    print(f"  [OK] Audio stream (16kHz mono WAV): {split_result.audio_path}")
    print(f"  [OK] Video stream (for Vision):     {split_result.video_path or 'Audio-only input'}")
    print(
        f"  [OK] Media Duration:               "
        f"{split_result.duration_s:.2f}s ({split_result.duration_ms}ms)"
    )

    # -------------------------------------------------------------------------
    # Step 2: Transcribe audio using Whisper (GroqCloud API)
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Transcribing audio via Whisper Large V3 Turbo (GroqCloud)...")
    try:
        speech_provider = GroqSpeechProvider()
        transcription = await speech_provider.transcribe(split_result.audio_path)
    except Exception as exc:
        print(f"  [WARN] Groq transcription unavailable ({exc}); using fallback provider")
        from app.providers.fake_speech import FakeSpeechProvider

        speech_provider = FakeSpeechProvider()
        transcription = await speech_provider.transcribe(split_result.audio_path)

    print(f"  [OK] Language Detected: {transcription.language}")
    print(f"  [OK] Segments count:    {len(transcription.segments)}")
    total_words = sum(len(s.words) for s in transcription.segments)
    print(f"  [OK] Total Words:       {total_words}")
    print("\n--- TRANSCRIPT TEXT ---")
    print(transcription.full_text or "(No spoken speech detected)")

    # Validate word timestamps
    if split_result.duration_ms > 0:
        ts_limitations = validate_word_timestamps(
            transcription, split_result.duration_ms, strict=False
        )
        if ts_limitations:
            for lim in ts_limitations:
                print(f"  [WARN] Timestamp notice: {lim.message}")

    # -------------------------------------------------------------------------
    # Step 3: Diarize speakers (PyAnnote or Fallback)
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Running speaker diarization...")
    try:
        from app.providers.pyannote_diarization import PyannoteDiarizationProvider

        diar_provider = PyannoteDiarizationProvider(
            min_speakers=num_speakers,
            max_speakers=num_speakers,
        )
        print("  [OK] Using PyAnnote Community-1 / 3.1 (local neural diarization)")
        diar_result = await diar_provider.diarize(split_result.audio_path)
    except Exception as exc:
        err_msg = str(exc)
        if "segmentation-3.0" in err_msg or "GatedRepoError" in type(exc).__name__:
            print(
                "  [WARN] PyAnnote gated model access needed: accept terms at "
                "https://hf.co/pyannote/segmentation-3.0 and https://hf.co/pyannote/speaker-diarization-3.1"
            )
        else:
            print(
                f"  [WARN] PyAnnote diarization failed ({type(exc).__name__}: {exc}); "
                "using fallback provider"
            )
        diar_provider = FakeDiarizationProvider()
        diar_result = await diar_provider.diarize(split_result.audio_path)

    print(f"  [OK] Speaker labels detected: {diar_result.speaker_labels}")
    print("--- SPEAKER TURNS ---")
    for turn in diar_result.segments:
        start_s = turn.start_ms / 1000.0
        end_s = turn.end_ms / 1000.0
        print(f"  [{start_s:05.2f}s - {end_s:05.2f}s] {turn.speaker_label}")

    # -------------------------------------------------------------------------
    # Step 4: Vision stream analysis (MediaPipe)
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Analyzing visual stream for gaze and body language...")
    try:
        from app.providers.mediapipe_vision import MediaPipeVisionProvider

        vision_provider = MediaPipeVisionProvider()
        print("  [OK] Using MediaPipe (FaceLandmarker + PoseLandmarker)")
    except Exception as exc:
        print(f"  [WARN] MediaPipe not initialized ({exc}); using fake provider")
        vision_provider = FakeVisionProvider()

    target_video = split_result.video_path or video_path
    from app.stages.vision import run_vision_stage

    vision_result = await run_vision_stage(
        target_video,
        vision_provider,
        media_duration_ms=split_result.duration_ms,
    )
    print(f"  [OK] Vision observations extracted: {len(vision_result.observations)}")
    unique_persons = sorted(
        list({obs.speaker_label for obs in vision_result.observations if obs.speaker_label})
    )
    print(f"  [OK] Distinct Persons detected ({len(unique_persons)}): {unique_persons}")
    if vision_result.limitations:
        for lim in vision_result.limitations:
            print(f"  [WARN] Vision limitation [{lim.code}]: {lim.message}")

    for obs in vision_result.observations[:15]:
        pid_str = f"[{obs.speaker_label}] " if obs.speaker_label else ""
        print(
            f"  [{obs.start_ms / 1000.0:05.2f}s - {obs.end_ms / 1000.0:05.2f}s] "
            f"{pid_str}Metric: {obs.metric:<25} | Value: {obs.value:<8} {obs.unit} "
            f"(Confidence: {obs.confidence:.2f})"
        )
    if len(vision_result.observations) > 15:
        print(f"  ... and {len(vision_result.observations) - 15} more observations")

    # -------------------------------------------------------------------------
    # Step 5: Audio acoustic & prosody analysis (Librosa)
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Analyzing acoustic features (speaking rate, pitch, pauses)...")
    try:
        from app.providers.librosa_audio import LibrosaAudioProvider

        audio_provider = LibrosaAudioProvider()
        print("  [OK] Using Librosa (onset detection + pYIN pitch + silence detection)")
    except Exception as exc:
        print(f"  [WARN] Librosa not initialized ({exc}); using fake provider")
        from app.providers.fake_audio import FakeAudioMetricsProvider

        audio_provider = FakeAudioMetricsProvider()

    from app.stages.audio import run_audio_stage

    audio_result = await run_audio_stage(
        split_result.audio_path,
        audio_provider,
        media_duration_ms=split_result.duration_ms,
    )
    print(f"  [OK] Acoustic observations extracted: {len(audio_result.observations)}")
    if audio_result.limitations:
        for lim in audio_result.limitations:
            print(f"  [WARN] Audio limitation [{lim.code}]: {lim.message}")

    for obs in audio_result.observations[:15]:
        print(
            f"  [{obs.start_ms / 1000.0:05.2f}s - {obs.end_ms / 1000.0:05.2f}s] "
            f"Metric: {obs.metric:<25} | Value: {obs.value:<8} {obs.unit} "
            f"(Confidence: {obs.confidence:.2f})"
        )
    if len(audio_result.observations) > 15:
        print(f"  ... and {len(audio_result.observations) - 15} more observations")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/test_video_pipeline.py "
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
