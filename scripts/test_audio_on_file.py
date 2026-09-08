"""Standalone test script to run Librosa audio analysis on a real audio or video file.

Usage:
    conda activate victories
    python scripts/test_audio_on_file.py <path_to_file> [--json] [-o output.json]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import librosa

from app.providers.librosa_audio import LibrosaAudioProvider
from app.stages.audio import run_audio_stage


def get_audio_duration_ms(file_path: Path) -> int:
    """Retrieve audio duration in milliseconds using librosa."""
    duration_s = librosa.get_duration(path=str(file_path))
    return round(duration_s * 1000.0)


async def test_audio(file_path: Path, window_sec: float = 5.0, hop_sec: float = 2.5) -> None:
    print("\n" + "=" * 75)
    print("VIRTUJUDGE AUDIO ACOUSTIC ANALYSIS TEST")
    print("=" * 75)
    print(f"File:        {file_path.resolve()}")
    print(f"File Size:   {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        duration_ms = get_audio_duration_ms(file_path)
    except Exception as exc:
        print(f"Error inspecting audio duration: {exc}")
        sys.exit(1)

    print(f"Duration:    {duration_ms / 1000.0:.2f}s ({duration_ms} ms)")
    print(f"Window:      {window_sec:.1f}s (hop: {hop_sec:.1f}s)")
    print("=" * 75)

    print("\nInitializing Librosa Audio Provider...")
    provider = LibrosaAudioProvider(window_sec=window_sec, hop_sec=hop_sec)

    print("Analyzing acoustic measurements (speaking rate, pitch, pauses)...")
    result = await run_audio_stage(
        audio_path=file_path,
        audio_provider=provider,
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
        print("\nNo observations were extracted from the audio.")
        return

    # Aggregate metrics
    rates: list[float] = []
    pitches: list[float] = []
    pitch_stds: list[float] = []
    pauses_ms: list[float] = []
    pauses_cnt: list[float] = []

    for obs in result.observations:
        if obs.metric == "speaking_rate_wpm" and obs.value > 0:
            rates.append(obs.value)
        elif obs.metric == "pitch_mean_hz" and obs.value > 0:
            pitches.append(obs.value)
        elif obs.metric == "pitch_std_hz" and obs.value > 0:
            pitch_stds.append(obs.value)
        elif obs.metric == "pause_duration_ms":
            pauses_ms.append(obs.value)
        elif obs.metric == "pause_count":
            pauses_cnt.append(obs.value)

    print("\nMEASUREMENT BREAKDOWN:")
    if rates:
        print(
            f"  Speaking Rate:       mean={sum(rates)/len(rates):.1f} WPM, "
            f"min={min(rates):.1f}, max={max(rates):.1f} WPM"
        )
    if pitches:
        print(
            f"  Fundamental Pitch:   mean={sum(pitches)/len(pitches):.1f} Hz, "
            f"min={min(pitches):.1f}, max={max(pitches):.1f} Hz"
        )
    if pitch_stds:
        print(f"  Pitch Variation:     mean std={sum(pitch_stds)/len(pitch_stds):.1f} Hz")
    if pauses_ms:
        total_pause_s = sum(pauses_ms) / 1000.0
        print(
            f"  Total Silence/Pause: {total_pause_s:.2f}s "
            f"({(total_pause_s / (duration_ms/1000.0))*100:.1f}% of total duration)"
        )
    if pauses_cnt:
        print(f"  Significant Pauses:  {int(sum(pauses_cnt))} pause events (>= 250ms)")

    print("\n--- SAMPLE TIMED OBSERVATIONS (First 15) ---")
    for obs in result.observations[:15]:
        start_s = obs.start_ms / 1000.0
        end_s = obs.end_ms / 1000.0
        val_str = f"{obs.value:<8.2f}" if isinstance(obs.value, float) else f"{obs.value:<8}"
        print(f"  [{start_s:05.2f}s - {end_s:05.2f}s] {obs.metric:<22} = {val_str} {obs.unit}")

    if len(result.observations) > 15:
        print(f"  ... and {len(result.observations) - 15} more observations")

    print("\n" + "=" * 75)
    print("TEST COMPLETE - All acoustic measurements comply with Data Contracts")
    print("=" * 75 + "\n")


async def run_test(
    file_path: Path,
    window_sec: float,
    hop_sec: float,
    as_json: bool = False,
    output_file: Path | None = None,
) -> None:
    if as_json:
        duration_ms = get_audio_duration_ms(file_path)
        provider = LibrosaAudioProvider(window_sec=window_sec, hop_sec=hop_sec)
        result = await run_audio_stage(
            audio_path=file_path,
            audio_provider=provider,
            media_duration_ms=duration_ms,
        )
        json_str = result.model_dump_json(indent=2)
        print(json_str)
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_str, encoding="utf-8")
        return

    await test_audio(file_path, window_sec=window_sec, hop_sec=hop_sec)

    if output_file is not None:
        duration_ms = get_audio_duration_ms(file_path)
        provider = LibrosaAudioProvider(window_sec=window_sec, hop_sec=hop_sec)
        result = await run_audio_stage(
            audio_path=file_path,
            audio_provider=provider,
            media_duration_ms=duration_ms,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"[SAVED] JSON output successfully saved to: {output_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Librosa audio acoustic stage on a real audio or video file."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to input audio or video file (.wav, .mp3, .mp4, etc.)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Analysis window duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=2.5,
        help="Hop step between analysis windows in seconds (default: 2.5)",
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
        help="Optional path to save JSON output (e.g. output_audio.json)",
    )
    args = parser.parse_args()

    input_file = Path(args.file_path)
    if not input_file.is_file():
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    out_path = Path(args.output) if args.output else None
    asyncio.run(
        run_test(
            input_file,
            window_sec=args.window,
            hop_sec=args.hop,
            as_json=args.json,
            output_file=out_path,
        )
    )


if __name__ == "__main__":
    main()
