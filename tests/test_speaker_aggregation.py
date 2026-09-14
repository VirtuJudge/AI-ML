"""Unit tests for speaker pre-aggregation, turn cleaning, and active observation filtering."""

from app.providers.types import (
    AudioObservation,
    DiarizationResult,
    SpeakerSegment,
    VisualObservation,
)
from app.stages.aggregation import (
    SpeakerIntervalResult,
    aggregate_by_speaker,
    aggregate_speaker_observations,
    clean_diarization_turns,
    extract_speaker_intervals,
    filter_active_speaker_observations,
    format_duration_ms,
    format_interval,
    format_speaker_summary_for_prompt,
    format_timestamp_ms,
)


# ============================================================================
# 1. Diarization De-noising & Turn Cleaning Tests
# ============================================================================


def test_clean_diarization_merges_micro_gaps() -> None:
    """Verify 2ms-300ms micro-gaps between consecutive turns of the same speaker are merged."""
    segments = [
        # 2ms acoustic flutter gap
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=1002, end_ms=2500, speaker_label="SPEAKER_00"),
        # Exactly 300ms gap (boundary case: should merge)
        SpeakerSegment(start_ms=2800, end_ms=4000, speaker_label="SPEAKER_00"),
    ]

    cleaned = clean_diarization_turns(segments, min_duration_ms=100, merge_gap_ms=300)

    assert len(cleaned) == 1
    assert cleaned[0].speaker_label == "SPEAKER_00"
    assert cleaned[0].start_ms == 0
    assert cleaned[0].end_ms == 4000


def test_clean_diarization_preserves_distinct_turns() -> None:
    """Verify gaps > merge_gap_ms (e.g. 301ms) are not merged into a single turn."""
    segments = [
        SpeakerSegment(start_ms=0, end_ms=2000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=2350, end_ms=5000, speaker_label="SPEAKER_00"),  # 350ms gap
    ]

    cleaned = clean_diarization_turns(segments, min_duration_ms=100, merge_gap_ms=300)

    assert len(cleaned) == 2
    assert cleaned[0].start_ms == 0
    assert cleaned[0].end_ms == 2000
    assert cleaned[1].start_ms == 2350
    assert cleaned[1].end_ms == 5000


def test_clean_diarization_merges_overlapping_turns() -> None:
    """Verify overlapping turns from the same speaker are consolidated."""
    segments = [
        SpeakerSegment(start_ms=100, end_ms=1000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=800, end_ms=2000, speaker_label="SPEAKER_00"),
    ]

    cleaned = clean_diarization_turns(segments)

    assert len(cleaned) == 1
    assert cleaned[0].start_ms == 100
    assert cleaned[0].end_ms == 2000


def test_clean_diarization_prunes_isolated_micro_noise() -> None:
    """Verify isolated acoustic noise bursts < min_duration_ms (e.g. 20ms or 80ms) are pruned."""
    segments = [
        SpeakerSegment(start_ms=0, end_ms=50, speaker_label="SPEAKER_00"),  # 50ms isolated noise
        SpeakerSegment(start_ms=5000, end_ms=5080, speaker_label="SPEAKER_01"),  # 80ms isolated noise
        SpeakerSegment(start_ms=10000, end_ms=12000, speaker_label="SPEAKER_00"),  # 2000ms real turn
    ]

    cleaned = clean_diarization_turns(segments, min_duration_ms=100, merge_gap_ms=300)

    assert len(cleaned) == 1
    assert cleaned[0].speaker_label == "SPEAKER_00"
    assert cleaned[0].start_ms == 10000
    assert cleaned[0].end_ms == 12000


def test_clean_diarization_merges_micro_burst_with_adjacent_speech() -> None:
    """Verify a micro-burst < 100ms that merges with adjacent speech >= 100ms is preserved."""
    segments = [
        # Micro-burst of 40ms, followed 50ms later by real speech of 200ms
        SpeakerSegment(start_ms=1000, end_ms=1040, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=1090, end_ms=1290, speaker_label="SPEAKER_00"),
    ]

    cleaned = clean_diarization_turns(segments, min_duration_ms=100, merge_gap_ms=300)

    assert len(cleaned) == 1
    assert cleaned[0].start_ms == 1000
    assert cleaned[0].end_ms == 1290
    assert (cleaned[0].end_ms - cleaned[0].start_ms) == 290


def test_clean_diarization_multi_speaker_separation() -> None:
    """Verify segments for different speakers do not merge into each other."""
    segments = [
        SpeakerSegment(start_ms=0, end_ms=2000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=2100, end_ms=4000, speaker_label="SPEAKER_01"),
        SpeakerSegment(start_ms=4100, end_ms=6000, speaker_label="SPEAKER_00"),
    ]

    cleaned = clean_diarization_turns(segments, min_duration_ms=100, merge_gap_ms=300)

    assert len(cleaned) == 3
    assert cleaned[0].speaker_label == "SPEAKER_00"
    assert cleaned[0].start_ms == 0
    assert cleaned[0].end_ms == 2000

    assert cleaned[1].speaker_label == "SPEAKER_01"
    assert cleaned[1].start_ms == 2100
    assert cleaned[1].end_ms == 4000

    assert cleaned[2].speaker_label == "SPEAKER_00"
    assert cleaned[2].start_ms == 4100
    assert cleaned[2].end_ms == 6000


def test_clean_diarization_handles_empty_and_unknown_speakers() -> None:
    """Verify empty input and unknown speaker labels are safely handled."""
    assert clean_diarization_turns([]) == []
    assert clean_diarization_turns(None) == []

    unknown_segs = [
        SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_UNKNOWN"),
        SpeakerSegment(start_ms=1000, end_ms=2000, speaker_label="UNKNOWN"),
        SpeakerSegment(start_ms=2000, end_ms=3000, speaker_label=""),
    ]
    assert clean_diarization_turns(unknown_segs) == []


def test_clean_diarization_accepts_diarization_result() -> None:
    """Verify clean_diarization_turns accepts a DiarizationResult container."""
    result = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=1000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=1100, end_ms=2000, speaker_label="SPEAKER_00"),
        ],
        speaker_labels=["SPEAKER_00"],
    )
    cleaned = clean_diarization_turns(result)
    assert len(cleaned) == 1
    assert cleaned[0].start_ms == 0
    assert cleaned[0].end_ms == 2000


# ============================================================================
# 2. Active-Speaker Observation Filtering Tests
# ============================================================================


def test_filter_active_speaker_visual_observations() -> None:
    """Verify passive visual observations (when person is not speaking) are discarded."""
    cleaned_turns = [
        SpeakerSegment(start_ms=0, end_ms=5000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=10000, end_ms=15000, speaker_label="SPEAKER_01"),
    ]

    observations = [
        # SPEAKER_00 active speaking window (0-5s) -> KEEP
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="posture_openness",
            value=0.85,
            unit="ratio",
            speaker_label="SPEAKER_00",
        ),
        # SPEAKER_00 passive audience observation while SPEAKER_01 speaks (10-15s) -> DISCARD
        VisualObservation(
            start_ms=11000,
            end_ms=12000,
            metric="posture_openness",
            value=0.60,
            unit="ratio",
            speaker_label="SPEAKER_00",
        ),
        # SPEAKER_01 active speaking window (10-15s) -> KEEP
        VisualObservation(
            start_ms=11000,
            end_ms=12000,
            metric="posture_openness",
            value=0.90,
            unit="ratio",
            speaker_label="SPEAKER_01",
        ),
        # Unattributed observation -> DISCARD
        VisualObservation(
            start_ms=1000,
            end_ms=2000,
            metric="posture_openness",
            value=0.80,
            unit="ratio",
            speaker_label=None,
        ),
    ]

    filtered = filter_active_speaker_observations(observations, cleaned_turns)

    assert len(filtered) == 2
    assert filtered[0].speaker_label == "SPEAKER_00"
    assert filtered[0].start_ms == 1000
    assert filtered[1].speaker_label == "SPEAKER_01"
    assert filtered[1].start_ms == 11000


def test_filter_active_speaker_audio_observations_silent_gaps() -> None:
    """Verify audio observations falling completely within silent gaps are discarded."""
    cleaned_turns = [
        SpeakerSegment(start_ms=0, end_ms=3000, speaker_label="SPEAKER_00"),
        SpeakerSegment(start_ms=7000, end_ms=10000, speaker_label="SPEAKER_00"),
    ]

    observations = [
        # In turn 1 -> KEEP
        AudioObservation(
            start_ms=500,
            end_ms=2500,
            metric="speaking_rate_wpm",
            value=135.0,
            unit="words_per_minute",
            speaker_label="SPEAKER_00",
        ),
        # In silent gap (3s - 7s) -> DISCARD
        AudioObservation(
            start_ms=4000,
            end_ms=6000,
            metric="speaking_rate_wpm",
            value=0.0,
            unit="words_per_minute",
            speaker_label="SPEAKER_00",
        ),
        # Overlapping boundary: [2500, 3500] overlaps turn 1 by 500ms -> KEEP
        AudioObservation(
            start_ms=2500,
            end_ms=3500,
            metric="speaking_rate_wpm",
            value=130.0,
            unit="words_per_minute",
            speaker_label="SPEAKER_00",
        ),
        # Touching boundary: [3000, 4000] has overlap 0 with [0, 3000] -> DISCARD
        AudioObservation(
            start_ms=3000,
            end_ms=4000,
            metric="speaking_rate_wpm",
            value=120.0,
            unit="words_per_minute",
            speaker_label="SPEAKER_00",
        ),
    ]

    filtered = filter_active_speaker_observations(observations, cleaned_turns)

    assert len(filtered) == 2
    assert filtered[0].start_ms == 500
    assert filtered[1].start_ms == 2500


# ============================================================================
# 3. Interval Extraction & Time Formatting Tests
# ============================================================================


def test_format_timestamp_and_interval() -> None:
    """Verify timestamp and interval formatting to MM:SS."""
    assert format_timestamp_ms(0) == "00:00"
    assert format_timestamp_ms(5000) == "00:05"
    assert format_timestamp_ms(65000) == "01:05"
    assert format_timestamp_ms(135000) == "02:15"
    assert format_interval(0, 135000) == "00:00 - 02:15"
    assert format_interval(220000, 250000) == "03:40 - 04:10"


def test_format_duration_ms() -> None:
    """Verify duration formatting."""
    assert format_duration_ms(45000) == "45s"
    assert format_duration_ms(180000) == "3m 00s"
    assert format_duration_ms(195000) == "3m 15s"


def test_extract_speaker_intervals() -> None:
    """Verify extraction of active speaking turns and total speaking duration."""
    cleaned_turns = [
        SpeakerSegment(start_ms=0, end_ms=135000, speaker_label="SPEAKER_00"),      # 2m 15s
        SpeakerSegment(start_ms=140000, end_ms=160000, speaker_label="SPEAKER_01"),  # 20s
        SpeakerSegment(start_ms=220000, end_ms=250000, speaker_label="SPEAKER_00"),  # 30s
    ]

    res = extract_speaker_intervals(cleaned_turns, "SPEAKER_00")

    assert isinstance(res, SpeakerIntervalResult)
    assert res.speaking_time_ms == 135000 + 30000  # 165000 ms
    assert len(res.intervals) == 2

    assert res.intervals[0]["start_ms"] == 0
    assert res.intervals[0]["end_ms"] == 135000
    assert res.intervals[0]["formatted"] == "00:00 - 02:15"

    assert res.intervals[1]["start_ms"] == 220000
    assert res.intervals[1]["end_ms"] == 250000
    assert res.intervals[1]["formatted"] == "03:40 - 04:10"

    assert res.formatted_intervals == ["00:00 - 02:15", "03:40 - 04:10"]

    # Tuple unpacking check
    intervals, total_ms = extract_speaker_intervals(cleaned_turns, "SPEAKER_00")
    assert total_ms == 165000
    assert len(intervals) == 2


def test_extract_speaker_intervals_non_participating_speaker() -> None:
    """Verify extracting intervals for a non-participating speaker returns empty results."""
    cleaned_turns = [
        SpeakerSegment(start_ms=0, end_ms=10000, speaker_label="SPEAKER_00"),
    ]
    res = extract_speaker_intervals(cleaned_turns, "SPEAKER_99")
    assert res.speaking_time_ms == 0
    assert res.intervals == []
    assert res.formatted_intervals == []


# ============================================================================
# 4. Windowed Aggregation Math Tests
# ============================================================================


def test_aggregate_speaker_observations_math() -> None:
    """Verify mean/sum calculations on deterministic test observations in 10s windows."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=20000, speaker_label="SPEAKER_00"),
        ],
        speaker_labels=["SPEAKER_00"],
    )

    audio_obs = [
        # Window 0 (0-10s)
        AudioObservation(start_ms=0, end_ms=5000, metric="speaking_rate_wpm", value=130.0, unit="wpm", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="speaking_rate_wpm", value=150.0, unit="wpm", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=0, end_ms=5000, metric="pitch_mean_hz", value=140.0, unit="hz", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="pitch_mean_hz", value=160.0, unit="hz", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=0, end_ms=5000, metric="pitch_std_hz", value=10.0, unit="hz", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="pitch_std_hz", value=20.0, unit="hz", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=0, end_ms=5000, metric="pause_duration_ms", value=200.0, unit="ms", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="pause_duration_ms", value=350.0, unit="ms", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=0, end_ms=5000, metric="pause_count", value=1.0, unit="count", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="pause_count", value=2.0, unit="count", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=0, end_ms=5000, metric="filler_count", value=0.0, unit="count", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=5000, end_ms=10000, metric="filler_count", value=1.0, unit="count", speaker_label="SPEAKER_00"),
        # Window 1 (10-20s)
        AudioObservation(start_ms=10000, end_ms=15000, metric="speaking_rate_wpm", value=140.0, unit="wpm", speaker_label="SPEAKER_00"),
    ]

    visual_obs = [
        # Window 0 (0-10s)
        VisualObservation(start_ms=1000, end_ms=2000, metric="gaze_direction", value=1.0, unit="idx", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="gaze_direction", value=2.0, unit="idx", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=1000, end_ms=2000, metric="head_pitch_degrees", value=-2.0, unit="deg", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="head_pitch_degrees", value=4.0, unit="deg", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=1000, end_ms=2000, metric="head_yaw_degrees", value=-1.5, unit="deg", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="head_yaw_degrees", value=3.5, unit="deg", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=1000, end_ms=2000, metric="posture_openness", value=0.80, unit="ratio", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="posture_openness", value=0.90, unit="ratio", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=1000, end_ms=2000, metric="shoulder_symmetry_ratio", value=0.94, unit="ratio", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="shoulder_symmetry_ratio", value=0.98, unit="ratio", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=1000, end_ms=2000, metric="upper_body_movement_px", value=2.0, unit="px", speaker_label="SPEAKER_00"),
        VisualObservation(start_ms=5000, end_ms=6000, metric="upper_body_movement_px", value=4.4, unit="px", speaker_label="SPEAKER_00"),
    ]

    by_speaker = aggregate_speaker_observations(visual_obs, audio_obs, diarization)

    assert "SPEAKER_00" in by_speaker
    profile = by_speaker["SPEAKER_00"]
    assert profile["speaking_time_ms"] == 20000
    assert len(profile["intervals"]) == 1
    assert profile["intervals"][0]["formatted"] == "00:00 - 00:20"

    windows = profile["windows"]
    assert len(windows) == 2

    # Window 0: 0-10s
    w0 = windows[0]
    assert w0["start_ms"] == 0
    assert w0["end_ms"] == 10000

    ac0 = w0["acoustic"]
    assert ac0["speaking_rate_wpm"] == 140.0  # mean(130, 150)
    assert ac0["pitch_mean_hz"] == 150.0      # mean(140, 160)
    assert ac0["pitch_std_hz"] == 15.0       # mean(10, 20)
    assert ac0["pause_duration_ms"] == 550   # sum(200, 350)
    assert ac0["pause_count"] == 3           # sum(1, 2)
    assert ac0["filler_count"] == 1          # sum(0, 1)

    vs0 = w0["visual"]
    assert vs0["gaze_direction"] == 1.50           # mean(1.0, 2.0)
    assert vs0["head_pitch_degrees"] == 1.0        # mean(-2.0, 4.0)
    assert vs0["head_yaw_degrees"] == 1.0          # mean(-1.5, 3.5)
    assert vs0["posture_openness"] == 0.85         # mean(0.80, 0.90)
    assert vs0["shoulder_symmetry_ratio"] == 0.96  # mean(0.94, 0.98)
    assert vs0["upper_body_movement_px"] == 3.2    # mean(2.0, 4.4)

    # Window 1: 10-20s (acoustic only, no visual)
    w1 = windows[1]
    assert w1["start_ms"] == 10000
    assert w1["end_ms"] == 20000
    assert w1["acoustic"]["speaking_rate_wpm"] == 140.0
    assert "visual" not in w1


def test_aggregate_omits_empty_windows() -> None:
    """Verify windows where the speaker was not presenting are omitted (no zero-filling)."""
    # Speaker presents at 0-10s and 40-50s; silent between 10-40s
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=10000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=40000, end_ms=50000, speaker_label="SPEAKER_00"),
        ],
        speaker_labels=["SPEAKER_00"],
    )

    audio_obs = [
        AudioObservation(start_ms=1000, end_ms=5000, metric="speaking_rate_wpm", value=135.0, unit="wpm", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=41000, end_ms=45000, metric="speaking_rate_wpm", value=142.0, unit="wpm", speaker_label="SPEAKER_00"),
    ]

    by_speaker = aggregate_by_speaker([], audio_obs, diarization)

    profile = by_speaker["SPEAKER_00"]
    windows = profile["windows"]

    # Only 2 windows should exist: [0, 10000] and [40000, 50000]
    # Windows [10000, 20000], [20000, 30000], [30000, 40000] MUST NOT be present!
    assert len(windows) == 2
    assert windows[0]["start_ms"] == 0
    assert windows[0]["end_ms"] == 10000
    assert windows[1]["start_ms"] == 40000
    assert windows[1]["end_ms"] == 50000


def test_aggregate_multiple_speakers_isolated_profiles() -> None:
    """Verify multiple presenting speakers receive isolated, clean profiles."""
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start_ms=0, end_ms=10000, speaker_label="SPEAKER_00"),
            SpeakerSegment(start_ms=10000, end_ms=20000, speaker_label="SPEAKER_01"),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
    )

    audio_obs = [
        AudioObservation(start_ms=2000, end_ms=4000, metric="speaking_rate_wpm", value=130.0, unit="wpm", speaker_label="SPEAKER_00"),
        AudioObservation(start_ms=12000, end_ms=14000, metric="speaking_rate_wpm", value=160.0, unit="wpm", speaker_label="SPEAKER_01"),
    ]

    by_speaker = aggregate_speaker_observations([], audio_obs, diarization)

    assert "SPEAKER_00" in by_speaker
    assert "SPEAKER_01" in by_speaker

    assert by_speaker["SPEAKER_00"]["speaking_time_ms"] == 10000
    assert len(by_speaker["SPEAKER_00"]["windows"]) == 1
    assert by_speaker["SPEAKER_00"]["windows"][0]["acoustic"]["speaking_rate_wpm"] == 130.0

    assert by_speaker["SPEAKER_01"]["speaking_time_ms"] == 10000
    assert len(by_speaker["SPEAKER_01"]["windows"]) == 1
    assert by_speaker["SPEAKER_01"]["windows"][0]["acoustic"]["speaking_rate_wpm"] == 160.0


# ============================================================================
# 5. Prompt Formatting Tests
# ============================================================================


def test_format_speaker_summary_for_prompt() -> None:
    """Verify format_speaker_summary_for_prompt creates bounded text context for AI-07."""
    by_speaker = {
        "SPEAKER_00": {
            "speaking_time_ms": 135000,
            "intervals": [
                {"start_ms": 0, "end_ms": 135000, "formatted": "00:00 - 02:15"},
            ],
            "windows": [
                {
                    "start_ms": 0,
                    "end_ms": 10000,
                    "acoustic": {
                        "speaking_rate_wpm": 138.0,
                        "pitch_mean_hz": 145.2,
                        "pause_duration_ms": 250,
                        "pause_count": 1,
                        "filler_count": 0,
                    },
                    "visual": {
                        "gaze_direction": 1.0,
                        "posture_openness": 0.85,
                        "shoulder_symmetry_ratio": 0.96,
                    },
                },
            ],
        },
    }

    mappings = {"SPEAKER_00": "Jane Founder"}

    summary = format_speaker_summary_for_prompt(by_speaker, mappings)

    assert "### Presenter Delivery Profiles:" in summary
    assert "Jane Founder (SPEAKER_00)" in summary
    assert "Active Speaking Turns: 00:00 - 02:15 (Total: 2m 15s)" in summary
    assert "Avg Pace: 138.0 WPM" in summary
    assert "Avg Pitch: 145.2 Hz" in summary
    assert "Avg Posture Openness: 0.85" in summary
    assert "[00:00 - 00:10]" in summary


def test_format_speaker_summary_empty_fallback() -> None:
    """Verify graceful handling when by_speaker is empty."""
    assert format_speaker_summary_for_prompt({}) == "No presenter delivery metrics available."
