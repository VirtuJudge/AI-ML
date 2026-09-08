"""pyannote Community-1 speaker diarization adapter.

Implements DiarizationProvider using pyannote.audio (or mocked pipeline) to produce
anonymous speaker labels (SPEAKER_00, SPEAKER_01, etc.) in SpeakerSegment format
without leaking model configuration to boundary contracts.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from app.contracts import Limitation
from app.providers.types import DiarizationResult, SpeakerSegment


class PyannoteDiarizationProvider:
    """Speaker diarization adapter powered by pyannote.audio Community-1 / 3.1."""

    DEFAULT_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote-community-1")

    def __init__(
        self,
        model_name: str | None = None,
        auth_token: str | None = None,
        device: str = "auto",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        pipeline_instance: Any = None,
    ) -> None:
        """Initialize PyannoteDiarizationProvider.

        Args:
            model_name: HuggingFace model repo ID or alias
                (default: env DIARIZATION_MODEL or 'pyannote-community-1').
            auth_token: HuggingFace auth token for downloading gated models.
                If omitted, reads from HUGGINGFACE_TOKEN or HF_TOKEN env var.
            device: Compute device ('cpu', 'cuda', or 'auto').
            min_speakers: Minimum number of speakers expected.
            max_speakers: Maximum number of speakers expected.
            pipeline_instance: Pre-instantiated pipeline instance
                (useful for unit testing without weights).

        Raises:
            ImportError: If pyannote.audio is not installed and pipeline_instance is not provided.
        """
        self.model_name = model_name or os.getenv("DIARIZATION_MODEL", self.DEFAULT_MODEL)

        resolved_token = (
            auth_token
            if auth_token is not None
            else (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"))
        )
        self.auth_token = resolved_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        if pipeline_instance is not None:
            self._pipeline = pipeline_instance
        else:
            try:
                from pyannote.audio import Pipeline
            except ImportError as err:
                raise ImportError(
                    "pyannote.audio is required for PyannoteDiarizationProvider. "
                    "Install it with 'pip install virtujudge-ai-ml[speech]' "
                    "or 'pip install pyannote.audio'."
                ) from err

            hf_model = (
                "pyannote/speaker-diarization-community-1"
                if self.model_name == "pyannote-community-1"
                else self.model_name
            )

            # pyannote 3.1+ uses `token`, while older versions use `use_auth_token`
            try:
                pipeline = Pipeline.from_pretrained(
                    hf_model,
                    token=resolved_token,
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    hf_model,
                    use_auth_token=resolved_token,  # type: ignore[call-arg]
                )

            if pipeline is None:
                raise RuntimeError(
                    f"Could not load pyannote pipeline '{self.model_name}'. "
                    "Verify your HUGGINGFACE_TOKEN and ensure you have accepted conditions on HF."
                )

            # Route to CUDA if available and requested
            try:
                import torch

                if self.device == "auto":
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    dev = torch.device(self.device)
                pipeline.to(dev)
            except Exception:
                pass

            self._pipeline = pipeline

    def _diarize_sync(self, audio_path: Path) -> DiarizationResult:
        """Synchronously run speaker diarization on the given audio path."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        kwargs: dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        diarization_output = self._pipeline(str(audio_path), **kwargs)

        segments: list[SpeakerSegment] = []
        speaker_map: dict[str, str] = {}
        uncertain_turns_count = 0

        # pyannote 4.0 returns DiarizeOutput with .speaker_diarization, while 3.x returns Annotation
        annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
        it = (
            annotation.itertracks(yield_label=True)
            if hasattr(annotation, "itertracks")
            else []
        )

        for item in it:
            if len(item) == 3:
                turn, _, raw_speaker = item
            else:
                turn, raw_speaker = item

            start_ms = round(getattr(turn, "start", 0.0) * 1000)
            end_ms = round(getattr(turn, "end", 0.0) * 1000)

            raw_str = str(raw_speaker).strip() if raw_speaker is not None else ""
            if not raw_str or raw_str.upper() in ("UNKNOWN", "?", "NONE"):
                uncertain_turns_count += 1
                speaker_label = "SPEAKER_UNKNOWN"
            else:
                if raw_str not in speaker_map:
                    # Anonymous label assignment: SPEAKER_00, SPEAKER_01, etc.
                    speaker_map[raw_str] = f"SPEAKER_{len(speaker_map):02d}"
                speaker_label = speaker_map[raw_str]

            segments.append(
                SpeakerSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_label=speaker_label,
                )
            )

        segments.sort(key=lambda s: (s.start_ms, s.end_ms))
        speaker_labels = sorted(speaker_map.values())

        metadata: dict[str, Any] = {
            "provider": "pyannote.audio",
            "model_name": self.model_name,
            "device": self.device,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "detected_speaker_count": len(speaker_labels),
            "uncertain_turns_count": uncertain_turns_count,
        }

        return DiarizationResult(
            segments=segments,
            speaker_labels=speaker_labels,
            metadata=metadata,
        )

    async def diarize(self, audio_path: Path) -> DiarizationResult:
        """Diarize an audio file asynchronously."""
        return await asyncio.to_thread(self._diarize_sync, audio_path)


def create_diarization_uncertainty_limitation(uncertain_count: int) -> Limitation:
    """Generate a log-safe limitation for uncertain or unassigned speaker turns."""
    return Limitation(
        code="uncertain_speaker_assignment",
        scope="diarization",
        message=(
            f"Speaker diarization produced {uncertain_count} unassigned or ambiguous turn(s). "
            "Individual speaker attribution may be incomplete."
        ),
        affected_dimensions=["individual_feedback", "speaker_attribution"],
    )


__all__ = [
    "PyannoteDiarizationProvider",
    "create_diarization_uncertainty_limitation",
]
