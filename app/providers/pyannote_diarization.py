"""pyannote speaker diarization adapter.

Implements DiarizationProvider using pyannote.audio (or mocked pipeline) to produce
anonymous speaker labels (SPEAKER_00, SPEAKER_01, etc.) in SpeakerSegment format
without leaking model configuration to boundary contracts.
"""

import asyncio
import inspect
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.contracts import Limitation
from app.providers.types import DiarizationResult, SpeakerSegment

CANONICAL_DIARIZATION_3_1 = "pyannote/speaker-diarization-3.1"
CANONICAL_DIARIZATION_COMMUNITY_1 = "pyannote/speaker-diarization-community-1"

# Minimum pyannote.audio (major, minor) version required per canonical model.
SUPPORTED_DIARIZATION_MODELS: dict[str, tuple[int, int]] = {
    CANONICAL_DIARIZATION_3_1: (3, 1),
    CANONICAL_DIARIZATION_COMMUNITY_1: (4, 0),
}

# Explicit alias -> canonical HuggingFace repo ID mapping.
# Aliases resolve strictly to the model they name without cross-model substitution.
DIARIZATION_MODEL_ALIASES: dict[str, str] = {
    CANONICAL_DIARIZATION_3_1: CANONICAL_DIARIZATION_3_1,
    "speaker-diarization-3.1": CANONICAL_DIARIZATION_3_1,
    "pyannote-3.1": CANONICAL_DIARIZATION_3_1,
    "3.1": CANONICAL_DIARIZATION_3_1,
    CANONICAL_DIARIZATION_COMMUNITY_1: CANONICAL_DIARIZATION_COMMUNITY_1,
    "speaker-diarization-community-1": CANONICAL_DIARIZATION_COMMUNITY_1,
    "pyannote-community-1": CANONICAL_DIARIZATION_COMMUNITY_1,
    "community-1": CANONICAL_DIARIZATION_COMMUNITY_1,
}

_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")
_GENERIC_ORIGINS = frozenset({"huggingface", "local"})


def _parse_version_tuple(version_str: str) -> tuple[int, ...] | None:
    """Parse leading numeric version components from a PEP 440-like version string."""
    match = _VERSION_RE.match(version_str)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def resolve_diarization_model(
    model_name: str | None = None,
    *,
    revision: str | None = None,
    default_model: str = CANONICAL_DIARIZATION_3_1,
) -> tuple[str, str, str | None, str | None]:
    """Resolve a diarization model name or alias to its canonical model ID.

    Returns:
        Tuple of ``(requested_model, canonical_model, model_alias, requested_revision)``.

    Raises:
        ValueError: If the model name, env var, or revision is empty or unsupported.
    """
    if model_name is not None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a non-empty supported pyannote diarization model or alias."
            )
        requested_model = model_name.strip()
    else:
        env_model = os.getenv("DIARIZATION_MODEL")
        if env_model is not None:
            if not env_model.strip():
                raise ValueError("DIARIZATION_MODEL must not be empty when set.")
            requested_model = env_model.strip()
        else:
            requested_model = default_model

    canonical_model = DIARIZATION_MODEL_ALIASES.get(requested_model.lower())
    if canonical_model is None:
        if "@" in requested_model:
            raise ValueError(
                f"Unsupported pyannote diarization model specification '{requested_model}'. "
                "Pass the model ID or alias in 'model_name' and pin versions via 'revision'."
            )
        supported = ", ".join(sorted(SUPPORTED_DIARIZATION_MODELS))
        aliases = ", ".join(sorted(k for k, v in DIARIZATION_MODEL_ALIASES.items() if k != v))
        raise ValueError(
            f"Unsupported pyannote diarization model '{requested_model}'. "
            f"Supported models: {supported} (aliases: {aliases})."
        )

    requested_revision: str | None = None
    if revision is not None:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string when provided.")
        requested_revision = revision.strip()

    model_alias = requested_model if requested_model != canonical_model else None
    return requested_model, canonical_model, model_alias, requested_revision


def _extract_str_attr(obj: Any, attr_names: tuple[str, ...]) -> str | None:
    """Return the first non-empty string attribute value on obj (ignoring mocks)."""
    for attr_name in attr_names:
        val = getattr(obj, attr_name, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_loaded_provenance(
    pipeline: Any,
    *,
    canonical_model: str,
    requested_revision: str | None,
) -> tuple[str, str | None]:
    """Extract actual loaded model ID and revision from a loaded pipeline instance."""
    reported_model = _extract_str_attr(
        pipeline,
        ("loaded_model", "model_id", "_otel_origin"),
    )
    if reported_model is not None and reported_model.lower() not in _GENERIC_ORIGINS:
        resolved_reported = DIARIZATION_MODEL_ALIASES.get(reported_model.lower(), reported_model)
        if resolved_reported != canonical_model:
            raise RuntimeError(
                f"Loaded pyannote pipeline model '{reported_model}' does not match "
                f"requested canonical model '{canonical_model}'."
            )

    reported_revision = _extract_str_attr(
        pipeline,
        ("loaded_revision", "revision", "model_revision", "_revision"),
    )
    if reported_revision is None:
        config = getattr(pipeline, "config", None)
        if isinstance(config, Mapping):
            cfg_rev = config.get("revision")
            if isinstance(cfg_rev, str) and cfg_rev.strip():
                reported_revision = cfg_rev.strip()

    return canonical_model, reported_revision or requested_revision


def _load_pretrained_pipeline(
    pipeline_cls: Any,
    canonical_model: str,
    *,
    token: str | None,
    revision: str | None,
) -> Any:
    """Call Pipeline.from_pretrained without silently dropping revision or substituting models."""
    from_pretrained = pipeline_cls.from_pretrained
    accepts_token = True
    accepts_use_auth_token = False
    accepts_revision = True

    try:
        sig = inspect.signature(from_pretrained)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            accepts_token = "token" in params
            accepts_use_auth_token = "use_auth_token" in params
            accepts_revision = "revision" in params
    except (ValueError, TypeError):
        pass

    if revision is not None and not accepts_revision:
        raise ValueError(
            f"Requested revision '{revision}' for '{canonical_model}', "
            "but Pipeline.from_pretrained does not accept a 'revision' argument."
        )

    extra_kwargs: dict[str, Any] = {}
    if revision is not None:
        extra_kwargs["revision"] = revision

    if accepts_token:
        try:
            return from_pretrained(canonical_model, token=token, **extra_kwargs)
        except TypeError as err:
            # Only fall back to legacy use_auth_token if 'token' itself was rejected.
            if "token" not in str(err):
                raise
            return from_pretrained(canonical_model, use_auth_token=token, **extra_kwargs)

    if accepts_use_auth_token:
        return from_pretrained(canonical_model, use_auth_token=token, **extra_kwargs)

    return from_pretrained(canonical_model, **extra_kwargs)


class PyannoteDiarizationProvider:
    """Speaker diarization adapter powered by pyannote.audio."""

    DEFAULT_MODEL = CANONICAL_DIARIZATION_3_1
    adapter_version = "pyannote_diarization/1.0.0"

    def __init__(
        self,
        model_name: str | None = None,
        auth_token: str | None = None,
        device: str = "auto",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        pipeline_instance: Any = None,
        revision: str | None = None,
    ) -> None:
        """Initialize PyannoteDiarizationProvider.

        Args:
            model_name: Supported HuggingFace model repo ID or alias
                (default: env DIARIZATION_MODEL or 'pyannote/speaker-diarization-3.1').
            auth_token: HuggingFace auth token for downloading gated models.
                If omitted, reads from HUGGINGFACE_TOKEN or HF_TOKEN env var.
            device: Compute device ('cpu', 'cuda', or 'auto').
            min_speakers: Minimum number of speakers expected.
            max_speakers: Maximum number of speakers expected.
            pipeline_instance: Pre-instantiated pipeline instance
                (useful for unit testing without weights).
            revision: Optional HuggingFace model revision (branch, tag, or commit SHA).

        Raises:
            ValueError: If model_name, revision, speaker bounds, or installed
                pyannote.audio version combination is unsupported.
            ImportError: If pyannote.audio is required and not installed.
        """
        if min_speakers is not None and min_speakers < 1:
            raise ValueError("min_speakers must be >= 1 when provided.")
        if max_speakers is not None and max_speakers < 1:
            raise ValueError("max_speakers must be >= 1 when provided.")
        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            raise ValueError(
                f"Unsupported speaker count range: min_speakers ({min_speakers}) > "
                f"max_speakers ({max_speakers})."
            )

        requested_model, canonical_model, model_alias, requested_revision = (
            resolve_diarization_model(
                model_name,
                revision=revision,
                default_model=self.DEFAULT_MODEL,
            )
        )

        self.requested_model: str = requested_model
        self.canonical_model: str = canonical_model
        self.model_alias: str | None = model_alias
        self.requested_revision: str | None = requested_revision

        resolved_token = (
            auth_token
            if auth_token is not None
            else (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"))
        )
        self.auth_token = resolved_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        self._device: Any = None
        if pipeline_instance is not None:
            self._pipeline = pipeline_instance
            if hasattr(pipeline_instance, "device"):
                self._device = pipeline_instance.device
            self.loaded_model, self.loaded_revision = _extract_loaded_provenance(
                pipeline_instance,
                canonical_model=canonical_model,
                requested_revision=requested_revision,
            )
        else:
            try:
                from pyannote.audio import Pipeline
            except ImportError as err:
                raise ImportError(
                    "pyannote.audio is required for PyannoteDiarizationProvider. "
                    "Install it with 'pip install virtujudge-ai-ml[speech]' "
                    "or 'pip install pyannote.audio'."
                ) from err

            pyannote_audio_mod = sys.modules.get("pyannote.audio")
            pyannote_version_str = getattr(pyannote_audio_mod, "__version__", None)
            if isinstance(pyannote_version_str, str):
                parsed_version = _parse_version_tuple(pyannote_version_str)
                required_version = SUPPORTED_DIARIZATION_MODELS[canonical_model]
                if parsed_version is not None and parsed_version[:2] < required_version:
                    req_str = ".".join(str(part) for part in required_version)
                    raise ValueError(
                        f"Diarization model '{canonical_model}' requires "
                        f"pyannote.audio>={req_str}, "
                        f"but version '{pyannote_version_str}' is installed. "
                        "Upgrade pyannote.audio or select a compatible model explicitly."
                    )

            pipeline = _load_pretrained_pipeline(
                Pipeline,
                canonical_model,
                token=resolved_token,
                revision=requested_revision,
            )

            if pipeline is None:
                raise RuntimeError(
                    f"Could not load pyannote pipeline '{canonical_model}'. "
                    "Verify your HUGGINGFACE_TOKEN and ensure you have accepted conditions on HF."
                )

            self.loaded_model, self.loaded_revision = _extract_loaded_provenance(
                pipeline,
                canonical_model=canonical_model,
                requested_revision=requested_revision,
            )

            # Route to CUDA if available and requested
            try:
                import torch

                if self.device == "auto":
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    dev = torch.device(self.device)
                pipeline.to(dev)
                self._device = dev
            except Exception:
                self._device = None

            self._pipeline = pipeline

        # Report the actual loaded canonical model as model_name for stage/checkpoint provenance.
        self.model_name: str = self.loaded_model

    def _diarize_sync(self, audio_path: Path) -> DiarizationResult:
        """Synchronously run speaker diarization on the given audio path."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        kwargs: dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        # Load audio waveform directly via soundfile to bypass torchcodec DLL issues on Windows
        try:
            import soundfile as sf
            import torch

            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)

            if self._device is not None:
                waveform = waveform.to(self._device)

            audio_input: Any = {"waveform": waveform, "sample_rate": sr}
        except Exception:
            audio_input = str(audio_path)

        diarization_output = self._pipeline(audio_input, **kwargs)

        segments: list[SpeakerSegment] = []
        speaker_map: dict[str, str] = {}
        uncertain_turns_count = 0

        # pyannote 4.0 returns DiarizeOutput with .speaker_diarization, while 3.x returns Annotation
        annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
        it = annotation.itertracks(yield_label=True) if hasattr(annotation, "itertracks") else []

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
            "model_name": self.loaded_model,
            "loaded_model": self.loaded_model,
            "loaded_revision": self.loaded_revision,
            "requested_model": self.requested_model,
            "model_alias": self.model_alias,
            "requested_revision": self.requested_revision,
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

    def checkpoint_config(self) -> dict[str, Any]:
        """Return safe configuration metadata for stage checkpoint fingerprinting."""
        return {
            "loaded_model": self.loaded_model,
            "loaded_revision": self.loaded_revision,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
        }

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
    "CANONICAL_DIARIZATION_3_1",
    "CANONICAL_DIARIZATION_COMMUNITY_1",
    "DIARIZATION_MODEL_ALIASES",
    "SUPPORTED_DIARIZATION_MODELS",
    "PyannoteDiarizationProvider",
    "create_diarization_uncertainty_limitation",
    "resolve_diarization_model",
]
