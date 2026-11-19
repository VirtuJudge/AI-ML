"""Stage checkpointing helpers for caching intermediate stage artifacts.

Provides helper functions to cache and reuse intermediate stage outputs in
ObjectStorageProtocol, preventing expensive recomputation across pipeline runs
while enforcing strict compatibility fingerprinting on producer, model, adapter,
stage, configuration, and source checksums.
"""

import hashlib
import json
import logging
import math
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import ArtifactRef
from app.storage.base import ObjectNotFoundError, ObjectStorageProtocol

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FINGERPRINT_VERSION = 1
DEFAULT_PRODUCER_VERSION = "ai-ml/0.1.0"

_SAFE_PROVIDER_CONFIG_ATTRS: tuple[tuple[str, str], ...] = (
    ("sample_fps", "sample_fps"),
    ("max_persons", "max_persons"),
    ("sample_rate", "sample_rate"),
    ("window_sec", "window_sec"),
    ("hop_sec", "hop_sec"),
    ("hop_length", "hop_length"),
    ("fmin", "fmin"),
    ("fmax", "fmax"),
    ("silence_threshold_db", "silence_threshold_db"),
    ("min_speakers", "min_speakers"),
    ("max_speakers", "max_speakers"),
    ("max_file_size_bytes", "max_file_size_bytes"),
    ("duration_ms", "duration_ms"),
    ("_duration_ms", "duration_ms"),
    ("sampling_rate", "sampling_rate"),
    ("fps", "fps"),
    ("version", "version"),
    ("face_model", "face_model"),
    ("pose_model", "pose_model"),
)


def _to_json_safe(value: Any) -> Any:
    """Convert arbitrary config values into deterministic JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_json_safe(value.value)
    if isinstance(value, BaseModel):
        return _to_json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(k): _to_json_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_to_json_safe(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return str(value)


class StageCheckpointFingerprint(BaseModel):
    """Versioned compatibility fingerprint for intermediate stage checkpoints."""

    fingerprint_version: int = CHECKPOINT_FINGERPRINT_VERSION
    stage: str
    stage_version: str
    producer_version: str = DEFAULT_PRODUCER_VERSION
    provider_mode: str = "fake"
    source_checksums: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stage_config: dict[str, Any] = Field(default_factory=dict)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-normalized dictionary representation."""
        dumped = self.model_dump(mode="json")
        normalized = _to_json_safe(dumped)
        assert isinstance(normalized, dict)
        return normalized

    def canonical_json(self) -> str:
        """Serialize fingerprint deterministically for hashing."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Return the SHA-256 hex digest of the canonical fingerprint."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


CompatibilityFingerprintInput = StageCheckpointFingerprint | Mapping[str, Any] | str


def extract_provider_fingerprint(provider: Any) -> dict[str, Any]:
    """Extract non-sensitive identity, version, and configuration metadata from a provider."""
    provider_cls = type(provider)
    provider_module = sys.modules.get(provider_cls.__module__)
    module_algo_version = (
        getattr(provider_module, "ALGORITHM_VERSION", None) if provider_module else None
    )

    model = getattr(provider, "model_name", None) or getattr(provider, "model", None) or "unknown"
    adapter_version = (
        getattr(provider, "adapter_version", None)
        or getattr(provider, "algorithm_version", None)
        or getattr(provider, "ALGORITHM_VERSION", None)
        or module_algo_version
        or "1.0.0"
    )
    feature_version = (
        getattr(provider, "feature_version", None)
        or getattr(provider, "algorithm_version", None)
        or getattr(provider, "ALGORITHM_VERSION", None)
        or module_algo_version
    )

    config: dict[str, Any] = {}
    checkpoint_config_fn = getattr(provider, "checkpoint_config", None)
    if callable(checkpoint_config_fn):
        raw_config = checkpoint_config_fn()
        if isinstance(raw_config, Mapping):
            for key, val in raw_config.items():
                config[str(key)] = _to_json_safe(val)

    for attr_name, config_key in _SAFE_PROVIDER_CONFIG_ATTRS:
        if config_key in config:
            continue
        if hasattr(provider, attr_name):
            attr_val = getattr(provider, attr_name)
            if attr_val is not None and not callable(attr_val):
                config[config_key] = _to_json_safe(attr_val)

    if provider_module is not None:
        for mod_attr, config_key in (
            ("FACE_MODEL_URL", "face_model"),
            ("POSE_MODEL_URL", "pose_model"),
        ):
            if config_key not in config and hasattr(provider_module, mod_attr):
                config[config_key] = _to_json_safe(getattr(provider_module, mod_attr))

    result: dict[str, Any] = {
        "provider_class": f"{provider_cls.__module__}.{provider_cls.__qualname__}",
        "model": _to_json_safe(model),
        "adapter_version": _to_json_safe(adapter_version),
        "config": _to_json_safe(config),
    }
    if feature_version is not None:
        result["feature_version"] = _to_json_safe(feature_version)
    return result


def build_stage_compatibility_fingerprint(
    *,
    stage: str,
    stage_version: str,
    producer_version: str = DEFAULT_PRODUCER_VERSION,
    provider_mode: str = "fake",
    source_checksums: Mapping[str, str] | None = None,
    providers: Mapping[str, Any] | None = None,
    stage_config: Mapping[str, Any] | None = None,
    fingerprint_version: int = CHECKPOINT_FINGERPRINT_VERSION,
) -> StageCheckpointFingerprint:
    """Construct a versioned StageCheckpointFingerprint from stage and provider metadata."""
    normalized_checksums = (
        {str(k): str(v) for k, v in sorted(source_checksums.items(), key=lambda kv: str(kv[0]))}
        if source_checksums
        else {}
    )

    normalized_providers: dict[str, dict[str, Any]] = {}
    if providers:
        for role, provider_or_meta in sorted(providers.items(), key=lambda kv: str(kv[0])):
            if isinstance(provider_or_meta, Mapping):
                safe_meta = _to_json_safe(provider_or_meta)
                normalized_providers[str(role)] = (
                    safe_meta if isinstance(safe_meta, dict) else {"value": safe_meta}
                )
            else:
                normalized_providers[str(role)] = extract_provider_fingerprint(provider_or_meta)

    normalized_stage_config: dict[str, Any] = {}
    if stage_config:
        safe_cfg = _to_json_safe(stage_config)
        if isinstance(safe_cfg, dict):
            normalized_stage_config = safe_cfg

    return StageCheckpointFingerprint(
        fingerprint_version=fingerprint_version,
        stage=stage,
        stage_version=stage_version,
        producer_version=producer_version,
        provider_mode=provider_mode,
        source_checksums=normalized_checksums,
        providers=normalized_providers,
        stage_config=normalized_stage_config,
    )


def normalize_compatibility_fingerprint(
    compatibility_fingerprint: CompatibilityFingerprintInput | None,
) -> tuple[dict[str, Any] | str | None, str | None]:
    """Normalize a compatibility fingerprint and return (canonical_value, sha256_hex)."""
    if compatibility_fingerprint is None:
        return None, None

    if isinstance(compatibility_fingerprint, StageCheckpointFingerprint):
        canonical_dict = compatibility_fingerprint.to_canonical_dict()
        return canonical_dict, compatibility_fingerprint.digest()

    if isinstance(compatibility_fingerprint, Mapping):
        safe = _to_json_safe(compatibility_fingerprint)
        assert isinstance(safe, dict)
        serialized = json.dumps(safe, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return safe, digest

    if isinstance(compatibility_fingerprint, str):
        serialized = json.dumps(compatibility_fingerprint)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return compatibility_fingerprint, digest

    raise TypeError(
        "compatibility_fingerprint must be StageCheckpointFingerprint, Mapping, str, or None."
    )


def compute_fingerprint_digest(compatibility_fingerprint: CompatibilityFingerprintInput) -> str:
    """Compute the canonical SHA-256 hex digest for a compatibility fingerprint."""
    _, digest = normalize_compatibility_fingerprint(compatibility_fingerprint)
    if digest is None:
        raise ValueError("compatibility_fingerprint must not be None")
    return digest


def _checkpoint_key(
    session_id: str,
    stage: str,
    input_checksum: str,
    compatibility_fingerprint: CompatibilityFingerprintInput | None = None,
) -> str:
    """Build standardized checkpoint object key for a session stage, checksum, and fingerprint."""
    clean_hash = input_checksum.removeprefix("sha256:")
    if compatibility_fingerprint is None:
        return f"ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json"
    digest = compute_fingerprint_digest(compatibility_fingerprint)
    return f"ai/session/{session_id}/checkpoints/{stage}_{clean_hash}_{digest[:16]}.json"


async def get_stage_checkpoint(
    storage: ObjectStorageProtocol,
    session_id: str,
    stage: str,
    input_checksum: str,
    expected_schema_version: int = CHECKPOINT_SCHEMA_VERSION,
    expected_producer_version: str = DEFAULT_PRODUCER_VERSION,
    expected_compatibility_fingerprint: CompatibilityFingerprintInput | None = None,
) -> dict[str, Any] | None:
    """Read and validate an intermediate stage checkpoint from object storage.

    Reads JSON from the checkpoint key. Verifies stage, schema_version,
    producer_version, input_checksum, and compatibility_fingerprint (when supplied,
    or ensures no fingerprinted checkpoint is reused without a matching fingerprint).
    If valid, returns the saved data dictionary.
    If missing (ObjectNotFoundError), corrupt, or mismatched, returns None.
    """
    key = _checkpoint_key(
        session_id,
        stage,
        input_checksum,
        compatibility_fingerprint=expected_compatibility_fingerprint,
    )
    expected_fp, expected_digest = normalize_compatibility_fingerprint(
        expected_compatibility_fingerprint
    )

    try:
        payload = await storage.read_json(key)
    except ObjectNotFoundError:
        logger.debug(
            "Stage checkpoint miss for stage=%s session=%s key=%s (not found)",
            stage,
            session_id,
            key,
        )
        return None
    except Exception:
        logger.warning(
            "Stage checkpoint read error for stage=%s session=%s key=%s; treating as miss",
            stage,
            session_id,
            key,
            exc_info=True,
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Stage checkpoint corrupt for stage=%s session=%s: expected dict payload, got %s",
            stage,
            session_id,
            type(payload).__name__,
        )
        return None

    if payload.get("stage") != stage:
        logger.info(
            "Stage checkpoint stage mismatch for stage=%s session=%s: expected %s, got %s",
            stage,
            session_id,
            stage,
            payload.get("stage"),
        )
        return None

    if payload.get("schema_version") != expected_schema_version:
        logger.info(
            "Stage checkpoint schema_version mismatch for stage=%s session=%s: expected %s, got %s",
            stage,
            session_id,
            expected_schema_version,
            payload.get("schema_version"),
        )
        return None

    if payload.get("producer_version") != expected_producer_version:
        logger.info(
            "Stage checkpoint producer_version mismatch for stage=%s session=%s: "
            "expected %s, got %s",
            stage,
            session_id,
            expected_producer_version,
            payload.get("producer_version"),
        )
        return None

    if payload.get("input_checksum") != input_checksum:
        logger.info(
            "Stage checkpoint input_checksum mismatch for stage=%s session=%s: expected %s, got %s",
            stage,
            session_id,
            input_checksum,
            payload.get("input_checksum"),
        )
        return None

    stored_fp = payload.get("compatibility_fingerprint")
    stored_digest = payload.get("fingerprint_digest")
    if expected_fp is not None:
        if stored_fp != expected_fp:
            logger.info(
                "Stage checkpoint compatibility_fingerprint mismatch for stage=%s session=%s",
                stage,
                session_id,
            )
            return None
        expected_full_digest = f"sha256:{expected_digest}"
        if stored_digest is not None and stored_digest != expected_full_digest:
            logger.info(
                "Stage checkpoint fingerprint_digest mismatch for stage=%s session=%s",
                stage,
                session_id,
            )
            return None
        if (
            isinstance(expected_fp, dict)
            and "fingerprint_version" in expected_fp
            and "fingerprint_version" in payload
            and payload.get("fingerprint_version") != expected_fp["fingerprint_version"]
        ):
            logger.info(
                "Stage checkpoint fingerprint_version mismatch for stage=%s session=%s: "
                "expected %s, got %s",
                stage,
                session_id,
                expected_fp["fingerprint_version"],
                payload.get("fingerprint_version"),
            )
            return None
    elif stored_fp is not None or stored_digest is not None:
        logger.info(
            "Stage checkpoint compatibility_fingerprint present for stage=%s session=%s "
            "without expected fingerprint; treating as miss",
            stage,
            session_id,
        )
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning(
            "Stage checkpoint data field invalid or missing for stage=%s session=%s: got %s",
            stage,
            session_id,
            type(data).__name__,
        )
        return None

    logger.info(
        "Stage checkpoint hit for stage=%s session=%s key=%s",
        stage,
        session_id,
        key,
    )
    return data


async def save_stage_checkpoint(
    storage: ObjectStorageProtocol,
    session_id: str,
    stage: str,
    input_checksum: str,
    data: dict[str, Any] | BaseModel,
    schema_version: int = CHECKPOINT_SCHEMA_VERSION,
    producer_version: str = DEFAULT_PRODUCER_VERSION,
    compatibility_fingerprint: CompatibilityFingerprintInput | None = None,
) -> ArtifactRef:
    """Construct and upload an intermediate stage checkpoint payload to object storage.

    Uploads to ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json
    (or {stage}_{clean_hash}_{fp_digest}.json when compatibility_fingerprint is supplied)
    using storage.upload_json(...) and returns the resulting ArtifactRef.
    """
    key = _checkpoint_key(
        session_id,
        stage,
        input_checksum,
        compatibility_fingerprint=compatibility_fingerprint,
    )
    data_dict = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    normalized_fp, fp_digest = normalize_compatibility_fingerprint(compatibility_fingerprint)

    payload: dict[str, Any] = {
        "stage": stage,
        "schema_version": schema_version,
        "producer_version": producer_version,
        "input_checksum": input_checksum,
        "created_at": datetime.now(UTC).isoformat(),
        "data": data_dict,
    }
    if normalized_fp is not None and fp_digest is not None:
        if isinstance(compatibility_fingerprint, StageCheckpointFingerprint):
            fp_version = compatibility_fingerprint.fingerprint_version
        elif isinstance(normalized_fp, dict) and isinstance(
            normalized_fp.get("fingerprint_version"), int
        ):
            fp_version = normalized_fp["fingerprint_version"]
        else:
            fp_version = CHECKPOINT_FINGERPRINT_VERSION
        payload["fingerprint_version"] = fp_version
        payload["compatibility_fingerprint"] = normalized_fp
        payload["fingerprint_digest"] = f"sha256:{fp_digest}"

    ref = await storage.upload_json(key, payload)
    logger.info(
        "Saved stage checkpoint for stage=%s session=%s key=%s artifact_id=%s",
        stage,
        session_id,
        key,
        ref.artifact_id,
    )
    return ref


__all__ = [
    "CHECKPOINT_FINGERPRINT_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_PRODUCER_VERSION",
    "CompatibilityFingerprintInput",
    "StageCheckpointFingerprint",
    "build_stage_compatibility_fingerprint",
    "compute_fingerprint_digest",
    "extract_provider_fingerprint",
    "get_stage_checkpoint",
    "normalize_compatibility_fingerprint",
    "save_stage_checkpoint",
]
