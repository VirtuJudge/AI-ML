"""Stage checkpointing helpers for caching intermediate stage artifacts.

Provides helper functions to cache and reuse intermediate stage outputs in
ObjectStorageProtocol, preventing expensive recomputation across pipeline runs.
"""

import logging
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc
from typing import Any

from pydantic import BaseModel

from app.contracts import ArtifactRef
from app.storage.base import ObjectNotFoundError, ObjectStorageProtocol

logger = logging.getLogger(__name__)


def _checkpoint_key(session_id: str, stage: str, input_checksum: str) -> str:
    """Build standardized checkpoint object key for a session stage and checksum."""
    clean_hash = input_checksum.removeprefix("sha256:")
    return f"ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json"


async def get_stage_checkpoint(
    storage: ObjectStorageProtocol,
    session_id: str,
    stage: str,
    input_checksum: str,
    expected_schema_version: int = 1,
) -> dict[str, Any] | None:
    """Read and validate an intermediate stage checkpoint from object storage.

    Reads JSON from the checkpoint key. Verifies schema_version == expected_schema_version
    and input_checksum == input_checksum.
    If valid, returns the saved data dictionary.
    If missing (ObjectNotFoundError), corrupt, or mismatched checksum/version, returns None.
    """
    key = _checkpoint_key(session_id, stage, input_checksum)

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

    if payload.get("schema_version") != expected_schema_version:
        logger.info(
            "Stage checkpoint schema_version mismatch for stage=%s session=%s: "
            "expected %s, got %s",
            stage,
            session_id,
            expected_schema_version,
            payload.get("schema_version"),
        )
        return None

    if payload.get("input_checksum") != input_checksum:
        logger.info(
            "Stage checkpoint input_checksum mismatch for stage=%s session=%s: "
            "expected %s, got %s",
            stage,
            session_id,
            input_checksum,
            payload.get("input_checksum"),
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
    schema_version: int = 1,
    producer_version: str = "ai-ml/0.1.0",
) -> ArtifactRef:
    """Construct and upload an intermediate stage checkpoint payload to object storage.

    Uploads to ai/session/{session_id}/checkpoints/{stage}_{clean_hash}.json
    using storage.upload_json(...) and returns the resulting ArtifactRef.
    """
    key = _checkpoint_key(session_id, stage, input_checksum)
    data_dict = data.model_dump(mode="json") if isinstance(data, BaseModel) else data

    payload: dict[str, Any] = {
        "stage": stage,
        "schema_version": schema_version,
        "producer_version": producer_version,
        "input_checksum": input_checksum,
        "created_at": datetime.now(UTC).isoformat(),
        "data": data_dict,
    }

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
    "get_stage_checkpoint",
    "save_stage_checkpoint",
]
