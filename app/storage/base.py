"""Object storage port and protocol definitions for VirtuJudge AI-ML.

Defines structural interfaces (typing.Protocol) for object storage adapters,
enabling swappable backends (Cloudflare R2, AWS S3, MinIO, or local disk).
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from app.contracts import ArtifactRef, ChecksumMismatchError

logger = logging.getLogger(__name__)


class ObjectStorageError(Exception):
    """Base exception for object storage errors."""


class ObjectNotFoundError(ObjectStorageError):
    """Raised when an object key is not found in storage."""


class ObjectAlreadyExistsError(ObjectStorageError):
    """Raised when a non-overwriting write targets an object key with different content."""


class ObjectStorageProtocol(Protocol):
    """Protocol for reading and writing media and artifacts in object storage."""

    async def download_file(self, object_key: str, destination: Path) -> Path:
        """Download an object from storage to a local file destination."""
        ...

    async def upload_file(
        self,
        object_key: str,
        source_path: Path,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
        *,
        overwrite: bool = True,
    ) -> ArtifactRef:
        """Upload a local file to object storage and return its ArtifactRef."""
        ...

    async def upload_json(
        self,
        object_key: str,
        data: dict[str, Any] | BaseModel,
        artifact_id: str | None = None,
        *,
        overwrite: bool = True,
    ) -> ArtifactRef:
        """Serialize data to JSON, upload to object storage, and return its ArtifactRef."""
        ...

    async def read_json(self, object_key: str) -> dict[str, Any]:
        """Download and deserialize a JSON object from storage."""
        ...

    async def delete_object(self, object_key: str) -> None:
        """Delete an object from storage."""
        ...

    async def delete_prefix(self, prefix: str, exclude_suffixes: list[str] | None = None) -> int:
        """Delete all objects matching prefix except those ending with any exclude_suffixes.

        Return count of deleted objects.
        """
        ...

    async def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching the given prefix."""
        ...


class S3StorageConfig(BaseModel):
    """Configuration bundle for S3-compatible object storage (Cloudflare R2, AWS S3, MinIO)."""

    endpoint_url: str
    bucket: str
    access_key: str
    secret_key: str
    region_name: str = "auto"

    @classmethod
    def from_env(cls) -> "S3StorageConfig | None":
        """Load configuration from environment variables if validly configured."""
        import os

        endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT", "").strip()
        bkt = os.getenv("OBJECT_STORAGE_BUCKET", "").strip()
        ak = os.getenv("OBJECT_STORAGE_ACCESS_KEY", "").strip()
        sk = os.getenv("OBJECT_STORAGE_SECRET_KEY", "").strip()
        region = os.getenv("OBJECT_STORAGE_REGION", "auto").strip() or "auto"

        if (
            endpoint
            and bkt
            and ak
            and sk
            and ak != "change-me"
            and sk != "change-me"
            and not endpoint.startswith("http://localhost")
        ):
            return cls(
                endpoint_url=endpoint,
                bucket=bkt,
                access_key=ak,
                secret_key=sk,
                region_name=region,
            )
        return None


def compute_file_sha256(path: Path, chunk_size: int = 64 * 1024) -> str:
    """Stream file in chunks to compute sha256 checksum without high memory consumption."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def normalize_sha256_checksum(checksum: str) -> str:
    """Normalize a SHA-256 checksum string for case-insensitive comparison."""
    return checksum.strip().lower()


def verify_file_checksum(
    path: Path,
    expected_checksum: str,
    *,
    stage: str = "ingestion",
    artifact_id: str | None = None,
) -> str:
    """Hash a resolved file and verify it matches the declared SHA-256 checksum."""
    actual_checksum = compute_file_sha256(path)
    if normalize_sha256_checksum(actual_checksum) != normalize_sha256_checksum(expected_checksum):
        logger.warning(
            "Input checksum mismatch at stage=%s artifact_id=%s",
            stage,
            artifact_id or "unknown",
        )
        raise ChecksumMismatchError(stage=stage, artifact_id=artifact_id)
    return actual_checksum


__all__ = [
    "ObjectAlreadyExistsError",
    "ObjectNotFoundError",
    "ObjectStorageError",
    "ObjectStorageProtocol",
    "S3StorageConfig",
    "compute_file_sha256",
    "normalize_sha256_checksum",
    "verify_file_checksum",
]
