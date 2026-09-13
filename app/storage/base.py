"""Object storage port and protocol definitions for VirtuJudge AI-ML.

Defines structural interfaces (typing.Protocol) for object storage adapters,
enabling swappable backends (Cloudflare R2, AWS S3, MinIO, or local disk).
"""

import hashlib
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from app.contracts import ArtifactRef


class ObjectStorageError(Exception):
    """Base exception for object storage errors."""


class ObjectNotFoundError(ObjectStorageError):
    """Raised when an object key is not found in storage."""


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
    ) -> ArtifactRef:
        """Upload a local file to object storage and return its ArtifactRef."""
        ...

    async def upload_json(
        self,
        object_key: str,
        data: dict[str, Any] | BaseModel,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        """Serialize data to JSON, upload to object storage, and return its ArtifactRef."""
        ...

    async def read_json(self, object_key: str) -> dict[str, Any]:
        """Download and deserialize a JSON object from storage."""
        ...

    async def delete_object(self, object_key: str) -> None:
        """Delete an object from storage."""
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


__all__ = [
    "ObjectNotFoundError",
    "ObjectStorageError",
    "ObjectStorageProtocol",
    "S3StorageConfig",
    "compute_file_sha256",
]


