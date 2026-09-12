"""Object storage port and protocol definitions for VirtuJudge AI-ML.

Defines structural interfaces (typing.Protocol) for object storage adapters,
enabling swappable backends (Cloudflare R2, AWS S3, MinIO, or local disk).
"""

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from app.contracts import ArtifactRef


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


__all__ = ["ObjectStorageProtocol"]

