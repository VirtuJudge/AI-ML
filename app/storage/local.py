"""Local disk object storage adapter for VirtuJudge AI-ML testing and offline runs."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import ulid
from pydantic import BaseModel

from app.contracts import ArtifactRef
from app.storage.s3 import ObjectNotFoundError


class LocalDiskObjectStorage:
    """Stores and retrieves objects from a local directory, computing real SHA-256 checksums."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir or ".storage").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, object_key: str) -> Path:
        clean_key = object_key.lstrip("/\\")
        return self.base_dir / clean_key

    async def download_file(self, object_key: str, destination: Path) -> Path:
        source = self._resolve_path(object_key)
        if not source.is_file():
            raise ObjectNotFoundError(f"Local object '{object_key}' not found at {source}.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    async def upload_file(
        self,
        object_key: str,
        source_path: Path,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file '{source_path}' does not exist.")

        dest = self._resolve_path(object_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, dest)

        data = dest.read_bytes()
        checksum = "sha256:" + hashlib.sha256(data).hexdigest()
        art_id = artifact_id or ulid.new().str

        return ArtifactRef(
            artifact_id=art_id,
            object_key=object_key,
            checksum=checksum,
            schema_version=1,
        )

    async def upload_json(
        self,
        object_key: str,
        data: dict[str, Any] | BaseModel,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        if isinstance(data, BaseModel):
            json_str = data.model_dump_json(indent=2)
        else:
            json_str = json.dumps(data, indent=2)

        raw_bytes = json_str.encode("utf-8")
        checksum = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        art_id = artifact_id or ulid.new().str

        dest = self._resolve_path(object_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw_bytes)

        return ArtifactRef(
            artifact_id=art_id,
            object_key=object_key,
            checksum=checksum,
            schema_version=1,
        )

    async def read_json(self, object_key: str) -> dict[str, Any]:
        path = self._resolve_path(object_key)
        if not path.is_file():
            raise ObjectNotFoundError(f"Local object '{object_key}' not found at {path}.")
        return json.loads(path.read_text(encoding="utf-8"))

    async def delete_object(self, object_key: str) -> None:
        path = self._resolve_path(object_key)
        if path.is_file():
            path.unlink()


__all__ = ["LocalDiskObjectStorage"]
