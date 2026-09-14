"""Local disk object storage adapter for VirtuJudge AI-ML testing and offline runs."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import ulid
from pydantic import BaseModel

from app.contracts import ArtifactRef
from app.storage.base import ObjectNotFoundError, compute_file_sha256


class LocalDiskObjectStorage:
    """Stores and retrieves objects from a local directory, computing real SHA-256 checksums."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir or ".storage").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, object_key: str) -> Path:
        clean_key = object_key.lstrip("/\\")
        dest = (self.base_dir / clean_key).resolve()
        if not dest.is_relative_to(self.base_dir.resolve()):
            raise ValueError(f"Path traversal detected in object key: '{object_key}'.")
        return dest

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

        checksum = compute_file_sha256(dest)
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
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    async def delete_object(self, object_key: str) -> None:
        path = self._resolve_path(object_key)
        if path.is_file():
            path.unlink()

    async def delete_prefix(
        self, prefix: str, exclude_suffixes: list[str] | None = None
    ) -> int:
        """Delete all objects matching prefix except those ending with any exclude_suffixes.

        Cleans up empty directories and returns count of deleted objects.
        """

        keys = await self.list_objects(prefix)
        suffixes = exclude_suffixes or []
        to_delete = [
            k for k in keys if not any(k.endswith(suffix) for suffix in suffixes)
        ]

        deleted_count = 0
        for key in to_delete:
            path = self._resolve_path(key)
            if path.is_file():
                path.unlink()
                deleted_count += 1

        # Clean up empty directories bottom-up, stopping before base_dir
        if self.base_dir.exists():
            for dirpath, _, _ in os.walk(self.base_dir, topdown=False):
                d = Path(dirpath)
                if d == self.base_dir:
                    continue
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    pass

        return deleted_count

    async def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching the given prefix."""
        clean_prefix = prefix.lstrip("/\\")
        resolved = (self.base_dir / clean_prefix).resolve()
        if not resolved.is_relative_to(self.base_dir.resolve()):
            raise ValueError(f"Path traversal detected in prefix: '{prefix}'.")

        if not self.base_dir.exists():
            return []

        matched: list[str] = []
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                key = path.relative_to(self.base_dir).as_posix()
                if key.startswith(clean_prefix):
                    matched.append(key)

        return sorted(matched)



__all__ = ["LocalDiskObjectStorage"]
