"""S3-compatible object storage adapter for VirtuJudge AI-ML.

Supports Cloudflare R2, AWS S3, and MinIO with path-style addressing and SigV4.
Executes blocking boto3 network operations in thread pools (asyncio.to_thread).
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import botocore.exceptions  # type: ignore[import-untyped]
import ulid
from botocore.client import BaseClient  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from pydantic import BaseModel

from app.contracts import ArtifactRef
from app.storage.base import (
    ObjectNotFoundError,
    ObjectStorageError,
    S3StorageConfig,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)


class S3ObjectStorage:
    """Async wrapper around boto3 S3 client for Cloudflare R2, AWS S3, and MinIO."""

    def __init__(
        self,
        config: S3StorageConfig | None = None,
        *,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region_name: str = "auto",
        client: BaseClient | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif endpoint_url and bucket and access_key and secret_key:
            self.config = S3StorageConfig(
                endpoint_url=endpoint_url,
                bucket=bucket,
                access_key=access_key,
                secret_key=secret_key,
                region_name=region_name,
            )
        else:
            raise ValueError(
                "Either a valid S3StorageConfig or all credentials "
                "(endpoint_url, bucket, access_key, secret_key) must be provided."
            )

        self.endpoint_url = self.config.endpoint_url
        self.bucket = self.config.bucket
        self.access_key = self.config.access_key
        self.secret_key = self.config.secret_key
        self.region_name = self.config.region_name
        self._custom_client = client
        self._client: BaseClient | None = client

    def _get_client(self) -> BaseClient:
        if self._client is not None:
            return self._client

        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5.0,
            read_timeout=30.0,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
            config=config,
        )
        return self._client

    async def download_file(self, object_key: str, destination: Path) -> Path:
        """Download an object from S3/R2 to local disk."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        def _download_sync() -> None:
            client = self._get_client()
            try:
                client.download_file(self.bucket, object_key, str(destination))
            except botocore.exceptions.ClientError as err:
                code = err.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    msg = f"Object '{object_key}' not found in bucket '{self.bucket}'."
                    raise ObjectNotFoundError(msg) from err
                raise ObjectStorageError(f"Failed to download '{object_key}': {err}") from err
            except Exception as exc:
                raise ObjectStorageError(f"Failed to download '{object_key}': {exc}") from exc

        await asyncio.to_thread(_download_sync)
        return destination

    async def upload_file(
        self,
        object_key: str,
        source_path: Path,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        """Upload a local file to S3/R2 and compute its sha256 checksum."""
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file '{source_path}' does not exist.")

        art_id = artifact_id or ulid.new().str

        def _upload_sync() -> str:
            checksum = compute_file_sha256(source_path)
            client = self._get_client()
            try:
                client.upload_file(
                    str(source_path),
                    self.bucket,
                    object_key,
                    ExtraArgs={"ContentType": content_type},
                )
            except Exception as exc:
                raise ObjectStorageError(f"Failed to upload '{object_key}': {exc}") from exc
            return checksum

        checksum = await asyncio.to_thread(_upload_sync)

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
        """Serialize data to JSON and upload to S3/R2."""
        if isinstance(data, BaseModel):
            json_str = data.model_dump_json(indent=2)
        else:
            json_str = json.dumps(data, indent=2)

        raw_bytes = json_str.encode("utf-8")
        checksum = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        art_id = artifact_id or ulid.new().str

        def _upload_sync() -> None:
            client = self._get_client()
            try:
                client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=raw_bytes,
                    ContentType="application/json",
                )
            except Exception as exc:
                raise ObjectStorageError(f"Failed to upload JSON '{object_key}': {exc}") from exc

        await asyncio.to_thread(_upload_sync)

        return ArtifactRef(
            artifact_id=art_id,
            object_key=object_key,
            checksum=checksum,
            schema_version=1,
        )

    async def read_json(self, object_key: str) -> dict[str, Any]:
        """Download and deserialize a JSON object from S3/R2."""
        def _read_sync() -> bytes:
            client = self._get_client()
            try:
                res = client.get_object(Bucket=self.bucket, Key=object_key)
                body = res.get("Body")
                return body.read() if body is not None else b""
            except botocore.exceptions.ClientError as err:
                code = err.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    msg = f"Object '{object_key}' not found in bucket '{self.bucket}'."
                    raise ObjectNotFoundError(msg) from err
                raise ObjectStorageError(f"Failed to read '{object_key}': {err}") from err
            except Exception as exc:
                raise ObjectStorageError(f"Failed to read '{object_key}': {exc}") from exc

        raw_bytes = await asyncio.to_thread(_read_sync)
        data: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
        return data

    async def delete_object(self, object_key: str) -> None:
        """Delete an object from S3/R2."""
        def _delete_sync() -> None:
            client = self._get_client()
            try:
                client.delete_object(Bucket=self.bucket, Key=object_key)
            except botocore.exceptions.ClientError as err:
                code = err.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    return
                raise ObjectStorageError(f"Failed to delete '{object_key}': {err}") from err
            except Exception as exc:
                raise ObjectStorageError(f"Failed to delete '{object_key}': {exc}") from exc

        await asyncio.to_thread(_delete_sync)

    async def delete_prefix(
        self, prefix: str, exclude_suffixes: list[str] | None = None
    ) -> int:
        """Delete all objects matching prefix except those ending with any exclude_suffixes.

        Uses delete_objects batch API (up to 1000 keys per batch) and returns deleted count.
        """
        keys = await self.list_objects(prefix)
        suffixes = exclude_suffixes or []
        keys_to_delete = [
            k for k in keys if not any(k.endswith(suffix) for suffix in suffixes)
        ]
        if not keys_to_delete:
            return 0

        def _delete_all_sync() -> int:
            client = self._get_client()
            chunk_size = 1000
            deleted_count = 0
            for i in range(0, len(keys_to_delete), chunk_size):
                batch = keys_to_delete[i : i + chunk_size]
                delete_dict = {
                    "Objects": [{"Key": k} for k in batch],
                    "Quiet": True,
                }
                try:
                    res = client.delete_objects(Bucket=self.bucket, Delete=delete_dict)
                    if isinstance(res, dict) and res.get("Errors"):
                        raise ObjectStorageError(
                            f"Failed to delete some objects in S3: {res['Errors']}"
                        )
                    deleted_count += len(batch)
                except botocore.exceptions.ClientError as err:
                    raise ObjectStorageError(
                        f"Failed to delete batch for prefix '{prefix}': {err}"
                    ) from err
                except ObjectStorageError:
                    raise
                except Exception as exc:
                    raise ObjectStorageError(
                        f"Failed to delete batch for prefix '{prefix}': {exc}"
                    ) from exc
            return deleted_count

        return await asyncio.to_thread(_delete_all_sync)

    async def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching the given prefix using list_objects_v2 paginator."""
        clean_prefix = prefix.lstrip("/")

        def _list_sync() -> list[str]:
            client = self._get_client()
            keys: list[str] = []
            try:
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket, Prefix=clean_prefix):
                    contents = page.get("Contents") or []
                    for obj in contents:
                        key = obj.get("Key")
                        if key:
                            keys.append(key)
            except botocore.exceptions.ClientError as err:
                raise ObjectStorageError(
                    f"Failed to list objects with prefix '{prefix}': {err}"
                ) from err
            except Exception as exc:
                raise ObjectStorageError(
                    f"Failed to list objects with prefix '{prefix}': {exc}"
                ) from exc
            return sorted(keys)

        return await asyncio.to_thread(_list_sync)


__all__ = [
    "ObjectNotFoundError",
    "ObjectStorageError",
    "S3ObjectStorage",
    "S3StorageConfig",
]

