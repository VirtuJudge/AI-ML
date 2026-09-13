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

import boto3
import botocore.exceptions
import ulid
from botocore.client import BaseClient
from botocore.config import Config
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
        return json.loads(raw_bytes.decode("utf-8"))

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


__all__ = [
    "ObjectNotFoundError",
    "ObjectStorageError",
    "S3ObjectStorage",
    "S3StorageConfig",
]
