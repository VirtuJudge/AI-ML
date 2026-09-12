"""Object storage package initialization and factory function."""

import os
from pathlib import Path

from app.storage.base import ObjectStorageProtocol
from app.storage.local import LocalDiskObjectStorage
from app.storage.s3 import ObjectNotFoundError, ObjectStorageError, S3ObjectStorage


def create_object_storage(
    *,
    endpoint_url: str | None = None,
    bucket: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region_name: str | None = None,
    base_dir: Path | str | None = None,
) -> ObjectStorageProtocol:
    """Instantiate S3ObjectStorage if credentials are provided, otherwise LocalDiskObjectStorage.

    Checks environment variables:
    - OBJECT_STORAGE_ENDPOINT
    - OBJECT_STORAGE_BUCKET
    - OBJECT_STORAGE_ACCESS_KEY
    - OBJECT_STORAGE_SECRET_KEY
    - OBJECT_STORAGE_REGION
    """
    endpoint = endpoint_url or os.getenv("OBJECT_STORAGE_ENDPOINT", "").strip()
    bkt = bucket or os.getenv("OBJECT_STORAGE_BUCKET", "").strip()
    ak = access_key or os.getenv("OBJECT_STORAGE_ACCESS_KEY", "").strip()
    sk = secret_key or os.getenv("OBJECT_STORAGE_SECRET_KEY", "").strip()
    region = region_name or os.getenv("OBJECT_STORAGE_REGION", "auto").strip()

    is_configured = bool(
        endpoint
        and bkt
        and ak
        and sk
        and ak != "change-me"
        and sk != "change-me"
        and not endpoint.startswith("http://localhost")
    )

    if is_configured:
        return S3ObjectStorage(
            endpoint_url=endpoint,
            bucket=bkt,
            access_key=ak,
            secret_key=sk,
            region_name=region,
        )

    return LocalDiskObjectStorage(base_dir=base_dir)


__all__ = [
    "LocalDiskObjectStorage",
    "ObjectNotFoundError",
    "ObjectStorageError",
    "ObjectStorageProtocol",
    "S3ObjectStorage",
    "create_object_storage",
]

