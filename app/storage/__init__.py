"""Object storage package initialization and factory function."""

from pathlib import Path

from app.storage.base import (
    ObjectNotFoundError,
    ObjectStorageError,
    ObjectStorageProtocol,
    S3StorageConfig,
)
from app.storage.local import LocalDiskObjectStorage
from app.storage.s3 import S3ObjectStorage


def create_object_storage(
    config: S3StorageConfig | None = None,
    *,
    endpoint_url: str | None = None,
    bucket: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region_name: str | None = None,
    base_dir: Path | str | None = None,
) -> ObjectStorageProtocol:
    """Instantiate S3ObjectStorage if configured, otherwise LocalDiskObjectStorage.

    Accepts an explicit S3StorageConfig bundle, or explicit credentials,
    or resolves configuration via S3StorageConfig.from_env().
    """
    if config is not None:
        return S3ObjectStorage(config=config)

    if endpoint_url and bucket and access_key and secret_key:
        s3_config = S3StorageConfig(
            endpoint_url=endpoint_url,
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            region_name=region_name or "auto",
        )
        return S3ObjectStorage(config=s3_config)

    env_config = S3StorageConfig.from_env()
    if env_config is not None:
        return S3ObjectStorage(config=env_config)

    return LocalDiskObjectStorage(base_dir=base_dir)


__all__ = [
    "LocalDiskObjectStorage",
    "ObjectNotFoundError",
    "ObjectStorageError",
    "ObjectStorageProtocol",
    "S3ObjectStorage",
    "S3StorageConfig",
    "create_object_storage",
]

