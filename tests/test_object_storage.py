"""Unit tests for Object Storage adapters (LocalDiskObjectStorage and S3ObjectStorage)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.storage import create_object_storage
from app.storage.base import ObjectNotFoundError, S3StorageConfig
from app.storage.local import LocalDiskObjectStorage
from app.storage.s3 import S3ObjectStorage


@pytest.mark.asyncio
async def test_local_disk_storage_upload_and_read_json(tmp_path: Path) -> None:
    """Verify LocalDiskObjectStorage serializes JSON, writes to disk, and computes sha256."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    sample_data = {
        "status": "ready",
        "question_count": 3,
        "tags": ["pitch", "ai"],
    }

    ref = await storage.upload_json("ai/test/analysis.json", sample_data)

    assert ref.object_key == "ai/test/analysis.json"
    assert len(ref.artifact_id) == 26
    assert ref.checksum.startswith("sha256:")
    assert len(ref.checksum) == 7 + 64

    # Read back and assert equality
    read_data = await storage.read_json("ai/test/analysis.json")
    assert read_data == sample_data


@pytest.mark.asyncio
async def test_local_disk_storage_file_roundtrip(tmp_path: Path) -> None:
    """Verify LocalDiskObjectStorage upload_file and download_file preserve content."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage_root")

    source_file = tmp_path / "test_input.txt"
    source_file.write_text("Hello VirtuJudge Object Storage!", encoding="utf-8")

    ref = await storage.upload_file("inputs/sample.txt", source_file)
    assert ref.object_key == "inputs/sample.txt"
    assert ref.checksum.startswith("sha256:")

    # Download to separate destination
    dest_file = tmp_path / "downloaded.txt"
    downloaded = await storage.download_file("inputs/sample.txt", dest_file)

    assert downloaded == dest_file
    assert dest_file.read_text(encoding="utf-8") == "Hello VirtuJudge Object Storage!"


@pytest.mark.asyncio
async def test_local_disk_storage_not_found(tmp_path: Path) -> None:
    """Verify ObjectNotFoundError is raised when object key does not exist."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path)
    with pytest.raises(ObjectNotFoundError):
        await storage.read_json("non_existent.json")

    with pytest.raises(ObjectNotFoundError):
        await storage.download_file("non_existent.txt", tmp_path / "out.txt")


@pytest.mark.asyncio
async def test_local_disk_storage_path_traversal_rejected(tmp_path: Path) -> None:
    """Verify path traversal attempts raise ValueError and are blocked."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "sandbox")
    with pytest.raises(ValueError, match="Path traversal detected"):
        await storage.read_json("../../secret.json")

    with pytest.raises(ValueError, match="Path traversal detected"):
        await storage.download_file("../../secret.txt", tmp_path / "out.txt")



@pytest.mark.asyncio
async def test_s3_storage_mocked_calls() -> None:
    """Verify S3ObjectStorage invokes boto3 client correctly."""
    mock_boto_client = MagicMock()
    storage = S3ObjectStorage(
        endpoint_url="https://mock.r2.cloudflarestorage.com",
        bucket="test-bucket",
        access_key="mock-key",
        secret_key="mock-secret",
        client=mock_boto_client,
    )

    # Test upload_json
    ref = await storage.upload_json("ai/test.json", {"key": "val"})
    assert ref.object_key == "ai/test.json"
    assert ref.checksum.startswith("sha256:")
    assert mock_boto_client.put_object.called

    call_kwargs = mock_boto_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "ai/test.json"
    assert call_kwargs["ContentType"] == "application/json"


@pytest.mark.asyncio
async def test_s3_storage_upload_file_uses_streaming_upload_file(tmp_path: Path) -> None:
    """Verify S3ObjectStorage.upload_file streams file via client.upload_file without buffering."""
    mock_boto_client = MagicMock()
    storage = S3ObjectStorage(
        endpoint_url="https://mock.r2.cloudflarestorage.com",
        bucket="test-bucket",
        access_key="mock-key",
        secret_key="mock-secret",
        client=mock_boto_client,
    )

    test_file = tmp_path / "video.mp4"
    test_content = b"fake video bytes " * 1024
    test_file.write_bytes(test_content)

    ref = await storage.upload_file(
        "presentations/video.mp4",
        test_file,
        content_type="video/mp4",
        artifact_id="art_video_001",
    )

    assert ref.object_key == "presentations/video.mp4"
    assert ref.artifact_id == "art_video_001"
    assert ref.checksum.startswith("sha256:")

    # Verify client.upload_file was used instead of put_object
    assert mock_boto_client.upload_file.called

    upload_args, upload_kwargs = mock_boto_client.upload_file.call_args
    assert upload_args == (str(test_file), "test-bucket", "presentations/video.mp4")
    assert upload_kwargs.get("ExtraArgs") == {"ContentType": "video/mp4"}


@pytest.mark.asyncio
async def test_s3_storage_config_direct_and_factory() -> None:
    """Verify S3StorageConfig bundles configuration and works with S3ObjectStorage & factory."""
    config = S3StorageConfig(
        endpoint_url="https://mock.r2.cloudflarestorage.com",
        bucket="my-bucket",
        access_key="my-key",
        secret_key="my-secret",
        region_name="auto",
    )
    mock_boto_client = MagicMock()
    storage = S3ObjectStorage(config=config, client=mock_boto_client)
    assert storage.config == config
    assert storage.endpoint_url == "https://mock.r2.cloudflarestorage.com"
    assert storage.bucket == "my-bucket"

    # Factory creates S3ObjectStorage when passed S3StorageConfig
    factory_storage = create_object_storage(config=config)
    assert isinstance(factory_storage, S3ObjectStorage)
    assert factory_storage.bucket == "my-bucket"


def test_s3_storage_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify S3StorageConfig.from_env loads valid env config and rejects invalid/defaults."""
    # Defaults / change-me returns None
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "https://r2.cloudflarestorage.com")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "change-me")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "change-me")
    assert S3StorageConfig.from_env() is None

    # Valid non-default env vars return S3StorageConfig
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "real-access-key")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "real-secret-key")
    cfg = S3StorageConfig.from_env()
    assert cfg is not None
    assert cfg.endpoint_url == "https://r2.cloudflarestorage.com"
    assert cfg.bucket == "bucket"
    assert cfg.access_key == "real-access-key"
    assert cfg.secret_key == "real-secret-key"
    assert cfg.region_name == "auto"


