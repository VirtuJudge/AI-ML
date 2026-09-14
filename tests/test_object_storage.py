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


@pytest.mark.asyncio
async def test_local_disk_storage_list_and_delete_prefix(tmp_path: Path) -> None:
    """Verify list_objects and delete_prefix with exclusion filter on LocalDiskObjectStorage."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage_root")

    # Create test objects under ai/session/01JTEST/
    await storage.upload_json("ai/session/01JTEST/analysis.json", {"type": "analysis"})
    await storage.upload_json("ai/session/01JTEST/checkpoints/speech_abc.json", {"type": "ckpt"})
    await storage.upload_json("ai/session/01JTEST/evaluation.json", {"type": "eval"})
    await storage.upload_json("ai/session/01JTEST/report.md", {"type": "report"})
    await storage.upload_json("ai/session/OTHER_SESSION/analysis.json", {"type": "other"})

    # Test list_objects
    keys = await storage.list_objects("ai/session/01JTEST")
    assert len(keys) == 4
    assert "ai/session/01JTEST/analysis.json" in keys
    assert "ai/session/01JTEST/evaluation.json" in keys
    assert "ai/session/01JTEST/report.md" in keys

    # Delete with exclude_suffixes
    deleted_count = await storage.delete_prefix(
        "ai/session/01JTEST",
        exclude_suffixes=["report.md", "evaluation.json"],
    )
    assert deleted_count == 2

    # Verify preserved
    remaining = await storage.list_objects("ai/session/01JTEST")
    assert remaining == [
        "ai/session/01JTEST/evaluation.json",
        "ai/session/01JTEST/report.md",
    ]
    # Verify other session untouched
    other = await storage.list_objects("ai/session/OTHER_SESSION")
    assert other == ["ai/session/OTHER_SESSION/analysis.json"]

    # Delete without exclude_suffixes deletes everything under prefix
    del_all = await storage.delete_prefix("ai/session/01JTEST")
    assert del_all == 2
    assert await storage.list_objects("ai/session/01JTEST") == []


@pytest.mark.asyncio
async def test_s3_storage_list_and_delete_prefix() -> None:
    """Verify list_objects and delete_prefix with exclusion on S3ObjectStorage."""
    mock_boto_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "ai/session/01J/analysis.json"},
                {"Key": "ai/session/01J/evaluation.json"},
                {"Key": "ai/session/01J/report.md"},
            ]
        }
    ]
    mock_boto_client.get_paginator.return_value = mock_paginator

    mock_boto_client.delete_objects.return_value = {
        "Deleted": [{"Key": "ai/session/01J/analysis.json"}]
    }

    storage = S3ObjectStorage(
        endpoint_url="https://mock.r2.cloudflarestorage.com",
        bucket="test-bucket",
        access_key="mock-key",
        secret_key="mock-secret",
        client=mock_boto_client,
    )

    keys = await storage.list_objects("ai/session/01J")
    assert len(keys) == 3

    deleted = await storage.delete_prefix(
        "ai/session/01J",
        exclude_suffixes=["report.md", "evaluation.json"],
    )
    assert deleted == 1
    mock_boto_client.delete_objects.assert_called_once_with(
        Bucket="test-bucket",
        Delete={"Objects": [{"Key": "ai/session/01J/analysis.json"}], "Quiet": True},
    )


@pytest.mark.asyncio
async def test_local_disk_storage_directory_cleanup(tmp_path: Path) -> None:
    """Verify delete_prefix cleans up empty directories and preserves base_dir."""
    storage_root = tmp_path / "cleanup_test"
    storage = LocalDiskObjectStorage(base_dir=storage_root)

    await storage.upload_json("sess1/nested/deep/temp.json", {"k": "v"})
    await storage.upload_json("sess1/report.md", {"k": "v"})

    deep_dir = storage_root / "sess1" / "nested" / "deep"
    assert deep_dir.is_dir()

    # Delete with exclude_suffixes leaving report.md
    deleted = await storage.delete_prefix("sess1", exclude_suffixes=["report.md"])
    assert deleted == 1
    assert not deep_dir.exists()
    assert not (storage_root / "sess1" / "nested").exists()
    assert (storage_root / "sess1").is_dir()
    assert (storage_root / "sess1" / "report.md").is_file()

    # Now delete report.md too
    deleted_all = await storage.delete_prefix("sess1")
    assert deleted_all == 1
    assert not (storage_root / "sess1").exists()
    assert storage_root.is_dir()


@pytest.mark.asyncio
async def test_local_disk_storage_path_traversal_prefix(tmp_path: Path) -> None:
    """Verify list_objects rejects path traversal in prefix."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    with pytest.raises(ValueError, match="Path traversal detected"):
        await storage.list_objects("../../escape")


@pytest.mark.asyncio
async def test_local_disk_storage_delete_prefix_nonexistent(tmp_path: Path) -> None:
    """Verify delete_prefix on nonexistent prefix returns 0 without error."""
    storage = LocalDiskObjectStorage(base_dir=tmp_path / "storage")
    deleted = await storage.delete_prefix("nonexistent/prefix")
    assert deleted == 0


@pytest.mark.asyncio
async def test_s3_storage_delete_prefix_chunking() -> None:
    """Verify delete_prefix batches requests exceeding 1000 items in S3ObjectStorage."""
    mock_boto_client = MagicMock()
    mock_paginator = MagicMock()
    keys = [{"Key": f"logs/item_{i:04d}.log"} for i in range(1500)]
    mock_paginator.paginate.return_value = [{"Contents": keys}]
    mock_boto_client.get_paginator.return_value = mock_paginator

    storage = S3ObjectStorage(
        endpoint_url="https://mock.r2.cloudflarestorage.com",
        bucket="test-bucket",
        access_key="mock-key",
        secret_key="mock-secret",
        client=mock_boto_client,
    )

    deleted = await storage.delete_prefix("logs/")
    assert deleted == 1500
    assert mock_boto_client.delete_objects.call_count == 2



