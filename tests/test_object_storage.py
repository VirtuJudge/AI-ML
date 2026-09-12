"""Unit tests for Object Storage adapters (LocalDiskObjectStorage and S3ObjectStorage)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.storage.local import LocalDiskObjectStorage
from app.storage.s3 import ObjectNotFoundError, S3ObjectStorage


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

