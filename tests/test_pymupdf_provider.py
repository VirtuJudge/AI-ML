"""Unit tests for app.providers.pymupdf_documents."""

from pathlib import Path

import pytest

from app.providers.base import DocumentProvider
from app.providers.fake_documents import FakeEmbeddingProvider
from app.providers.pymupdf_documents import (
    DocumentExtractionError,
    PyMuPDFDocumentProvider,
)
from app.providers.types import DocumentChunk


@pytest.fixture
def fixtures_doc_dir() -> Path:
    """Fixture pointing to tests/fixtures/documents/ directory."""
    return Path(__file__).parent / "fixtures" / "documents"


def test_pymupdf_provider_satisfies_protocol() -> None:
    """Verify PyMuPDFDocumentProvider satisfies DocumentProvider Protocol."""
    provider: DocumentProvider = PyMuPDFDocumentProvider()
    assert hasattr(provider, "extract_and_embed")
    assert callable(provider.extract_and_embed)


@pytest.mark.asyncio
async def test_pdf_page_extraction(fixtures_doc_dir: Path) -> None:
    """Verify known 2-page PDF extracts expected page content and provenance."""
    pdf_path = fixtures_doc_dir / "sample_2page.pdf"
    if not pdf_path.is_file():
        pytest.skip("sample_2page.pdf fixture not found")

    provider = PyMuPDFDocumentProvider()
    chunks = await provider.extract_and_embed(
        pdf_path, asset_version_id="01JTESTPDF000000000000001"
    )

    assert len(chunks) == 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)

    # Page 1 checks
    p1 = chunks[0]
    assert p1.page_or_slide == 1
    assert "Executive Summary" in p1.text
    assert p1.asset_version_id == "01JTESTPDF000000000000001"
    assert p1.extraction_method.startswith("pymupdf/")
    assert p1.chunking_version == "v1_512_64"

    # Page 2 checks
    p2 = chunks[1]
    assert p2.page_or_slide == 2
    assert "Financial Projections" in p2.text
    assert p2.asset_version_id == "01JTESTPDF000000000000001"


@pytest.mark.asyncio
async def test_pptx_slide_extraction(fixtures_doc_dir: Path) -> None:
    """Verify known 3-slide PPTX extracts expected slide content and provenance."""
    pptx_path = fixtures_doc_dir / "sample_3slide.pptx"
    if not pptx_path.is_file():
        pytest.skip("sample_3slide.pptx fixture not found")

    provider = PyMuPDFDocumentProvider()
    chunks = await provider.extract_and_embed(
        pptx_path, asset_version_id="01JTESTPPT000000000000001"
    )

    assert len(chunks) == 3
    slide_numbers = [c.page_or_slide for c in chunks]
    assert slide_numbers == [1, 2, 3]

    assert "Introduction to VirtuJudge" in chunks[0].text
    assert "Market Opportunity" in chunks[1].text
    assert "Technical Architecture" in chunks[2].text

    for c in chunks:
        assert c.asset_version_id == "01JTESTPPT000000000000001"
        assert c.extraction_method == "python-pptx"


@pytest.mark.asyncio
async def test_chunking_produces_versioned_chunks(fixtures_doc_dir: Path) -> None:
    """Verify long text gets chunked with valid start/end offsets and deterministic IDs."""
    pdf_path = fixtures_doc_dir / "sample_2page.pdf"
    if not pdf_path.is_file():
        pytest.skip("sample_2page.pdf fixture not found")

    # Use very small chunk size to force multiple chunks per page
    provider = PyMuPDFDocumentProvider(chunk_size=30, chunk_overlap=10)
    chunks = await provider.extract_and_embed(
        pdf_path, asset_version_id="01JTESTPDF000000000000002"
    )

    assert len(chunks) > 2
    for c in chunks:
        assert c.start_offset is not None
        assert c.end_offset is not None
        assert c.start_offset < c.end_offset
        assert c.chunk_id.startswith("01JTESTPDF000000000000002_p")


@pytest.mark.asyncio
async def test_embedding_generation_integration(fixtures_doc_dir: Path) -> None:
    """Verify that injecting an EmbeddingProvider populates chunk embeddings."""
    pdf_path = fixtures_doc_dir / "sample_2page.pdf"
    if not pdf_path.is_file():
        pytest.skip("sample_2page.pdf fixture not found")

    embedding_provider = FakeEmbeddingProvider(dimension=64)
    provider = PyMuPDFDocumentProvider(embedding_provider=embedding_provider)
    chunks = await provider.extract_and_embed(
        pdf_path, asset_version_id="01JTESTPDF000000000000003"
    )

    assert len(chunks) == 2
    for c in chunks:
        assert c.embedding is not None
        assert len(c.embedding) == 64
        assert c.embedding_dimensions == 64


@pytest.mark.asyncio
async def test_encrypted_pdf_fails_safely(fixtures_doc_dir: Path) -> None:
    """Verify encrypted PDF raises DocumentExtractionError."""
    enc_path = fixtures_doc_dir / "encrypted.pdf"
    if not enc_path.is_file():
        pytest.skip("encrypted.pdf fixture not found")

    provider = PyMuPDFDocumentProvider()
    with pytest.raises(DocumentExtractionError, match="password-protected or encrypted"):
        await provider.extract_and_embed(enc_path)


@pytest.mark.asyncio
async def test_empty_pdf_returns_empty(fixtures_doc_dir: Path) -> None:
    """Verify empty PDF with no text returns an empty chunk list."""
    empty_path = fixtures_doc_dir / "empty.pdf"
    if not empty_path.is_file():
        pytest.skip("empty.pdf fixture not found")

    provider = PyMuPDFDocumentProvider()
    chunks = await provider.extract_and_embed(empty_path)
    assert chunks == []


@pytest.mark.asyncio
async def test_unsupported_format_raises(tmp_path: Path) -> None:
    """Verify unsupported file format raises DocumentExtractionError."""
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("random notes")

    provider = PyMuPDFDocumentProvider()
    with pytest.raises(DocumentExtractionError, match="Unsupported document format"):
        await provider.extract_and_embed(txt_path)


@pytest.mark.asyncio
async def test_prompt_injection_text_preserved_as_data(fixtures_doc_dir: Path) -> None:
    """Verify prompt-injection document text is extracted literally as untrusted data."""
    injection_path = fixtures_doc_dir / "prompt_injection.pdf"
    if not injection_path.is_file():
        pytest.skip("prompt_injection.pdf fixture not found")

    provider = PyMuPDFDocumentProvider()
    chunks = await provider.extract_and_embed(injection_path)

    assert len(chunks) >= 1
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in chunks[0].text
