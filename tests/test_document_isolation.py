"""Isolated, budgeted document parsing: timeout, crash, cancellation and size budgets."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.contracts import AssetInput
from app.document_parser import (
    TRUNCATED_CHUNK_LIMIT,
    TRUNCATED_PAGE_LIMIT,
    TRUNCATED_TEXT_LIMIT,
    DocumentExtractionLimits,
)
from app.providers.document_errors import DocumentExtractionTimeout, DocumentParserCrashed
from app.providers.pymupdf_documents import PyMuPDFDocumentProvider
from app.stages.documents import run_document_stage

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
SAMPLE_PDF = FIXTURES / "sample_2page.pdf"
SAMPLE_PPTX = FIXTURES / "sample_3slide.pptx"
HANG = (sys.executable, "-c", "import time; time.sleep(60)")
CRASH = (sys.executable, "-c", "import os; os._exit(3)")
FAKE_SHA256 = "sha256:" + "a" * 64


def _doc_asset(path: Path, artifact_id: str) -> AssetInput:
    return AssetInput(
        artifact_id=artifact_id,
        object_key=str(path),
        checksum=FAKE_SHA256,
        media_type="application/pdf",
    )


async def test_isolated_extraction_matches_in_process_with_provenance() -> None:
    isolated = await PyMuPDFDocumentProvider().extract_and_embed(
        SAMPLE_PDF, asset_version_id="doc1"
    )
    in_process = await PyMuPDFDocumentProvider(isolate=False).extract_and_embed(
        SAMPLE_PDF, asset_version_id="doc1"
    )
    assert [c.model_dump() for c in isolated] == [c.model_dump() for c in in_process]
    assert {c.page_or_slide for c in isolated} == {1, 2}
    assert all(c.extraction_method.startswith("pymupdf/") for c in isolated)


async def test_parser_timeout_kills_child() -> None:
    provider = PyMuPDFDocumentProvider(
        parser_command=HANG, limits=DocumentExtractionLimits(timeout_seconds=0.5)
    )
    with pytest.raises(DocumentExtractionTimeout):
        await provider.extract_and_embed(SAMPLE_PDF, asset_version_id="doc1")
    assert provider.last_parser_process is not None
    assert provider.last_parser_process.returncode is not None  # killed and reaped


async def test_child_crash_is_typed_and_worker_stays_usable() -> None:
    provider = PyMuPDFDocumentProvider(parser_command=CRASH)
    with pytest.raises(DocumentParserCrashed):
        await provider.extract_and_embed(SAMPLE_PDF, asset_version_id="doc1")

    # The next job in the same worker process still extracts normally.
    healthy = PyMuPDFDocumentProvider()
    chunks = await healthy.extract_and_embed(SAMPLE_PDF, asset_version_id="doc2")
    assert chunks


async def test_cancellation_stops_the_parser() -> None:
    provider = PyMuPDFDocumentProvider(
        parser_command=HANG, limits=DocumentExtractionLimits(timeout_seconds=30)
    )
    task = asyncio.create_task(provider.extract_and_embed(SAMPLE_PDF, asset_version_id="doc1"))
    for _ in range(100):
        if provider.last_parser_process is not None:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.last_parser_process is not None
    assert provider.last_parser_process.returncode is not None


@pytest.mark.parametrize("isolate", [True, False])
async def test_page_budget_truncates_and_records_reason(isolate: bool) -> None:
    provider = PyMuPDFDocumentProvider(
        isolate=isolate, limits=DocumentExtractionLimits(max_pages=1)
    )
    chunks = await provider.extract_and_embed(SAMPLE_PDF, asset_version_id="doc1")
    assert chunks and {c.page_or_slide for c in chunks} == {1}
    assert provider.truncation_reasons["doc1"] == [TRUNCATED_PAGE_LIMIT]


async def test_text_and_chunk_budgets_bound_oversized_expansion() -> None:
    text_bound = PyMuPDFDocumentProvider(
        isolate=False,
        chunk_size=20,
        chunk_overlap=0,
        limits=DocumentExtractionLimits(max_text_chars=30),
    )
    chunks = await text_bound.extract_and_embed(SAMPLE_PPTX, asset_version_id="deck")
    assert sum(len(c.text) for c in chunks) <= 30
    assert TRUNCATED_TEXT_LIMIT in text_bound.truncation_reasons["deck"]

    chunk_bound = PyMuPDFDocumentProvider(
        isolate=False,
        chunk_size=10,
        chunk_overlap=0,
        limits=DocumentExtractionLimits(max_chunks=2),
    )
    chunks = await chunk_bound.extract_and_embed(SAMPLE_PDF, asset_version_id="doc1")
    assert len(chunks) == 2
    assert TRUNCATED_CHUNK_LIMIT in chunk_bound.truncation_reasons["doc1"]


async def test_document_stage_reports_safe_typed_limitations() -> None:
    truncating = PyMuPDFDocumentProvider(limits=DocumentExtractionLimits(max_pages=1))
    result = await run_document_stage([_doc_asset(SAMPLE_PDF, "doc1")], truncating)
    assert [lim.code for lim in result.limitations] == ["document_truncated"]
    assert result.chunks

    hanging = PyMuPDFDocumentProvider(
        parser_command=HANG, limits=DocumentExtractionLimits(timeout_seconds=0.5)
    )
    result = await run_document_stage([_doc_asset(SAMPLE_PDF, "doc2")], hanging)
    assert [lim.code for lim in result.limitations] == ["document_extraction_timeout"]

    crashing = PyMuPDFDocumentProvider(parser_command=CRASH)
    result = await run_document_stage([_doc_asset(SAMPLE_PDF, "doc3")], crashing)
    assert [lim.code for lim in result.limitations] == ["document_parser_crashed"]
    assert all("sample_2page" not in lim.message for lim in result.limitations)


def test_limits_from_env_and_validation() -> None:
    limits = DocumentExtractionLimits.from_env(
        {"AI_DOC_PARSE_TIMEOUT_SECONDS": "5", "AI_DOC_MAX_PAGES": "10", "AI_DOC_MAX_MEMORY_MB": "0"}
    )
    assert limits.timeout_seconds == 5 and limits.max_pages == 10
    assert limits.max_memory_mb is None
    with pytest.raises(ValueError):
        DocumentExtractionLimits(timeout_seconds=0)
    with pytest.raises(ValueError):
        DocumentExtractionLimits.from_env({"AI_DOC_MAX_PAGES": "lots"})
