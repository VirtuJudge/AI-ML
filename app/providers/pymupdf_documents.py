"""PyMuPDF and python-pptx document extraction provider for VirtuJudge AI-ML.

Extracts text from PDF and PPTX documents with page/slide provenance, splits into
versioned chunks, and optionally generates text embeddings via an EmbeddingProvider.
Treats all document text strictly as untrusted data.

Native parsing runs in a separate, killable child process (``python -m
app.document_parser``) with a wall-clock deadline, a memory cap where the OS supports
it, and page/slide, extracted-text and chunk budgets. A parser crash, hang or runaway
document therefore cannot take down or monopolise the worker, and cancelling the job
stops the parser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.document_parser import (
    DocumentExtractionLimits,
    ParserError,
    extract_document,
)
from app.providers.base import EmbeddingProvider
from app.providers.document_errors import (
    DocumentExtractionError,
    DocumentExtractionTimeout,
    DocumentParserCrashed,
    DocumentResourceLimitExceeded,
)
from app.providers.types import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
CHUNKING_VERSION = "v1_512_64"

_REAP_TIMEOUT_SECONDS = 5.0
DEFAULT_PARSER_COMMAND: tuple[str, ...] = (sys.executable, "-m", "app.document_parser")


class PyMuPDFDocumentProvider:
    """Document extraction provider powered by PyMuPDF (PDF) and python-pptx (PPTX)."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        chunking_version: str = CHUNKING_VERSION,
        limits: DocumentExtractionLimits | None = None,
        isolate: bool = True,
        parser_command: Sequence[str] | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_version = chunking_version
        self.limits = limits if limits is not None else DocumentExtractionLimits.from_env()
        self.isolate = isolate
        # Overridable for deterministic timeout/crash tests.
        self.parser_command: tuple[str, ...] = tuple(parser_command or DEFAULT_PARSER_COMMAND)
        # Budget truncation reasons per asset_version_id, consumed by the document stage.
        self.truncation_reasons: dict[str, list[str]] = {}
        self.last_parser_process: asyncio.subprocess.Process | None = None

    def _request(self, doc_path: Path, asset_version_id: str) -> dict[str, Any]:
        return {
            "doc_path": str(doc_path),
            "asset_version_id": asset_version_id,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_version": self.chunking_version,
            "limits": self.limits.to_dict(),
        }

    def _extract_in_process(
        self, doc_path: Path, asset_version_id: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            return extract_document(
                doc_path,
                asset_version_id=asset_version_id,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunking_version=self.chunking_version,
                limits=self.limits,
            )
        except MemoryError as exc:
            raise DocumentResourceLimitExceeded(
                "Document parser exceeded its memory budget."
            ) from exc
        except ParserError as exc:
            raise DocumentExtractionError(str(exc)) from exc

    async def _extract_isolated(
        self, doc_path: Path, asset_version_id: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Run the parser in a child process; kill and reap it on timeout or cancellation."""
        request = json.dumps(self._request(doc_path, asset_version_id)).encode("utf-8")
        process = await asyncio.create_subprocess_exec(
            *self.parser_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.last_parser_process = process
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(request), timeout=self.limits.timeout_seconds
            )
        except TimeoutError as exc:
            raise DocumentExtractionTimeout(
                "Document parser exceeded its time limit and was stopped."
            ) from exc
        finally:
            # On timeout, cancellation or any error the child is killed so it cannot keep
            # parsing; it is always awaited so no zombie process is left behind.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_SECONDS)

        try:
            result = json.loads(stdout.decode("utf-8")) if stdout else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            result = None
        if not isinstance(result, dict) or process.returncode != 0:
            raise DocumentParserCrashed(
                f"Document parser exited unexpectedly (exit code {process.returncode})."
            )
        if result.get("status") == "ok":
            return list(result.get("chunks", [])), list(result.get("truncation_reasons", []))
        message = str(result.get("message") or "Document parser failed.")
        if result.get("kind") == "resource":
            raise DocumentResourceLimitExceeded(message)
        raise DocumentExtractionError(message)

    async def extract_and_embed(
        self, doc_path: Path, *, asset_version_id: str = ""
    ) -> list[DocumentChunk]:
        """Extract chunks from PDF/PPTX in an isolated process and embed if a provider is set."""
        doc_path = Path(doc_path)
        resolved_version = asset_version_id or doc_path.stem
        if not doc_path.is_file():
            raise FileNotFoundError(f"Document file not found: {doc_path}")

        if self.isolate:
            records, reasons = await self._extract_isolated(doc_path, resolved_version)
        else:
            records, reasons = await asyncio.to_thread(
                self._extract_in_process, doc_path, resolved_version
            )
        if reasons:
            self.truncation_reasons[resolved_version] = reasons

        chunks = [DocumentChunk.model_validate(record) for record in records]
        if not chunks:
            return []

        if self.embedding_provider is not None:
            texts = [chunk.text for chunk in chunks]
            try:
                embeddings = await self.embedding_provider.embed(texts)
                model_name = getattr(self.embedding_provider, "model_name", None)
                for chunk, emb in zip(chunks, embeddings, strict=True):
                    chunk.embedding = emb
                    chunk.embedding_dimensions = len(emb)
                    chunk.embedding_model = model_name
            except Exception as exc:
                # Log safe identifier only, never the host file path or document content
                logger.warning(
                    "Embedding generation failed for asset_version_id=%s: %s",
                    resolved_version,
                    exc,
                )
                raise DocumentExtractionError(f"Embedding generation failed: {exc}") from exc

        return chunks


__all__ = [
    "CHUNKING_VERSION",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_PARSER_COMMAND",
    "DocumentExtractionError",
    "DocumentExtractionLimits",
    "DocumentExtractionTimeout",
    "DocumentParserCrashed",
    "DocumentResourceLimitExceeded",
    "PyMuPDFDocumentProvider",
]
