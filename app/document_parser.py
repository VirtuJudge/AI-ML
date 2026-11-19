"""Standalone, budgeted PDF/PPTX text extraction for an isolated parser process.

This module deliberately imports nothing from ``app.providers`` (whose package import
pulls in optional ML stacks), so ``python -m app.document_parser`` starts quickly. It
reads one JSON request on stdin and writes one JSON result on stdout:

    request:  {"doc_path", "asset_version_id", "chunk_size", "chunk_overlap",
               "chunking_version", "limits": {...}}
    result:   {"status": "ok", "chunks": [...], "truncation_reasons": [...]}
              {"status": "error", "kind": "extraction" | "resource", "message": "..."}

Error messages are safe summaries and never contain document text. The parent process
(``PyMuPDFDocumentProvider``) owns the deadline and kills this process when needed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOC_TIMEOUT_ENV = "AI_DOC_PARSE_TIMEOUT_SECONDS"
DOC_MAX_PAGES_ENV = "AI_DOC_MAX_PAGES"
DOC_MAX_TEXT_CHARS_ENV = "AI_DOC_MAX_TEXT_CHARS"
DOC_MAX_CHUNKS_ENV = "AI_DOC_MAX_CHUNKS"
DOC_MAX_MEMORY_MB_ENV = "AI_DOC_MAX_MEMORY_MB"

DEFAULT_DOC_TIMEOUT_SECONDS = 60.0
DEFAULT_DOC_MAX_PAGES = 200
DEFAULT_DOC_MAX_TEXT_CHARS = 2_000_000
DEFAULT_DOC_MAX_CHUNKS = 5_000
DEFAULT_DOC_MAX_MEMORY_MB = 1_024

TRUNCATED_PAGE_LIMIT = "page_limit"
TRUNCATED_TEXT_LIMIT = "text_limit"
TRUNCATED_CHUNK_LIMIT = "chunk_limit"


class ParserError(Exception):
    """A safe, content-free parser failure (encrypted, unsupported, unreadable)."""


@dataclass(frozen=True)
class DocumentExtractionLimits:
    """Resource budgets applied to one document extraction."""

    timeout_seconds: float = DEFAULT_DOC_TIMEOUT_SECONDS
    max_pages: int = DEFAULT_DOC_MAX_PAGES
    max_text_chars: int = DEFAULT_DOC_MAX_TEXT_CHARS
    max_chunks: int = DEFAULT_DOC_MAX_CHUNKS
    max_memory_mb: int | None = DEFAULT_DOC_MAX_MEMORY_MB

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Document parse timeout must be a positive finite number.")
        for name in ("max_pages", "max_text_chars", "max_chunks"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Document extraction {name} must be positive.")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("Document extraction max_memory_mb must be positive when set.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DocumentExtractionLimits:
        """Build limits from AI_DOC_* environment variables, falling back to defaults."""
        source = os.environ if env is None else env

        def _num(key: str, default: float) -> float:
            raw = source.get(key, "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as err:
                raise ValueError(f"{key} must be a number.") from err

        memory = _num(DOC_MAX_MEMORY_MB_ENV, DEFAULT_DOC_MAX_MEMORY_MB)
        return cls(
            timeout_seconds=_num(DOC_TIMEOUT_ENV, DEFAULT_DOC_TIMEOUT_SECONDS),
            max_pages=int(_num(DOC_MAX_PAGES_ENV, DEFAULT_DOC_MAX_PAGES)),
            max_text_chars=int(_num(DOC_MAX_TEXT_CHARS_ENV, DEFAULT_DOC_MAX_TEXT_CHARS)),
            max_chunks=int(_num(DOC_MAX_CHUNKS_ENV, DEFAULT_DOC_MAX_CHUNKS)),
            max_memory_mb=int(memory) if memory > 0 else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_pages": self.max_pages,
            "max_text_chars": self.max_text_chars,
            "max_chunks": self.max_chunks,
            "max_memory_mb": self.max_memory_mb,
        }


class _Budget:
    """Tracks page/text/chunk budgets while one document is extracted."""

    def __init__(self, limits: DocumentExtractionLimits) -> None:
        self.limits = limits
        self.text_chars = 0
        self.chunk_count = 0
        self.reasons: list[str] = []

    def _flag(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    @property
    def exhausted(self) -> bool:
        return (
            self.text_chars >= self.limits.max_text_chars
            or self.chunk_count >= self.limits.max_chunks
        )

    def allow_page(self, page_number: int) -> bool:
        """Return False (and record why) when no further page/slide may be read."""
        if page_number > self.limits.max_pages:
            self._flag(TRUNCATED_PAGE_LIMIT)
            return False
        if self.exhausted:
            if self.text_chars >= self.limits.max_text_chars:
                self._flag(TRUNCATED_TEXT_LIMIT)
            if self.chunk_count >= self.limits.max_chunks:
                self._flag(TRUNCATED_CHUNK_LIMIT)
            return False
        return True

    def take_text(self, text: str) -> str:
        remaining = self.limits.max_text_chars - self.text_chars
        if len(text) > remaining:
            self._flag(TRUNCATED_TEXT_LIMIT)
            text = text[: max(0, remaining)]
        self.text_chars += len(text)
        return text

    def take_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = self.limits.max_chunks - self.chunk_count
        if len(chunks) > remaining:
            self._flag(TRUNCATED_CHUNK_LIMIT)
            chunks = chunks[: max(0, remaining)]
        self.chunk_count += len(chunks)
        return chunks


def chunk_text(
    text: str,
    *,
    asset_version_id: str,
    page_or_slide: int,
    extraction_method: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_version: str,
) -> list[dict[str, Any]]:
    """Split page/slide text into versioned, deterministic chunk records."""
    text = text.strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    text_len = len(text)
    step = max(1, chunk_size - chunk_overlap)
    start_offset = 0
    while start_offset < text_len:
        end_offset = min(text_len, start_offset + chunk_size)
        chunk_key = f"{asset_version_id}:{page_or_slide}:{start_offset}:{chunking_version}"
        chunk_hash = hashlib.sha256(chunk_key.encode()).hexdigest()[:16]
        chunks.append(
            {
                "chunk_id": f"{asset_version_id}_p{page_or_slide}_{chunk_hash}",
                "asset_version_id": asset_version_id,
                "page_or_slide": page_or_slide,
                "text": text[start_offset:end_offset],
                "start_offset": start_offset,
                "end_offset": end_offset,
                "extraction_method": extraction_method,
                "chunking_version": chunking_version,
            }
        )
        if end_offset >= text_len:
            break
        start_offset += step
    return chunks


def _open_pdf(doc_path: Path) -> tuple[str, Any]:
    import pymupdf

    method = f"pymupdf/{pymupdf.__version__}"
    try:
        doc = pymupdf.open(str(doc_path))  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise ParserError(f"Failed to open PDF document ({type(exc).__name__}).") from exc
    return method, doc


def extract_document(
    doc_path: Path,
    *,
    asset_version_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_version: str,
    limits: DocumentExtractionLimits,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract budgeted chunk records from a PDF or PPTX; return chunks and truncation reasons."""
    if not doc_path.is_file():
        raise ParserError("Document file not found.")

    budget = _Budget(limits)
    all_chunks: list[dict[str, Any]] = []

    def add(text: str, page_or_slide: int, method: str) -> None:
        records = chunk_text(
            budget.take_text(text),
            asset_version_id=asset_version_id,
            page_or_slide=page_or_slide,
            extraction_method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_version=chunking_version,
        )
        all_chunks.extend(budget.take_chunks(records))

    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        method, doc = _open_pdf(doc_path)
        try:
            if doc.is_encrypted and not doc.authenticate(""):
                raise ParserError("PDF document is password-protected or encrypted.")
            for page_index in range(len(doc)):
                if not budget.allow_page(page_index + 1):
                    break
                add(doc[page_index].get_text() or "", page_index + 1, method)
        finally:
            doc.close()
    elif suffix == ".pptx":
        from pptx import Presentation

        try:
            prs = Presentation(str(doc_path))
        except Exception as exc:
            raise ParserError(f"Failed to open PPTX presentation ({type(exc).__name__}).") from exc
        for slide_index, slide in enumerate(prs.slides):
            if not budget.allow_page(slide_index + 1):
                break
            slide_texts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = "".join(run.text for run in paragraph.runs).strip()
                        if text:
                            slide_texts.append(text)
                elif shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                            c_text = cell.text.strip()
                            if c_text:
                                slide_texts.append(c_text)
            add("\n".join(slide_texts).strip(), slide_index + 1, "python-pptx")
    else:
        raise ParserError(
            f"Unsupported document format '{suffix}'. Only .pdf and .pptx are supported."
        )
    return all_chunks, budget.reasons


def _apply_memory_limit(max_memory_mb: int | None) -> None:
    """Cap this process's address space where the OS supports it (POSIX ``resource``)."""
    if max_memory_mb is None:
        return
    try:
        import resource
    except ImportError:  # pragma: no cover - not available on Windows
        return
    setrlimit = getattr(resource, "setrlimit", None)
    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    if setrlimit is None or rlimit_as is None:  # pragma: no cover
        return
    limit_bytes = max_memory_mb * 1024 * 1024
    # The platform may refuse the limit; parsing still runs under the wall-clock deadline.
    with contextlib.suppress(ValueError, OSError):
        setrlimit(rlimit_as, (limit_bytes, limit_bytes))


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Handle one extraction request and return a JSON-serialisable result."""
    try:
        limits = DocumentExtractionLimits(**request["limits"])
        _apply_memory_limit(limits.max_memory_mb)
        chunks, reasons = extract_document(
            Path(request["doc_path"]),
            asset_version_id=request["asset_version_id"],
            chunk_size=request["chunk_size"],
            chunk_overlap=request["chunk_overlap"],
            chunking_version=request["chunking_version"],
            limits=limits,
        )
        return {"status": "ok", "chunks": chunks, "truncation_reasons": reasons}
    except MemoryError:
        return {
            "status": "error",
            "kind": "resource",
            "message": "Document parser exceeded its memory budget.",
        }
    except ParserError as exc:
        return {"status": "error", "kind": "extraction", "message": str(exc)}
    except Exception as exc:  # parser bugs: report the type only, never content
        return {
            "status": "error",
            "kind": "extraction",
            "message": f"Document parser failed ({type(exc).__name__}).",
        }


def main() -> int:
    request = json.loads(sys.stdin.read())
    result = run_request(request)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
