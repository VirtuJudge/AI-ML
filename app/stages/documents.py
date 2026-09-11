"""Document extraction stage for VirtuJudge AI-ML.

Orchestrates parsing of PDF and PPTX supporting documents into versioned chunks
with provenance, generating safe limitations for missing or malformed files,
and guaranteeing that extracted document text is never written to normal logs.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.contracts import AssetInput, Limitation
from app.providers.base import DocumentProvider
from app.providers.types import DocumentChunk

logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".pptx"})


class DocumentStageResult(BaseModel):
    """Result of supporting document extraction stage."""

    chunks: list[DocumentChunk] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def run_document_stage(
    documents: list[AssetInput],
    document_provider: DocumentProvider,
    *,
    practice_session_id: str = "",
) -> DocumentStageResult:
    """Run the supporting documents extraction stage.

    Args:
        documents: List of supporting document asset inputs.
        document_provider: DocumentProvider adapter (real or fake).
        practice_session_id: Optional parent session ID for correlation.

    Returns:
        DocumentStageResult containing all extracted chunks, limitations, and operational metadata.
    """
    if not documents:
        # A session with no supporting documents can still complete normally without limitations
        return DocumentStageResult(
            chunks=[],
            limitations=[],
            metadata={
                "stage": "documents",
                "provider": document_provider.__class__.__name__,
                "document_count": 0,
                "chunk_count": 0,
            },
        )

    all_chunks: list[DocumentChunk] = []
    limitations: list[Limitation] = []

    for doc in documents:
        doc_path = Path(doc.object_key)
        artifact_id = doc.artifact_id or doc_path.name
        suffix = doc_path.suffix.lower()

        if not doc_path.is_file():
            logger.warning("Supporting document file not found: artifact_id=%s", artifact_id)
            limitations.append(
                Limitation(
                    code="document_file_missing",
                    scope="documents",
                    message=f"Supporting document '{artifact_id}' file was not found.",
                    affected_dimensions=["market_and_business_model", "technology_and_moat"],
                )
            )
            continue

        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            logger.warning(
                "Unsupported document format '%s' for artifact_id=%s", suffix, artifact_id
            )
            limitations.append(
                Limitation(
                    code="unsupported_document_format",
                    scope="documents",
                    message=(
                        f"Supporting document '{artifact_id}' format '{suffix}' is not supported. "
                        "Only PDF and PPTX documents are accepted."
                    ),
                    affected_dimensions=["market_and_business_model", "technology_and_moat"],
                )
            )
            continue

        try:
            chunks = await document_provider.extract_and_embed(
                doc_path, asset_version_id=artifact_id
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            err_msg = str(exc).lower()
            logger.warning("Document extraction failed for artifact_id=%s", artifact_id)
            if "encrypted" in err_msg or "password" in err_msg:
                code = "document_malformed"
                msg = f"Supporting document '{artifact_id}' is encrypted or password-protected."
            elif "unsupported" in err_msg:
                code = "unsupported_document_format"
                msg = f"Supporting document '{artifact_id}' has an unsupported internal format."
            else:
                code = "document_extraction_failed"
                msg = f"Failed to extract content from supporting document '{artifact_id}'."

            limitations.append(
                Limitation(
                    code=code,
                    scope="documents",
                    message=msg,
                    affected_dimensions=["market_and_business_model", "technology_and_moat"],
                )
            )

    # Note: extracted chunk text is strictly excluded from log messages for privacy and security
    logger.debug(
        "Document extraction completed: processed_docs=%d total_chunks=%d limitations=%d",
        len(documents),
        len(all_chunks),
        len(limitations),
    )

    metadata: dict[str, Any] = {
        "stage": "documents",
        "provider": document_provider.__class__.__name__,
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
    }
    if practice_session_id:
        metadata["practice_session_id"] = practice_session_id

    return DocumentStageResult(
        chunks=all_chunks,
        limitations=limitations,
        metadata=metadata,
    )


__all__ = [
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "DocumentStageResult",
    "run_document_stage",
]

