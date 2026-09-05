"""Fake document ingestion and embedding providers."""

from pathlib import Path

from app.providers.types import DocumentChunk


class FakeDocumentProvider:
    """Deterministic document parser and chunk extractor."""

    async def extract_and_embed(self, doc_path: Path) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id="chunk_doc_01",
                page_number=1,
                text=(
                    "Executive Summary: VirtuJudge delivers realistic pitch "
                    "rehearsal with instant feedback."
                ),
                embedding=[0.01] * 128,
            ),
            DocumentChunk(
                chunk_id="chunk_doc_02",
                page_number=2,
                text=(
                    "Financial Projections: Projected customer acquisition cost of $450 "
                    "with 85% gross margins."
                ),
                embedding=[0.02] * 128,
            ),
        ]


class FakeEmbeddingProvider:
    """Deterministic text embedding generator."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.05 * (i + 1)] * self.dimension for i in range(len(texts))]


__all__ = ["FakeDocumentProvider", "FakeEmbeddingProvider"]
