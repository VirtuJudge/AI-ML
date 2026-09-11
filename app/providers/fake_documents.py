"""Fake document ingestion and embedding providers."""

from pathlib import Path

from app.providers.types import DocumentChunk


class FakeDocumentProvider:
    """Deterministic document parser and chunk extractor."""

    async def extract_and_embed(
        self, doc_path: Path, *, asset_version_id: str = ""
    ) -> list[DocumentChunk]:
        resolved_version = asset_version_id or "01JEXAMPLE0000000000DOC01"
        return [
            DocumentChunk(
                chunk_id=f"{resolved_version}_p1_c1",
                asset_version_id=resolved_version,
                page_or_slide=1,
                text=(
                    "Executive Summary: VirtuJudge delivers realistic pitch "
                    "rehearsal with instant feedback."
                ),
                start_offset=0,
                end_offset=74,
                extraction_method="fake/1.0.0",
                chunking_version="v1_fake",
                embedding_model="fake-embedding-128d",
                embedding_dimensions=128,
                embedding=[0.01] * 128,
            ),
            DocumentChunk(
                chunk_id=f"{resolved_version}_p2_c1",
                asset_version_id=resolved_version,
                page_or_slide=2,
                text=(
                    "Financial Projections: Projected customer acquisition cost of $450 "
                    "with 85% gross margins."
                ),
                start_offset=0,
                end_offset=86,
                extraction_method="fake/1.0.0",
                chunking_version="v1_fake",
                embedding_model="fake-embedding-128d",
                embedding_dimensions=128,
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
