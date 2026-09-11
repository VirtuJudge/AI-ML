"""Unit tests for document store and retrieval (app.document_store)."""

import pytest

from app.document_store import (
    DocumentStore,
    FakeDocumentStore,
)
from app.providers.types import DocumentChunk


def test_document_store_satisfies_protocol() -> None:
    """Verify FakeDocumentStore satisfies DocumentStore Protocol."""
    store: DocumentStore = FakeDocumentStore()
    assert hasattr(store, "store_chunks")
    assert hasattr(store, "retrieve_top_k")
    assert hasattr(store, "delete_by_session")


@pytest.mark.asyncio
async def test_store_and_retrieve_roundtrip() -> None:
    """Verify storing chunks and retrieving top-k returns stored data."""
    store = FakeDocumentStore()
    session_id = "01JSESSION00000000000000001"

    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            asset_version_id="version_01",
            page_or_slide=1,
            text="First slide content",
            extraction_method="pymupdf/1.25.0",
            chunking_version="v1",
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            asset_version_id="version_01",
            page_or_slide=2,
            text="Second slide content",
            extraction_method="pymupdf/1.25.0",
            chunking_version="v1",
        ),
    ]

    stored_count = await store.store_chunks(session_id, chunks)
    assert stored_count == 2

    results = await store.retrieve_top_k(session_id, query_embedding=[], k=5)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_01"
    assert results[1].chunk_id == "chunk_02"


@pytest.mark.asyncio
async def test_session_isolation() -> None:
    """Verify documents from another session/team cannot appear in retrieval results."""
    store = FakeDocumentStore()
    session_a = "01JSESSION_TEAM_A_00000001"
    session_b = "01JSESSION_TEAM_B_00000002"

    chunks_a = [
        DocumentChunk(
            chunk_id="team_a_secret_chunk",
            asset_version_id="ver_a",
            page_or_slide=1,
            text="Team A confidential IP strategy",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
        )
    ]
    chunks_b = [
        DocumentChunk(
            chunk_id="team_b_public_chunk",
            asset_version_id="ver_b",
            page_or_slide=1,
            text="Team B public pitch deck",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
        )
    ]

    await store.store_chunks(session_a, chunks_a)
    await store.store_chunks(session_b, chunks_b)

    # Querying Session B MUST NOT return Team A's chunks
    retrieved_b = await store.retrieve_top_k(session_b, query_embedding=[], k=10)
    assert len(retrieved_b) == 1
    assert retrieved_b[0].chunk_id == "team_b_public_chunk"
    assert not any(c.chunk_id == "team_a_secret_chunk" for c in retrieved_b)

    # Querying Session A MUST NOT return Team B's chunks
    retrieved_a = await store.retrieve_top_k(session_a, query_embedding=[], k=10)
    assert len(retrieved_a) == 1
    assert retrieved_a[0].chunk_id == "team_a_secret_chunk"
    assert not any(c.chunk_id == "team_b_public_chunk" for c in retrieved_a)


@pytest.mark.asyncio
async def test_version_filtering() -> None:
    """Verify retrieval can be filtered to specific asset_version_ids."""
    store = FakeDocumentStore()
    session_id = "01JSESSION00000000000000003"

    chunks = [
        DocumentChunk(
            chunk_id="v1_chunk",
            asset_version_id="asset_ver_1",
            page_or_slide=1,
            text="Old deprecated slide",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
        ),
        DocumentChunk(
            chunk_id="v2_chunk",
            asset_version_id="asset_ver_2",
            page_or_slide=1,
            text="Current updated slide",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
        ),
    ]

    await store.store_chunks(session_id, chunks)

    # Retrieve only asset_ver_2
    filtered_results = await store.retrieve_top_k(
        session_id, query_embedding=[], k=5, asset_version_ids=["asset_ver_2"]
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "v2_chunk"


@pytest.mark.asyncio
async def test_similarity_ranking() -> None:
    """Verify retrieval ranks chunks by cosine similarity when embeddings exist."""
    store = FakeDocumentStore()
    session_id = "01JSESSION00000000000000004"

    # Vectors of dimension 3
    # Query vector is [1.0, 0.0, 0.0]
    # chunk_close has embedding [0.9, 0.1, 0.0] -> high similarity
    # chunk_far has embedding [0.0, 0.0, 1.0] -> orthogonal (similarity 0.0)
    chunks = [
        DocumentChunk(
            chunk_id="chunk_far",
            asset_version_id="ver_01",
            page_or_slide=1,
            text="Far topic",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
            embedding=[0.0, 0.0, 1.0],
        ),
        DocumentChunk(
            chunk_id="chunk_close",
            asset_version_id="ver_01",
            page_or_slide=2,
            text="Close topic",
            extraction_method="pymupdf/1.0",
            chunking_version="v1",
            embedding=[0.9, 0.1, 0.0],
        ),
    ]

    await store.store_chunks(session_id, chunks)

    query_vec = [1.0, 0.0, 0.0]
    ranked = await store.retrieve_top_k(session_id, query_vec, k=2)

    assert len(ranked) == 2
    assert ranked[0].chunk_id == "chunk_close"
    assert ranked[1].chunk_id == "chunk_far"


@pytest.mark.asyncio
async def test_delete_by_session() -> None:
    """Verify delete_by_session removes only the targeted session's records."""
    store = FakeDocumentStore()
    session_a = "01JSESSION_DEL_A"
    session_b = "01JSESSION_DEL_B"

    chunk_a = DocumentChunk(
        chunk_id="ca",
        asset_version_id="v",
        page_or_slide=1,
        text="A",
        extraction_method="m",
        chunking_version="v1",
    )
    chunk_b = DocumentChunk(
        chunk_id="cb",
        asset_version_id="v",
        page_or_slide=1,
        text="B",
        extraction_method="m",
        chunking_version="v1",
    )

    await store.store_chunks(session_a, [chunk_a])
    await store.store_chunks(session_b, [chunk_b])

    deleted_count = await store.delete_by_session(session_a)
    assert deleted_count == 1

    assert await store.retrieve_top_k(session_a, query_embedding=[]) == []
    assert len(await store.retrieve_top_k(session_b, query_embedding=[])) == 1

