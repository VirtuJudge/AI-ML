"""Unit tests for document store and retrieval (app.document_store)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.document_store import (
    DocumentStore,
    FakeDocumentStore,
    PgVectorDocumentStore,
    _normalize_database_url,
    create_document_store,
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


def test_normalize_database_url_supabase() -> None:
    """Verify Supabase URL normalization and SSL enforcement."""
    raw_url = "postgresql+asyncpg://postgres:secret%2F123@db.dummyproject123.supabase.co:5432/postgres"
    clean_dsn, ssl_mode = _normalize_database_url(raw_url)

    assert clean_dsn.startswith("postgresql://")
    assert ssl_mode == "require"


def test_normalize_database_url_special_characters() -> None:
    """Verify passwords with special characters are safely escaped."""
    raw_url = r"postgresql+asyncpg://postgres:dummy\pw}*with&/chars@db.dummyproject123.supabase.co:5432/postgres"
    clean_dsn, ssl_mode = _normalize_database_url(raw_url)

    assert clean_dsn.startswith("postgresql://")
    assert ssl_mode == "require"
    assert "@db.dummyproject123.supabase.co:5432/postgres" in clean_dsn


@pytest.mark.asyncio
async def test_create_document_store_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_document_store falls back to FakeDocumentStore when no DATABASE_URL."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = await create_document_store()
    assert isinstance(store, FakeDocumentStore)


@pytest.mark.asyncio
async def test_pgvector_store_sql_and_vector_cast() -> None:
    """Verify PgVectorDocumentStore executes parameterised SQL with vector casts and query branching."""
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.fetch = AsyncMock(
        return_value=[
            {
                "chunk_id": "c1",
                "asset_version_id": "v1",
                "page_or_slide": 1,
                "text": "test text",
                "start_offset": 0,
                "end_offset": 9,
                "extraction_method": "pymupdf/1.0",
                "chunking_version": "v1",
                "embedding_model": "test-model",
                "embedding_dimensions": 3,
            }
        ]
    )

    # Context manager mock for conn.transaction()
    mock_conn.transaction.return_value.__aenter__ = AsyncMock()
    mock_conn.transaction.return_value.__aexit__ = AsyncMock()

    # Context manager mock for db_pool.acquire()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    store = PgVectorDocumentStore(mock_pool)

    # 1. Initialize schema
    await store.initialize_schema()
    mock_conn.execute.assert_called_with(store.TABLE_DDL)

    # 2. Store chunks with vector cast
    chunk = DocumentChunk(
        chunk_id="c1",
        asset_version_id="v1",
        page_or_slide=1,
        text="test text",
        extraction_method="pymupdf/1.0",
        chunking_version="v1",
        embedding=[0.1, 0.2, 0.3],
    )
    await store.store_chunks("session_01", [chunk])
    insert_call_args = mock_conn.execute.call_args[0]
    assert "CAST($12 AS vector)" in insert_call_args[0]
    assert insert_call_args[1] == "session_01"

    # 3. Retrieve top-k with query embedding (orders by vector distance)
    await store.retrieve_top_k("session_01", [0.1, 0.2, 0.3], k=5)
    query_call_args = mock_conn.fetch.call_args[0]
    assert "ORDER BY embedding <=> $3::vector" in query_call_args[0]

    # 4. Retrieve top-k without query embedding (does NOT crash, orders structurally)
    await store.retrieve_top_k("session_01", [], k=5)
    non_vec_call_args = mock_conn.fetch.call_args[0]
    assert "ORDER BY page_or_slide ASC, start_offset ASC NULLS LAST" in non_vec_call_args[0]
    assert "<=>" not in non_vec_call_args[0]


@pytest.mark.asyncio
async def test_known_question_retrieval_accuracy() -> None:
    """Acceptance Test: Known questions retrieve the correct page/slide."""
    store = FakeDocumentStore()
    session_id = "01JACCEPTANCE_SESSION_01"

    # Page 1: Financial projections
    # Page 2: Technical architecture
    chunks = [
        DocumentChunk(
            chunk_id="chunk_page_1_fin",
            asset_version_id="deck_v1",
            page_or_slide=1,
            text="Financial Projections: Projected ARR is 10M with 85% gross margin.",
            extraction_method="pymupdf/1.25.0",
            chunking_version="v1",
            embedding=[0.95, 0.05, 0.0],
        ),
        DocumentChunk(
            chunk_id="chunk_page_2_tech",
            asset_version_id="deck_v1",
            page_or_slide=2,
            text="Technical Architecture: Microservices, PostgreSQL pgvector, and WebRTC streaming.",
            extraction_method="pymupdf/1.25.0",
            chunking_version="v1",
            embedding=[0.05, 0.95, 0.0],
        ),
    ]
    await store.store_chunks(session_id, chunks)

    # Query 1: "What are your financial projections and margins?" (close to embedding [1.0, 0.0, 0.0])
    results_fin = await store.retrieve_top_k(session_id, [1.0, 0.0, 0.0], k=1)
    assert len(results_fin) == 1
    assert results_fin[0].page_or_slide == 1
    assert "Financial Projections" in results_fin[0].text

    # Query 2: "Explain your system architecture and vector storage." (close to embedding [0.0, 1.0, 0.0])
    results_tech = await store.retrieve_top_k(session_id, [0.0, 1.0, 0.0], k=1)
    assert len(results_tech) == 1
    assert results_tech[0].page_or_slide == 2
    assert "Technical Architecture" in results_tech[0].text

