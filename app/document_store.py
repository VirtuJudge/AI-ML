"""Document chunk storage and vector retrieval for VirtuJudge AI-ML.

Manages AI-owned pgvector tables and deterministic in-memory stores for document chunks.
Guarantees session and asset-version isolation so documents from another session or version
cannot appear in retrieval results.
"""

import math
import os
import urllib.parse
from typing import Any, Protocol

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

from app.providers.types import DocumentChunk


class DocumentStore(Protocol):
    """Protocol for document chunk vector storage and retrieval."""

    async def store_chunks(self, session_id: str, chunks: list[DocumentChunk]) -> int:
        """Store extracted document chunks associated with a practice session."""
        ...

    async def retrieve_top_k(
        self,
        session_id: str,
        query_embedding: list[float],
        *,
        k: int = 5,
        asset_version_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Retrieve top-k most relevant chunks strictly filtered by session and asset versions."""
        ...

    async def delete_by_session(self, session_id: str) -> int:
        """Delete all stored chunks for a given practice session."""
        ...


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vector lists."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeDocumentStore:
    """Deterministic in-memory document chunk and vector store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[DocumentChunk]] = {}

    async def store_chunks(self, session_id: str, chunks: list[DocumentChunk]) -> int:
        """Store chunks strictly scoped to session_id."""
        if not session_id:
            raise ValueError("session_id cannot be empty")
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].extend(chunks)
        return len(chunks)

    async def retrieve_top_k(
        self,
        session_id: str,
        query_embedding: list[float],
        *,
        k: int = 5,
        asset_version_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Retrieve top-k chunks filtered strictly by session_id and asset_version_ids."""
        session_chunks = self._store.get(session_id, [])

        if asset_version_ids is not None:
            allowed_versions = set(asset_version_ids)
            filtered = [c for c in session_chunks if c.asset_version_id in allowed_versions]
        else:
            filtered = list(session_chunks)

        if not filtered:
            return []

        # If chunks have embeddings, rank by cosine similarity
        has_embeddings = any(c.embedding is not None for c in filtered)
        if has_embeddings and query_embedding:
            ranked = sorted(
                filtered,
                key=lambda c: (
                    _cosine_similarity(c.embedding, query_embedding)
                    if c.embedding is not None
                    else -1.0
                ),
                reverse=True,
            )
            return ranked[:k]

        return filtered[:k]

    async def delete_by_session(self, session_id: str) -> int:
        """Delete all chunks for a session and return count of deleted items."""
        chunks = self._store.pop(session_id, [])
        return len(chunks)


class PgVectorDocumentStore:
    """PostgreSQL + pgvector storage implementation for AI-owned document chunks."""

    TABLE_DDL = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS ai_document_chunks (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        asset_version_id TEXT NOT NULL,
        page_or_slide INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_offset INTEGER,
        end_offset INTEGER,
        extraction_method TEXT NOT NULL,
        chunking_version TEXT NOT NULL,
        embedding_model TEXT,
        embedding_dimensions INTEGER,
        embedding vector,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(session_id, chunk_id)
    );

    CREATE INDEX IF NOT EXISTS idx_ai_doc_chunks_session ON ai_document_chunks(session_id);
    CREATE INDEX IF NOT EXISTS idx_ai_doc_chunks_version ON ai_document_chunks(asset_version_id);
    """

    def __init__(self, db_pool: Any = None) -> None:
        self.db_pool = db_pool

    async def initialize_schema(self) -> None:
        """Execute DDL to ensure table and indices exist."""
        if self.db_pool is None:
            raise RuntimeError("Database connection pool is not configured.")
        async with self.db_pool.acquire() as conn:
            await conn.execute(self.TABLE_DDL)

    async def store_chunks(self, session_id: str, chunks: list[DocumentChunk]) -> int:
        if self.db_pool is None:
            raise RuntimeError("Database connection pool is not configured.")
        if not chunks:
            return 0

        insert_sql = """
        INSERT INTO ai_document_chunks (
            session_id, chunk_id, asset_version_id, page_or_slide, text,
            start_offset, end_offset, extraction_method, chunking_version,
            embedding_model, embedding_dimensions, embedding
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CAST($12 AS vector))
        ON CONFLICT (session_id, chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding;
        """
        async with self.db_pool.acquire() as conn, conn.transaction():
            for c in chunks:
                await conn.execute(
                    insert_sql,
                    session_id,
                    c.chunk_id,
                    c.asset_version_id,
                    c.page_or_slide,
                    c.text,
                    c.start_offset,
                    c.end_offset,
                    c.extraction_method,
                    c.chunking_version,
                    c.embedding_model,
                    c.embedding_dimensions,
                    str(c.embedding) if c.embedding else None,
                )
        return len(chunks)

    async def retrieve_top_k(
        self,
        session_id: str,
        query_embedding: list[float],
        *,
        k: int = 5,
        asset_version_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        if self.db_pool is None:
            raise RuntimeError("Database connection pool is not configured.")

        if not query_embedding:
            query_sql = """
            SELECT chunk_id, asset_version_id, page_or_slide, text,
                   start_offset, end_offset, extraction_method, chunking_version,
                   embedding_model, embedding_dimensions
            FROM ai_document_chunks
            WHERE session_id = $1
              AND ($2::text[] IS NULL OR asset_version_id = ANY($2))
            ORDER BY page_or_slide ASC, start_offset ASC NULLS LAST
            LIMIT $3;
            """
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    query_sql,
                    session_id,
                    asset_version_ids,
                    k,
                )
        else:
            query_sql = """
            SELECT chunk_id, asset_version_id, page_or_slide, text,
                   start_offset, end_offset, extraction_method, chunking_version,
                   embedding_model, embedding_dimensions
            FROM ai_document_chunks
            WHERE session_id = $1
              AND ($2::text[] IS NULL OR asset_version_id = ANY($2))
            ORDER BY embedding <=> $3::vector
            LIMIT $4;
            """
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    query_sql,
                    session_id,
                    asset_version_ids,
                    str(query_embedding),
                    k,
                )

        return [
            DocumentChunk(
                chunk_id=r["chunk_id"],
                asset_version_id=r["asset_version_id"],
                page_or_slide=r["page_or_slide"],
                text=r["text"],
                start_offset=r["start_offset"],
                end_offset=r["end_offset"],
                extraction_method=r["extraction_method"],
                chunking_version=r["chunking_version"],
                embedding_model=r["embedding_model"],
                embedding_dimensions=r["embedding_dimensions"],
            )
            for r in rows
        ]

    async def delete_by_session(self, session_id: str) -> int:
        if self.db_pool is None:
            raise RuntimeError("Database connection pool is not configured.")
        async with self.db_pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM ai_document_chunks WHERE session_id = $1;", session_id
            )
            parts = res.split()
            return int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0


def _normalize_database_url(url: str) -> tuple[str, str | None]:
    """Normalize SQLAlchemy / Supabase connection URLs for asyncpg.

    Handles:
    - Stripping '+asyncpg' driver dialect prefix.
    - Enabling SSL for Supabase ('supabase.co') or when sslmode=require.
    - Handling unescaped special characters in credentials.
    """
    clean = url.replace("postgresql+asyncpg://", "postgresql://")
    ssl_mode = "require" if ("supabase" in clean or "sslmode=require" in clean) else None

    # Handle URL parsing and password quoting if necessary
    try:
        parsed = urllib.parse.urlparse(clean)
        if parsed.password and ("@" in parsed.password or "/" in parsed.password or "\\" in parsed.password):
            quoted_pw = urllib.parse.quote_plus(parsed.password)
            user_part = f"{parsed.username}:{quoted_pw}" if parsed.username else quoted_pw
            netloc = f"{user_part}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            clean = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass

    return clean, ssl_mode


async def create_document_store(database_url: str | None = None) -> DocumentStore:
    """Create a PgVectorDocumentStore if database_url is configured, otherwise FakeDocumentStore.

    Supports Supabase and PostgreSQL pgvector tables out-of-the-box:
    - Auto-initializes vector extension and ai_document_chunks table.
    - Uses connection pooling with SSL encryption for hosted databases.
    """
    db_url = database_url or os.getenv("DATABASE_URL")
    if not db_url:
        return FakeDocumentStore()

    if asyncpg is None:
        raise RuntimeError("asyncpg is required for PgVectorDocumentStore but is not installed.")

    clean_dsn, ssl_mode = _normalize_database_url(db_url)
    pool = await asyncpg.create_pool(
        clean_dsn,
        ssl=ssl_mode,
        statement_cache_size=0,
    )
    store = PgVectorDocumentStore(pool)
    await store.initialize_schema()
    return store


__all__ = [
    "DocumentStore",
    "FakeDocumentStore",
    "PgVectorDocumentStore",
    "create_document_store",
]
