"""Subprocess bridge exercising the real AI consumer from backend acceptance tests."""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from app.contracts import QueueMessage
from app.document_store import FakeDocumentStore, PgVectorDocumentStore
from app.pipeline import FakePipeline
from app.providers.types import DocumentChunk
from app.storage.local import LocalDiskObjectStorage
from app.worker import process_job


async def main() -> None:
    message = QueueMessage.model_validate(json.loads(sys.stdin.read()))
    storage = LocalDiskObjectStorage(Path(sys.argv[1]))
    target = message.payload["practice_session_ids"][0]
    for key in ("analysis.json", "checkpoints/speech.json", "report.md", "evaluation.json"):
        await storage.upload_json(f"ai/session/{target}/{key}", {"synthetic": True})
    await storage.upload_json("ai/session/unrelated/report.md", {"synthetic": True})
    pool = None
    schema = "erasure_bridge_" + uuid4().hex
    url = os.environ.get("ERASURE_TEST_AI_DATABASE_URL")
    if url:
        import asyncpg

        pool = await asyncpg.create_pool(url, min_size=1, max_size=1)
        async with pool.acquire() as connection:
            await connection.execute(f'CREATE SCHEMA "{schema}"')
            await connection.execute(f'SET search_path TO "{schema}"')
            # The production DELETE seam does not require loading embeddings.
            await connection.execute(
                "CREATE TABLE ai_document_chunks (session_id text, asset_version_id text)"
            )
            await connection.execute(
                "INSERT INTO ai_document_chunks VALUES ($1, 'version'), ('unrelated', 'other')",
                target,
            )
        await pool.close()
        pool = await asyncpg.create_pool(
            url, min_size=1, max_size=1, server_settings={"search_path": schema}
        )
        documents = PgVectorDocumentStore(pool)
    else:
        documents = FakeDocumentStore()
        await documents.store_chunks(
            target,
            [
                DocumentChunk(
                    chunk_id="chunk", asset_version_id="version", page_or_slide=1, text="synthetic"
                )
            ],
        )
        await documents.store_chunks(
            "unrelated",
            [
                DocumentChunk(
                    chunk_id="other", asset_version_id="other", page_or_slide=1, text="synthetic"
                )
            ],
        )
    try:
        result = await process_job(
            message, FakePipeline(object_storage=storage, document_store=documents)
        )
        if pool:
            async with pool.acquire() as connection:
                assert (
                    await connection.fetchval(
                        "SELECT count(*) FROM ai_document_chunks WHERE session_id = $1", target
                    )
                    == 0
                )
                assert (
                    await connection.fetchval(
                        "SELECT count(*) FROM ai_document_chunks WHERE session_id = 'unrelated'"
                    )
                    == 1
                )
        else:
            assert await documents.retrieve_top_k(target, []) == []
            assert await documents.retrieve_top_k("unrelated", [])
        assert await storage.list_objects(f"ai/session/{target}/") == []
        assert await storage.list_objects("ai/session/unrelated/")
        print(result.model_dump_json())
    finally:
        if pool:
            async with pool.acquire() as connection:
                await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
