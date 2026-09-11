"""Smoke test: verify we can connect to the Supabase database from the AI-ML service."""

import asyncio
import os
import sys
from pathlib import Path

# Load .env directly to ensure environment variables are present
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.is_file():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.document_store import _normalize_database_url


async def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL is not set in .env or environment.")
        sys.exit(1)

    print(f"Raw DATABASE_URL: {db_url[:30]}...{db_url[-20:]}")

    clean_dsn, ssl_mode = _normalize_database_url(db_url)
    print(f"Normalized DSN:   {clean_dsn[:30]}...{clean_dsn[-20:]}")
    print(f"SSL mode:         {ssl_mode}")

    try:
        import asyncpg
    except ImportError:
        print("ERROR: asyncpg is not installed. Run: pip install asyncpg")
        sys.exit(1)

    print("\nConnecting to Supabase PostgreSQL...")
    try:
        conn = await asyncpg.connect(clean_dsn, ssl=ssl_mode)
        version = await conn.fetchval("SELECT version();")
        print("SUCCESS: Connected to PostgreSQL")
        print(f"  Server version: {version}")

        # Quick check: can we see tables?
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 10;"
        )
        print(f"  Public tables ({len(tables)}): {[r['tablename'] for r in tables]}")

        # Check if pgvector extension is available
        ext = await conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'vector';"
        )
        print(f"  pgvector installed: {bool(ext)}")

        await conn.close()
        print("\nConnection closed successfully. All good!")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
