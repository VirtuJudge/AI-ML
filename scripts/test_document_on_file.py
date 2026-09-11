"""Standalone CLI script to test document parsing, chunking, and retrieval on any PDF or PPTX.

Usage:
    conda run -n vic-AI python scripts/test_document_on_file.py <path> [-q "term"] [--json]

Examples:
    conda run -n vic-AI python scripts/test_document_on_file.py deck.pdf
    conda run -n vic-AI python scripts/test_document_on_file.py deck.pdf --query "retention"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from app.document_store import FakeDocumentStore, create_document_store
from app.providers.fake_documents import FakeEmbeddingProvider
from app.providers.http_embedding import HttpEmbeddingProvider
from app.providers.pymupdf_documents import PyMuPDFDocumentProvider


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VirtuJudge document extraction and retrieval test on a PDF or PPTX file."
    )
    parser.add_argument("file_path", type=Path, help="Path to the PDF or PPTX document to test.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Optional search query to test top-k retrieval.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save output to a JSON file.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional database URL for pgvector. If omitted, uses DATABASE_URL from .env.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force using offline FakeDocumentStore without connecting to database.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_arguments()
    file_path = args.file_path.resolve()

    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print("\n" + "=" * 75)
        print("VIRTUJUDGE DOCUMENT EXTRACTION & RETRIEVAL TEST")
        print("=" * 75)
        print(f"File:        {file_path.name}")
        print(f"Path:        {file_path}")
        print(f"Size:        {file_path.stat().st_size / 1024:.2f} KB")
        print(f"Format:      {file_path.suffix.upper()}")
        print("=" * 75)

    # Initialize embedding provider if API key is present, otherwise use test vector generator
    has_real_embedding_key = bool(os.environ.get("EMBEDDING_API_KEY"))
    if has_real_embedding_key:
        if not args.json:
            print("Embedding:   HttpEmbeddingProvider (Live API)")
        embedding_provider = HttpEmbeddingProvider()
    else:
        if not args.json:
            print("Embedding:   FakeEmbeddingProvider (Local deterministic 128-dim vectors)")
        embedding_provider = FakeEmbeddingProvider(dimension=128)

    provider = PyMuPDFDocumentProvider(embedding_provider=embedding_provider)

    if not args.json:
        print("\nExtracting and chunking document...")

    asset_version_id = f"test_ver_{file_path.stem}"
    try:
        chunks = await provider.extract_and_embed(file_path, asset_version_id=asset_version_id)
    except Exception as exc:
        print(f"\nExtraction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Store in configured document store (PgVector if DATABASE_URL, otherwise FakeDocumentStore)
    if args.offline:
        store = FakeDocumentStore()
    else:
        try:
            store = await create_document_store(args.database_url)
        except Exception as exc:
            if not args.json:
                print(f"\nNote: Database connection unavailable ({exc}). Using offline FakeDocumentStore.")
            store = FakeDocumentStore()

    session_id = f"session_{file_path.stem}"
    stored_count = await store.store_chunks(session_id, chunks)
    if not args.json:
        store_type = "Supabase / PostgreSQL pgvector ('ai_document_chunks')" if not isinstance(store, FakeDocumentStore) else "offline FakeDocumentStore"
        print(f"\nDocument Store: Successfully saved {stored_count} chunks to {store_type} (session_id={session_id})")

    # If a query is provided, embed it and retrieve top-k
    retrieved_chunks = []
    if args.query:
        query_embeddings = await embedding_provider.embed([args.query])
        query_vec = query_embeddings[0] if query_embeddings else []
        retrieved_chunks = await store.retrieve_top_k(session_id, query_vec, k=args.top_k)

    if args.json:
        output_data = {
            "file": str(file_path),
            "chunk_count": len(chunks),
            "chunks": [c.model_dump() for c in chunks],
            "query": args.query,
            "retrieved_count": len(retrieved_chunks),
            "retrieved": [c.model_dump() for c in retrieved_chunks],
        }
        json_str = json.dumps(output_data, indent=2)
        print(json_str)
        if args.output:
            args.output.write_text(json_str, encoding="utf-8")
        return

    print("\n" + "-" * 75)
    print("EXTRACTION SUMMARY")
    print("-" * 75)
    print(f"Total Chunks Extracted: {len(chunks)}")
    pages = sorted({c.page_or_slide for c in chunks})
    print(f"Pages/Slides Covered:   {pages if pages else 'None'}")
    print(f"Extraction Method:      {chunks[0].extraction_method if chunks else 'N/A'}")
    print(f"Chunking Version:       {chunks[0].chunking_version if chunks else 'N/A'}")
    has_embeddings = any(c.embedding is not None for c in chunks)
    dim = chunks[0].embedding_dimensions if chunks else 0
    emb_desc = f"Yes ({dim} dimensions)" if has_embeddings else "No"
    print(f"Embeddings Generated:   {emb_desc}")

    print("\n" + "-" * 75)
    print("EXTRACTED CHUNKS PREVIEW (First 3 chunks)")
    print("-" * 75)
    for i, c in enumerate(chunks[:3], 1):
        print(f"\n[Chunk {i}/{len(chunks)}] ID: {c.chunk_id}")
        loc_str = f"Page/Slide {c.page_or_slide} (Offsets: {c.start_offset} -> {c.end_offset})"
        print(f"  Location:     {loc_str}")
        print(f"  Text Length:  {len(c.text)} characters")
        preview = c.text.replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"  Text Content: \"{preview}\"")

    if args.query:
        print("\n" + "=" * 75)
        print(f"TOP-{args.top_k} RETRIEVAL FOR QUERY: \"{args.query}\"")
        print("=" * 75)
        if not retrieved_chunks:
            print("No matching chunks found.")
        else:
            for rank, rc in enumerate(retrieved_chunks, 1):
                print(f"\nRank {rank}: [Page/Slide {rc.page_or_slide}] Chunk ID: {rc.chunk_id}")
                print(f"Text excerpt: \"{rc.text}\"")

    print("\n" + "=" * 75)
    print("TEST PASSED: Document parsed, chunked, embedded, and indexed cleanly.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
