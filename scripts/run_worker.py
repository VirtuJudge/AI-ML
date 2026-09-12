"""Runner script for the AI Worker daemon.

Usage:
    python scripts/run_worker.py
"""

import asyncio

from app.__main__ import main

if __name__ == "__main__":
    asyncio.run(main())
