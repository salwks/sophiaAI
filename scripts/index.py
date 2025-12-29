#!/usr/bin/env python3
"""
Sophia AI: Indexing Script
============================
처리된 논문 인덱싱

Usage:
    uv run python scripts/index.py --full
    uv run python scripts/index.py --incremental
    uv run python scripts/index.py --stats
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.indexer import main

if __name__ == "__main__":
    main()
