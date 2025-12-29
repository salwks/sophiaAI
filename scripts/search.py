#!/usr/bin/env python3
"""
MARIA-Mammo: Search Script
==========================
논문 검색 CLI

Usage:
    uv run python scripts/search.py "DBT vs FFDM"
    uv run python scripts/search.py -i  # Interactive mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.engine import main

if __name__ == "__main__":
    main()
